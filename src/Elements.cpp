#include "Elements.h"

using hnsw::Elements;

static auto get_level_probabilities(const hnsw::IndexParams &params) {
  std::vector<double> probabilities;
  auto mL = 1. / log(params.M);
  for (int level = 0;; level++) {
    double p = exp(-level / mL) * (1. - exp(-1. / mL));
    if (p < 1e-12) {
      break;
    } else {
      probabilities.push_back(p);
    }
  }
  return probabilities;
}


Elements *Elements::Create(const hnsw::IndexParams &params, size_t block_size_bytes) {
  auto get_total_estimated_n_blocks = [](const LayerState *levels, int n_levels) {
    size_t n_blocks = 0;
    for (int i = 0; i < n_levels; i++) {
      n_blocks += levels[i].EstimatedTotalBlocks;
    }
    return n_blocks;
  };
  // Initializing layer_states based the calculated LayerState probabilities, no
  // memory allocation yet, we need to first calculate statistics for
  // all layer_states to get estimate for the total block/memory consumption
  auto level_probabilities = get_level_probabilities(params);
  int max_levels = (int) level_probabilities.size();
  auto *layer_states = (LayerState *) malloc(sizeof(LayerState) * max_levels);
  if (layer_states == nullptr) {
    return nullptr;
  }

  auto max_links_per_non_zero_layer = params.M;
  auto max_links_per_zero_layer = params.M * 2;
  auto data_size_bytes = sizeof(float_t) * params.Dimension;
  auto links_size_bytes_non_zero_layer = sizeof(Link) * max_links_per_non_zero_layer;
  auto links_size_bytes_zero_layer = sizeof(Link) * max_links_per_zero_layer;
  for (int level = 0; level < max_levels; level++) {
    auto p = level_probabilities[level];
    auto total_links_size_bytes = links_size_bytes_zero_layer + level * links_size_bytes_non_zero_layer;
    auto bytes_per_element = sizeof(Elements::ElementHeader) + total_links_size_bytes + data_size_bytes;
    auto estimated_total_elements =
        (uint32_t) (p * params.MaxElems);
    auto estimated_total_bytes = estimated_total_elements * bytes_per_element;
    auto estimated_total_blocks =
        estimated_total_bytes == 0 ? 0 : estimated_total_bytes / block_size_bytes +
                                         (estimated_total_bytes % block_size_bytes > 0 ? 1 : 0);
    LayerState layer_state{
        .Level = level,
        .Probability = p,
        .BytesPerElement = bytes_per_element,
        .EstimatedTotalBytes = estimated_total_bytes,
        .EstimatedTotalElements = estimated_total_elements,
        .EstimatedTotalBlocks = estimated_total_blocks,
        .BlockPtr = nullptr,
        .BlockFreeBytes = 0,
        .DeleteListHead = nullptr,
        .DeleteListTail = nullptr,
        .API = nullptr
    };
    memcpy(layer_states + level, &layer_state, sizeof(LayerState));
  }


  // Allocate lookup table for element memory address lookups by element id,
  // reserving id = 0 for "NoElement"
  auto **lookup = (void **) malloc((params.MaxElems + 1) * sizeof(void *));

  // Allocate memory for each LayerState + 25% extra in case some layer(s) require more than expected
  // Using single block so that we can switch the implementation to mmap in future
  const auto n_blocks = size_t((double) get_total_estimated_n_blocks(layer_states, max_levels) * 1.25);
  auto n_free_blocks = n_blocks;
  auto *blocks = (uint8_t *) malloc(n_blocks * block_size_bytes);
  if (blocks == nullptr) {
    free(layer_states);
    return nullptr;
  }
  // Prepare each layer before use
  for (int level = 0; level < max_levels; level++) {
    // Allocate at least the estimated number of required blocks for each layer
    auto *layer_state = layer_states + level;
    n_free_blocks -= layer_state->EstimatedTotalBlocks;
    layer_state->BlockPtr = blocks + (n_free_blocks * block_size_bytes);
    layer_state->BlockFreeBytes = layer_state->EstimatedTotalBlocks * block_size_bytes;

    // And also initialize public "layer API" instance with pre-calculated
    // constants
    uint32_t link_bytes_before_this_level =
        level == 0 ? 0 : links_size_bytes_zero_layer + level * links_size_bytes_non_zero_layer;
    uint32_t links_offset = sizeof(ElementHeader) + data_size_bytes + link_bytes_before_this_level;
    int max_links = int(level == 0 ? max_links_per_zero_layer : max_links_per_non_zero_layer);

    layer_state->API = new Layer(lookup, level, links_offset, max_links);
  }

  return new Elements(
      params,
      layer_states,
      max_levels,
      lookup,
      blocks,
      block_size_bytes,
      n_free_blocks);
}

Elements::Elements(
    const IndexParams &params,
    LayerState *layer_states,
    int n_layers,
    void **lookup,
    uint8_t *blocks,
    size_t block_size_bytes,
    size_t n_free_blocks)
    : params_(params),
      rnd_(1337), // NOLINT
      n_levels_(n_layers),
      layers_states_(layer_states),
      blocks_(blocks),
      block_size_bytes_(block_size_bytes),
      n_free_blocks_(n_free_blocks),
      delete_list_head_(nullptr),
      next_elem_id_(1),
      lookup_(lookup) {
}


Elements::~Elements() {
  for (int level = 0; level < n_levels_; level++) {
    delete layers_states_[level].API;
  }
  free(layers_states_);
  free(blocks_);
  free(lookup_);
};

hnsw::element_id_t Elements::AllocateNextElement(uint32_t external_id, const float_t *data) {
  if (delete_list_head_ != nullptr) {
    auto id = delete_list_head_->DeletedElementId;
    auto level_idx = delete_list_head_->DeletedElementLevel;
    auto next_id = delete_list_head_->NextDeletedElementId;
    auto &layer = layers_states_[level_idx];
    if (layer.DeleteListHead == layer.DeleteListTail) {
      layer.DeleteListTail = nullptr;
    }
    layer.DeleteListHead = nullptr;
    if (next_id != NoElement) {
      auto *next_deleted = (DeleteListNode *) lookup_[next_id];
      next_deleted->PrevDeletedElementId = NoElement;
      delete_list_head_ = next_deleted;
      layers_states_[next_deleted->DeletedElementLevel].DeleteListHead = next_deleted;
    } else {
      delete_list_head_ = nullptr;
    }
    initialize_element(lookup_[id], layer, external_id, data, params_.Dimension);
    return id;
  }

  auto &layer = select_next_random_layer();
  if (layer.BlockFreeBytes < layer.BytesPerElement) {
    if (n_free_blocks_ == 0) {
      return NoElement;
    }
    n_free_blocks_ -= 1;
    layer.BlockPtr = blocks_ + (n_free_blocks_ * block_size_bytes_);
    layer.BlockFreeBytes = block_size_bytes_;
  }
  auto id = next_elem_id_++;
  auto *ptr = layer.BlockPtr;
  layer.BlockPtr += layer.BytesPerElement;
  layer.BlockFreeBytes -= layer.BytesPerElement;
  lookup_[id] = ptr;
  initialize_element(ptr, layer, external_id, data, params_.Dimension);
  return id;
}

void Elements::FreeElement(element_id_t id) {
  int level = GetElementLevel(id);
  auto &layer = layers_states_[level];
  // First find next and previous element from the global delete list
  DeleteListNode *next = nullptr;
  DeleteListNode *prev = nullptr;
  if (layer.DeleteListHead != nullptr) {
    // Easy case: layer already have deletions, so next element is the current
    // head of layer deletion list and previous can be found by following the list
    next = layer.DeleteListHead;
    prev = next->PrevDeletedElementId != NoElement ? (DeleteListNode *) lookup_[next->PrevDeletedElementId] : nullptr;
  } else {
    // LayerState does not deletions yet, need to find previous and next by traversing
    // the levels (higher levels are always placed before so trying to find next
    // from lower levels and prev from higher ones)
    for (int i = level - 1; next == nullptr && i >= 0; i--) {
      next = layers_states_[i].DeleteListHead;
    }
    for (int i = level + 1; next == nullptr && i < n_levels_; i++) {
      prev = layers_states_[i].DeleteListTail;
    }
  }

  // Create delete list node by reusing actual element get_container - it's not used
  // so just rewrite the element ptr (+ some vector get_container) with delete
  // list node metadata
  auto *node = (DeleteListNode *) lookup_[id];
  node->DeletedElementId = id;
  node->DeletedElementLevel = level;
  node->NextDeletedElementId = next != nullptr ? next->DeletedElementId : NoElement;
  node->PrevDeletedElementId = prev != nullptr ? prev->DeletedElementId : NoElement;
  // Fix linking
  if (next != nullptr) {
    next->PrevDeletedElementId = id;
  }
  if (prev != nullptr) {
    prev->NextDeletedElementId = id;
  }
  if (layer.DeleteListHead == layer.DeleteListTail) {
    layer.DeleteListTail = node;
  }
  if (layer.DeleteListHead == delete_list_head_) {
    delete_list_head_ = node;
  }
  // Finally, mark the new delete list node as the head of the
  // original element layer
  layer.DeleteListHead = node;
}

int hnsw::Elements::GetElementLevel(hnsw::element_id_t id) {
  return ((ElementHeader *) lookup_[id])->Level;
}

Elements::Layer *Elements::GetLayerAt(int level) {
  return layers_states_[level].API;
}

Elements::LayerState &Elements::select_next_random_layer() {
  double f = rnd_() / double(std::mt19937::max());
  for (int i = 0; i < n_levels_; i++) {
    auto p = layers_states_[i].Probability;
    if (f < p) {
      return layers_states_[i];
    }
    f -= p;
  }
  return layers_states_[n_levels_ - 1];
}

void Elements::initialize_element(
    void *ptr,
    const LayerState &layer,
    uint32_t external_id,
    const float_t *data,
    uint32_t dimension) {
  // Set header
  auto *header = (ElementHeader *) ptr;
  header->ExternalId = external_id;
  header->Level = layer.Level;
  // Set data
  memcpy(((uint8_t *) ptr) + sizeof(ElementHeader), data, dimension * sizeof(float_t));
  // Set links
  auto header_plus_data_bytes = sizeof(ElementHeader) + dimension * sizeof(float_t);
  auto link_bytes = layer.BytesPerElement - header_plus_data_bytes;
  memset(((uint8_t *) ptr) + header_plus_data_bytes, NoElement, link_bytes);
}

hnsw::Link *Elements::Layer::GetNeighbors(element_id_t id) {
  return (Link * )(((uint8_t *) lookup_[id]) + links_offset_);
}

float_t *Elements::Layer::GetData(element_id_t id) {
  return (float_t *) (((uint8_t *) lookup_[id]) + sizeof(ElementHeader));
}

void Elements::Layer::ConnectElements(element_id_t from, element_id_t to, hnsw::Link &via) {
  if (via.OutgoingId != NoElement) {
    // TODO: remove "from" from the incoming links of "via.OutgoingId"
  }
  via.OutgoingId = to;
  // TODO add incoming link from "from" to "to"
}
