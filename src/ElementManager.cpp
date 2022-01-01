#include "ElementManager.h"

using hnsw::ElementManager;

#define LEVEL_MASK 0x7FFFFFFF
#define VISITED_BIT 0x80000000

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


ElementManager *ElementManager::Create(const hnsw::IndexParams &params, size_t block_size_bytes) {
  auto get_total_estimated_n_blocks = [](const Level *levels, int n_levels) {
    size_t n_blocks = 0;
    for (int i = 0; i < n_levels; i++) {
      n_blocks += levels[i].EstimatedTotalBlocks;
    }
    return n_blocks;
  };
  // Initializing levels based the calculated Index probabilities, no
  // memory allocation yet, we need to first calculate statistics for
  // all levels to get estimate for the total block/memory consumption
  auto level_p = get_level_probabilities(params);
  int n_levels = (int) level_p.size();
  auto *levels = (Level *) malloc(sizeof(Level) * n_levels);
  if (levels == nullptr) {
    return nullptr;
  }
  for (int i = 0; i < n_levels; i++) {
    auto p = level_p[i];
    auto n_edges_per_element = params.M0 + (i * params.M);
    auto bytes_per_element =
        sizeof(hnsw::ElementHeader) +
        sizeof(hnsw::Link) * n_edges_per_element +
        sizeof(float_t) * params.Dimension;
    auto estimated_total_elements =
        (uint32_t) (p * params.MaxElems);
    auto estimated_total_bytes = estimated_total_elements * bytes_per_element;
    auto estimated_total_blocks =
        estimated_total_bytes == 0 ? 0 : estimated_total_bytes / block_size_bytes +
                                         (estimated_total_bytes % block_size_bytes > 0 ? 1 : 0);
    Level level{
        .Index = i,
        .Probability = p,
        .BytesPerElement = bytes_per_element,
        .EstimatedTotalBytes = estimated_total_bytes,
        .EstimatedTotalElements = estimated_total_elements,
        .EstimatedTotalBlocks = estimated_total_blocks,
        .BlockPtr = nullptr,
        .BlockFreeBytes = 0,
        .DeletedHead = nullptr,
        .DeletedTail = nullptr
    };
    memcpy(levels + i, &level, sizeof(Level));
  }
  // Allocate offsets for fast memory lookups
  auto *offsets = (void **) malloc(params.MaxElems * sizeof(void *));

  // Allocate memory for each Index + 25% extra in case some layer(s) require more than expected
  // Using single block so that we can switch the implementation to mmap in future
  const auto n_blocks = size_t((double) get_total_estimated_n_blocks(levels, n_levels) * 1.25);
  auto n_free_blocks = n_blocks;
  auto *blocks = (uint8_t *) malloc(n_blocks * block_size_bytes);
  if (blocks == nullptr) {
    free(levels);
    return nullptr;
  }
  for (int i = 0; i < n_levels; i++) {
    auto *level = levels + i;
    n_free_blocks -= level->EstimatedTotalBlocks;
    level->BlockPtr = blocks + (n_free_blocks * block_size_bytes);
    level->BlockFreeBytes = level->EstimatedTotalBlocks * block_size_bytes;
  }

  return new ElementManager(params, levels, n_levels, offsets, blocks, block_size_bytes, n_free_blocks);
}

ElementManager::ElementManager(const IndexParams &params, Level *levels, int n_levels,
                               void **lookup,
                               uint8_t *blocks,
                               size_t block_size_bytes,
                               size_t n_free_blocks)
    : rnd_(1337), // NOLINT
      n_links_per_level_(int(params.M)),
      data_size_bytes_(int(params.Dimension * sizeof(float_t))),
      n_levels_(n_levels),
      levels_(levels),
      blocks_(blocks),
      block_size_bytes_(block_size_bytes),
      n_free_blocks_(n_free_blocks),
      deleted_list_head_(nullptr),
      next_elem_id_(1),
      n_elements_(0),
      elem_lookup_(lookup),
      visited_queue_() {
}


ElementManager::~ElementManager() = default;

hnsw::element_id_t ElementManager::AllocateNextElement(uint32_t external_id) {
  if (deleted_list_head_ != nullptr) {
    auto reused_id = deleted_list_head_->Id;
    auto level_idx = deleted_list_head_->Level;
    auto next_id = deleted_list_head_->NextId;
    auto *level = levels_ + level_idx;
    if (level->DeletedHead == level->DeletedTail) {
      level->DeletedTail = nullptr;
    }
    level->DeletedHead = nullptr;
    if (next_id != NoElement) {
      auto *next_deleted = (DeletedListNode *) GetPtr(next_id);
      next_deleted->PrevId = NoElement;
      deleted_list_head_ = next_deleted;
      levels_[next_deleted->Level].DeletedHead = next_deleted;
    } else {
      deleted_list_head_ = nullptr;
    }
    initialize_element(GetPtr(reused_id), level, external_id);
    return reused_id;
  }

  auto *level = next_random_level();
  if (level->BlockFreeBytes < level->BytesPerElement) {
    if (n_free_blocks_ == 0) {
      return NoElement;
    }
    n_free_blocks_ -= 1;
    level->BlockPtr = blocks_ + (n_free_blocks_ * block_size_bytes_);
    level->BlockFreeBytes = block_size_bytes_;
  }
  auto new_id = next_elem_id_++;
  auto *ptr = level->BlockPtr;
  level->BlockPtr += level->BytesPerElement;
  level->BlockFreeBytes -= level->BytesPerElement;
  elem_lookup_[new_id - 1] = ptr;
  initialize_element(ptr, level, external_id);
  return new_id;
}

void ElementManager::FreeElement(element_id_t id) {
  auto *ptr = GetPtr(id);
  auto level_idx = GetLevel(ptr);
  auto *level = levels_ + level_idx;
  // First find next and previous element from the global delete list
  DeletedListNode *next = nullptr;
  DeletedListNode *prev = nullptr;
  if (level->DeletedHead != nullptr) {
    // Easy case: level already have deletions, so next element is the current
    // head of level deletion list and previous can be found by following the list
    next = level->DeletedHead;
    prev = next->PrevId != NoElement ? (DeletedListNode *) GetPtr(next->PrevId) : nullptr;
  } else {
    // Level does not deletions yet, need to find previous and next by traversing
    // the levels (higher levels are always placed before so trying to find next
    // from lower levels and prev from higher ones)
    for (int i = level_idx - 1; next == nullptr && i >= 0; i--) {
      next = levels_[i].DeletedHead;
    }
    for (int i = level_idx + 1; next == nullptr && i < n_levels_; i++) {
      prev = levels_[i].DeletedTail;
    }
  }

  // Create delete list node by reusing actual element data - it's not used
  // so just rewrite the element ptr (+ some vector data) with delete
  // list node metadata
  auto *node = (DeletedListNode *) ptr;
  node->Id = id;
  node->Level = level_idx;
  node->NextId = next != nullptr ? next->Id : NoElement;
  node->PrevId = prev != nullptr ? prev->Id : NoElement;
  // Fix linking
  if (next != nullptr) {
    next->PrevId = id;
  }
  if (prev != nullptr) {
    prev->NextId = id;
  }
  if (level->DeletedHead == level->DeletedTail) {
    level->DeletedTail = node;
  }
  if (level->DeletedHead == deleted_list_head_) {
    deleted_list_head_ = node;
  }
  // Finally, mark the new delete list node as the head of the
  // original element level
  level->DeletedHead = node;
}

void *hnsw::ElementManager::GetPtr(element_id_t id) const {
  return elem_lookup_[id - 1];
}

float_t *ElementManager::GetData(void *ptr) {
  return (float_t *) ((uint8_t *) ptr + sizeof(ElementHeader));
}

hnsw::Link *ElementManager::GetLinks(void *ptr, int level) const {
  return (Link * )((uint8_t *) ptr + sizeof(ElementHeader) + data_size_bytes_ +
                   (level == 0 ? 0 : (((level + 1) * n_links_per_level_ * sizeof(Link)))));
}

int ElementManager::GetLevel(void *ptr) {
  return ((ElementHeader *) ptr)->Flags & LEVEL_MASK; // NOLINT
}

bool ElementManager::IsVisited(void *ptr) {
  return (((ElementHeader *) ptr)->Flags & VISITED_BIT) != 0;
}

void ElementManager::MarkVisited(void *ptr) {
  ((ElementHeader *) ptr)->Flags |= VISITED_BIT;
  visited_queue_.push(ptr);
}

uint32_t ElementManager::GetExternalId(void *ptr) {
  return ((ElementHeader *) ptr)->ExternalId;
}

ElementManager::Level *ElementManager::next_random_level() {
  double f = rnd_() / double(std::mt19937::max());
  for (int i = 0; i < n_levels_; i++) {
    auto p = levels_[i].Probability;
    if (f < p) {
      return levels_ + i;
    }
    f -= p;
  }
  return levels_ + (n_levels_ - 1);
}

void ElementManager::initialize_element(void *ptr, const ElementManager::Level *level, uint32_t external_id) const {
  auto *header = (ElementHeader *) ptr;
  header->ExternalId = external_id;
  header->Flags = level->Index;
  for (int i = 0; i < level->Index; i++) {
    auto *links = GetLinks(ptr, i);
    int count = (i == 0 ? 2 : 1) * n_links_per_level_;
    memset(links, Link::NotUsed, count * sizeof(Link));
  }
}

void ElementManager::ClearVisitedMarkers() {
  while (!visited_queue_.empty()) {
    auto *ptr = visited_queue_.front();
    ((ElementHeader *) ptr)->Flags |= ~VISITED_BIT;
    visited_queue_.pop();
  }
}

int ElementManager::GetMaxLinks(int level) const {
  return level == 0 ? 2 * n_links_per_level_ : n_links_per_level_;
}

