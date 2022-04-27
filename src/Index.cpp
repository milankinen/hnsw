#include <set>
#include "Index.h"


using hnsw::Index;

Index::Index(const hnsw::IndexParams &params, hnsw::Elements *elems, distance_function get_distance)
    : params_(params),
      dimension_(params.Dimension),
      elems_(elems),
      entrypoint_(NoElement),
      get_distance_(get_distance) {

}

hnsw::element_id_t Index::Insert(float_t *point, uint32_t external_id) {
  // TODO use arg
  int ef = 200;

  auto *elems = elems_;
  const auto id = elems->AllocateNextElement(external_id, point);
  if (id == NoElement) {
    return Failure;
  }

  if (entrypoint_ == NoElement) {
    entrypoint_ = id;
    return id;
  }

  auto ep = entrypoint_;
  int inserted_level = elems->GetElementLevel(id);
  int cur_max_level = elems->GetElementLevel(ep);
  float_t distance_to_ep = get_distance_(point, elems->GetLayerAt(cur_max_level)->GetData(ep), dimension_);

  // Algorithm 1, lines 5-7
  for (int level = cur_max_level; level > inserted_level; level--) {
    update_entrypoint_to_nearest(point, ep, distance_to_ep, level);
  }
  CandidateQueue nearest_elements_furthest_first(Candidate::cmp_furthest_first);
  nearest_elements_furthest_first.push(Candidate{
      .Id = entrypoint_,
      .Distance = distance_to_ep
  });

  // Algorithm 1, lines 8-17
  VisitedElementsSet visited;
  bool extend_candidates = true;
  bool keep_pruned_connections = true;
  for (int level = inserted_level; level >= 0; level--) {
    search_layer_update_nearest(point, nearest_elements_furthest_first, visited, ef, level);
    CandidateQueue neighbors(Candidate::cmp_nearest_first);
    select_neighbors(point, neighbors, nearest_elements_furthest_first, visited, level, extend_candidates,
                     keep_pruned_connections);
    add_links(id, neighbors, level);
  }

  // Algorithm 1, lines 18-19
  if (inserted_level > cur_max_level) {
    entrypoint_ = id;
  }

  return id;
}

void Index::update_entrypoint_to_nearest(
    float_t *point,
    element_id_t &entrypoint,
    float_t &distance_to_entrypoint,
    int level) const {
  // Adapted implementation of faiss greedy_update_nearest,
  // from https://github.com/facebookresearch/faiss (MIT)
  // This is basically Algorihm 2 from the paper but adapted to ef=1 constraint
  // which removes the need for priority queue and visited set, thus making
  // the algorithm execution faster
  auto *layer = elems_->GetLayerAt(level);
  int max_neighbors = layer->MaxNeighbors;
  while (true) {
    element_id_t nearest = entrypoint;
    auto *neighbors = layer->GetNeighbors(nearest);
    for (int i = 0; i < max_neighbors; i++) {
      auto &neighbor = neighbors[i];
      if (Link::IsUnused(neighbor)) {
        break;
      }
      element_id_t element = neighbor.OutgoingId;
      float_t distance = get_distance_(point, layer->GetData(element), dimension_);
      if (distance < distance_to_entrypoint) {
        entrypoint = element;
        distance_to_entrypoint = distance;
      }
    }
    if (nearest == entrypoint) {
      return;
    }
  }
}


void Index::search_layer_update_nearest(
    float_t *point,
    CandidateQueue &nearest_elements_furthest_first,
    VisitedElementsSet &visited,
    int ef,
    int level) {
  auto *layer = elems_->GetLayerAt(level);
  const int max_neighbors = layer->MaxNeighbors;
  CandidateQueue candidates(Candidate::cmp_nearest_first, nearest_elements_furthest_first.get_container());

  while (!candidates.empty()) {
    const Candidate nearest_candidate = candidates.pop_first();
    if (nearest_candidate.Distance > nearest_elements_furthest_first.top().Distance) {
      return;
    }
    element_id_t candidate = nearest_candidate.Id;
    auto *cand_neighbors = layer->GetNeighbors(candidate);
    for (int i = 0; i < max_neighbors; i++) {
      auto neighbor = cand_neighbors[i].OutgoingId;
      if (neighbor == NoElement) {
        break;
      }
      if (visited.count(neighbor) == 0) {
        visited.insert(neighbor);
        float_t distance = get_distance_(point, layer->GetData(neighbor), dimension_);
        const Candidate &furthest_of_nearest = nearest_elements_furthest_first.top();
        if (distance < furthest_of_nearest.Distance || nearest_elements_furthest_first.size() < ef) {
          Candidate new_cand{
              .Id = neighbor,
              .Distance = distance,
          };
          if (nearest_elements_furthest_first.size() >= ef) {
            nearest_elements_furthest_first.pop();
          }
          nearest_elements_furthest_first.emplace(new_cand);
          candidates.emplace(new_cand);
        }
      }
    }
  }
}

void Index::select_neighbors(
    float_t *point,
    Index::CandidateQueue &neighbors,
    const Index::CandidateQueue &candidates,
    const VisitedElementsSet &visited,
    int level,
    bool extend_candidates,
    bool keep_pruned_connections) {
  int M = int(level == 0 ? params_.M0 : params_.M);
  CandidateQueue work_queue(Candidate::cmp_nearest_first, candidates.get_container());
  CandidateQueue discarded(Candidate::cmp_nearest_first);

  if (extend_candidates) {
    auto *layer = elems_->GetLayerAt(level);
    int max_neighbors = layer->MaxNeighbors;
    for (const auto &cand: candidates.get_container()) {
      auto *cand_neighbors = layer->GetNeighbors(cand.Id);
      for (int i = 0; i < max_neighbors; i++) {
        element_id_t neighbor = cand_neighbors[i].OutgoingId;
        if (neighbor == NoElement) {
          break;
        }
        if (visited.count(neighbor) == 0) {
          float_t distance = get_distance_(point, layer->GetData(neighbor), dimension_);
          work_queue.emplace(Candidate{
              .Id = neighbor,
              .Distance = distance
          });
        }
      }
    }
  }

  while (!work_queue.empty() && neighbors.size() < M) {
    Candidate e = work_queue.pop_first();
    if (neighbors.empty() || e.Distance < neighbors.top().Distance) {
      neighbors.emplace(e);
    } else if (keep_pruned_connections) {
      discarded.emplace(e);
    }
  }
  while (!discarded.empty() && neighbors.size() < M) {
    neighbors.emplace(discarded.pop_first());
  }
}

/*
void remove_incoming(hnsw::Elements *elems, hnsw::element_id_t from, hnsw::element_id_t incoming_to_remove,
                     hnsw::element_id_t next_incoming, int level) {
  int max_links = elems->GetMaxLinks(level);
  hnsw::element_id_t head = 1; // TODO elems->GetHead(from)
  if (head == incoming_to_remove) {
    // TODO elems->SetHead(head, next_incoming);
    return;
  }
  while (true) {
    hnsw::element_id_t next_head = head;
    auto *links = elems->GetLinks(elems->GetPtr(head), level);
    for (int i = 0; i < max_links; i++) {
      if (links[i].OutgoingId == from) {
        next_head = links[i].IncomingNextId;
        if (next_head == incoming_to_remove) {
          links[i].IncomingNextId = next_incoming;
          return;
        }
      }
    }
    assert(next_head != head);
  }
}*/

void add_link_from_element_to_neighbor(
    hnsw::Elements::Layer *layer,
    hnsw::element_id_t element,
    hnsw::element_id_t neighbor,
    hnsw::Link &element_link) {
  layer->ConnectElements(element, neighbor, element_link);
}

void add_link_from_neighbor_to_element(
    hnsw::Elements::Layer *layer,
    hnsw::element_id_t element,
    hnsw::element_id_t neighbor,
    float_t distance_from_neighbor_to_element,
    hnsw::distance_function get_distance,
    uint32_t dimension
) {
  int max_neighbors = layer->MaxNeighbors;
  auto *neighbors_of_neighbor = layer->GetNeighbors(neighbor);
  for (int i = 0; i < max_neighbors; i++) {
    if (hnsw::Link::IsUnused(neighbors_of_neighbor[i])) {
      // Found free link for neighborhood connection => no need for shrinking anymore,
      // just add a directed link from neighbor to element via the free link
      layer->ConnectElements(neighbor, element, neighbors_of_neighbor[i]);
      return;
    }
  }

  // We didn't find any free link, so we need to perfom shrinking. Using
  // simple heuristic here => dicarding the furthest neighbor
  float_t *neighbor_data = layer->GetData(neighbor);
  float_t furthest_dist = distance_from_neighbor_to_element;
  int discarded_idx = -1;
  for (int i = 0; i < max_neighbors; i++) {
    hnsw::element_id_t candidate = neighbors_of_neighbor[i].OutgoingId;
    float_t dist = get_distance(neighbor_data, layer->GetData(candidate), dimension);
    if (dist > furthest_dist) {
      discarded_idx = i;
      furthest_dist = dist;
    }
  }
  // If we found existing element that is further from the neighbor than the
  // new element, then we must establish a new connection from neighbor
  // to the new element, discarding the previous one. Otherwise, we don't
  // need to do anything since we're already added connection from element
  // to this neighbor
  if (discarded_idx != -1) {
    layer->ConnectElements(neighbor, element, neighbors_of_neighbor[discarded_idx]);
  }
}

void Index::add_links(
    hnsw::element_id_t element,
    const Index::CandidateQueue &neighbors,
    int level) {
  auto *layer = elems_->GetLayerAt(level);
  auto *links = layer->GetNeighbors(element);
  int neighbor_idx = 0;
  for (const auto &neighbor: neighbors.get_container()) {
    add_link_from_element_to_neighbor(layer, element, neighbor.Id, links[neighbor_idx]);
    add_link_from_neighbor_to_element(layer, element, neighbor.Id, neighbor.Distance, get_distance_, dimension_);
    neighbor_idx++;
  }
}

