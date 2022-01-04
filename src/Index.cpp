#include <set>
#include "Index.h"


using hnsw::Index;

Index::Index(const hnsw::IndexParams &params, hnsw::ElementManager *elems, distance_function get_distance)
    : params_(params),
      dimension_(params.Dimension),
      elems_(elems),
      entrypoint_(ElementManager::NoElement),
      get_distance_(get_distance) {

}

hnsw::element_id_t Index::Insert(float_t *point, uint32_t external_id) {
  // TODO use arg
  int ef = 200;

  auto *elems = elems_;
  const auto id = elems->AllocateNextElement(external_id);
  if (id == ElementManager::NoElement) {
    return Failure;
  }

  if (entrypoint_ == ElementManager::NoElement) {
    entrypoint_ = id;
    return id;
  }

  auto ep = entrypoint_;
  auto *ep_ptr = elems->GetPtr(ep);
  auto *ptr = elems->GetPtr(id);
  int inserted_level = ElementManager::GetLevel(ptr);
  int cur_max_level = ElementManager::GetLevel(ep_ptr);
  float_t distance_to_ep = get_distance_(point, ElementManager::GetData(ep_ptr), dimension_);

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
    add_links(id, neighbors);
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
  const auto dim = dimension_;
  const auto get_dist = get_distance_;
  auto *elems = elems_;
  int max_links = elems->GetMaxLinks(level);
  while (true) {
    element_id_t nearest = entrypoint;
    auto *links = elems->GetLinks(elems->GetPtr(nearest), level);
    for (int i = 0; i < max_links; i++) {
      auto &link = links[i];
      if (link.OutgoingId == Link::NotUsed) {
        break;
      }
      element_id_t element = link.OutgoingId;
      float_t distance = get_dist(point, ElementManager::GetData(elems->GetPtr(element)), dim);
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
  const auto dim = dimension_;
  const auto get_dist = get_distance_;
  auto *elems = elems_;
  const int max_links = elems->GetMaxLinks(level);
  CandidateQueue candidates(Candidate::cmp_nearest_first, nearest_elements_furthest_first.get_container());

  while (!candidates.empty()) {
    const Candidate nearest_candidate = candidates.pop_first();
    if (nearest_candidate.Distance > nearest_elements_furthest_first.top().Distance) {
      return;
    }
    auto *cand_ptr = elems->GetPtr(nearest_candidate.Id);
    auto *cand_links = elems->GetLinks(cand_ptr, level);
    for (int i = 0; i < max_links; i++) {
      auto neighbor_id = cand_links[i].OutgoingId;
      if (neighbor_id == Link::NotUsed) {
        break;
      }
      if (visited.count(neighbor_id) == 0) {
        visited.insert(neighbor_id);
        auto *neighbor_ptr = elems->GetPtr(neighbor_id);
        float_t distance = get_dist(point, ElementManager::GetData(neighbor_ptr), dim);
        const Candidate &furthest_of_nearest = nearest_elements_furthest_first.top();
        if (distance < furthest_of_nearest.Distance || nearest_elements_furthest_first.size() < ef) {
          Candidate new_cand{
              .Id = neighbor_id,
              .Distance = distance,
          };
          if (nearest_elements_furthest_first.size() >= ef) {
            nearest_elements_furthest_first.pop();
          }
          nearest_elements_furthest_first.push(new_cand);
          candidates.push(new_cand);
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
    auto *elems = elems_;
    const auto dim = dimension_;
    const auto get_dist = get_distance_;
    const int max_neighbors = elems->GetMaxLinks(level);
    for (const auto &cand: candidates.get_container()) {
      auto *cand_neighbors = elems->GetLinks(elems->GetPtr(cand.Id), level);
      for (int i = 0; i < max_neighbors; i++) {
        element_id_t neighbor_id = cand_neighbors[i].OutgoingId;
        if (neighbor_id == Link::NotUsed) {
          break;
        }
        if (visited.count(cand.Id) == 0) {
          float_t distance = get_dist(point, ElementManager::GetData(elems->GetPtr(neighbor_id)), dim);
          work_queue.emplace(Candidate{
              .Id = neighbor_id,
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

void remove_incoming(hnsw::ElementManager *elems, hnsw::element_id_t from, hnsw::element_id_t incoming_to_remove,
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
}

void Index::add_links(
    float_t *point,
    hnsw::element_id_t element,
    const Index::CandidateQueue &neighbors,
    int level) {
  auto *elems = elems_;
  const auto dim = dimension_;
  const auto get_dist = get_distance_;
  int max_links = elems->GetMaxLinks(level);
  auto *ptr = elems->GetPtr(element);
  auto *links = elems->GetLinks(ptr, level);
  int link_idx = 0;
  for (const auto &neighbor: neighbors.get_container()) {
    auto &link = links[link_idx];
    elems->SetOutgoingLink(element, link, neighbor.Id);
    link_idx++;
    bool needs_shrink = true;
    auto *neighbor_links = elems->GetLinks(elems->GetPtr(neighbor.Id), level);
    for (int i = 0; i < max_links; i++) {
      if (neighbor_links[i].OutgoingId == Link::NotUsed) {
        elems->SetOutgoingLink(neighbor.Id, link, element);
        needs_shrink = false;
        break;
      }
    }
    if (needs_shrink) {
      float_t *neighbor_point = ElementManager::GetData(elems->GetPtr(neighbor.Id));
      float_t furthest_dist = get_dist(neighbor_point, point, dim);
      int furthest_idx = -1;
      for (int i = 0; i < max_links; i++) {
        element_id_t nn_id = neighbor_links[i].OutgoingId;
        float_t dist = get_dist(neighbor_point, ElementManager::GetData(elems->GetPtr(nn_id)), dim);
        if (dist > furthest_dist) {
          furthest_idx = i;
          furthest_dist = dist;
        }
      }
      if (furthest_idx != -1) {
        // some old element should be dropped from neighbor's outbound links
        auto &link_to_shrink = neighbor_links[furthest_idx];
        element_id_t elem_to_shrink = link_to_shrink.OutgoingId;
        remove_incoming(elems, elem_to_shrink, neighbor.Id, link_to_shrink.IncomingNextId, level);
      } else {
        // element that we just added was dropped from neighbor's outbound links
        element_id_t head = 1; // TODO elems->GetHead(neighbor);
        link.IncomingNextId = head;
        // TODO elems->SetHead(neighbor, element);
      }
    }
  }
}



