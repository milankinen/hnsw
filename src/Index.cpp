#include "Index.h"


using hnsw::Index;

Index::Index(const hnsw::IndexParams &params, hnsw::ElementManager *elems, distance_function get_distance)
    : params_(params),
      dimension_(params.Dimension),
      elems_(elems),
      entrypoint_(nullptr),
      get_distance_(get_distance) {

}

hnsw::element_id_t Index::Insert(float_t *point, uint32_t external_id) {
  auto *elems = elems_;
  const auto id = elems->AllocateNextElement(external_id);
  if (id == ElementManager::NoElement) {
    return Failure;
  }

  if (entrypoint_ == nullptr) {
    entrypoint_ = elems->GetPtr(id);
    return id;
  }

  auto *ep = entrypoint_;
  auto *ptr = elems->GetPtr(id);
  int inserted_level = ElementManager::GetLevel(ptr);
  int cur_max_level = ElementManager::GetLevel(ep);
  float_t distance_to_ep = get_distance_(point, ElementManager::GetData(ep), dimension_);

  // Algorithm 1, lines 5-7
  for (int level = cur_max_level; level > inserted_level; level--) {
    update_entrypoint_to_nearest(point, ep, distance_to_ep, level);
  }

  // Algorithm 1, lines 8-17
  for (int level = inserted_level; level >= 0; level--) {
    // TODO add_links_starting_from
  }

  // Algorithm 1, lines 18-19
  if (inserted_level > cur_max_level) {
    entrypoint_ = ptr;
  }

  return id;
}

void Index::update_entrypoint_to_nearest(float_t *point, void *&entrypoint, float_t &distance_to_entrypoint,
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
    void *nearest = entrypoint;
    auto *links = elems->GetLinks(nearest, level);
    for (int i = 0; i < max_links; i++) {
      auto &link = links[i];
      if (link.OutgoingId == Link::NotUsed) {
        break;
      }
      auto *element = elems->GetPtr(link.OutgoingId);
      float_t distance = get_dist(point, ElementManager::GetData(element), dim);
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
