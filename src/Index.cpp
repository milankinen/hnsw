#include "Index.h"


using hnsw::Index;

Index::Index(const hnsw::IndexParams &params, hnsw::ElementManager *elems)
    : params_(params),
      elems_(elems),
      entrypoint_(nullptr) {

}

hnsw::element_id_t Index::Insert(float_t *point, uint32_t external_id) {
  auto *elems = elems_;
  const auto id = elems->AllocateNextElement(external_id);
  if (id == ElementManager::NoElement) {
    return Failure;
  }

  auto *ep = entrypoint_;
  if (ep == nullptr) {
    entrypoint_ = elems->GetPtr(id);
    return id;
  }

  auto *ptr = elems->GetPtr(id);
  int inserted_level = ElementManager::GetLevel(ptr);
  int cur_max_level = ElementManager::GetLevel(ep);

  for (int level = cur_max_level; level > inserted_level; level--) {
    // TODO greedy_update_nearest
  }

  for (int level = inserted_level; level >= 0; level--) {
    // TODO add_links_starting_from
  }

  if (inserted_level > cur_max_level) {
    entrypoint_ = ptr;
  }

  return id;
}
