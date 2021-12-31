#include "Index.h"


using hnsw::Index;

Index::Index(const hnsw::IndexParams &params, hnsw::ElementManager *elems)
    : params_(params),
      elems_(elems),
      entrypoint_(nullptr) {

}

hnsw::id_t Index::Insert(float_t *point, uint32_t external_id) {
  auto *elems = elems_;
  const auto id = elems->AllocateNextElement(external_id);
  if (id == ElementManager::NoElement) {
    return Failure;
  }
  auto *elem_ptr = elems->GetPtr(id);
  auto *header = elems->GetHeader(elem_ptr);

  if (entrypoint_ == nullptr) {
    entrypoint_ = elem_ptr;
    return id;
  }

  return id;
}
