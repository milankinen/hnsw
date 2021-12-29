#include "hnsw.h"

hnsw::Index::Index(const hnsw::IndexParams &params, hnsw::ElementManager *elems)
    : params_(params),
      elems_(elems) {

}
