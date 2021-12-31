#pragma once

#include <cstdint>
#include <random>
#include <vector>
#include "IndexParams.h"
#include "ElementManager.h"

namespace hnsw {

  class Index {

  public:

    static const element_id_t Failure = UINT32_MAX;

    Index(const IndexParams &params, ElementManager *elems);

    element_id_t Insert(float_t *data, uint32_t external_id);

  private:

    IndexParams params_;
    ElementManager *elems_;
    void *entrypoint_;

  };
}

