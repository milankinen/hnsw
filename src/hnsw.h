#pragma once

#include <cstdint>
#include <random>
#include <vector>
#include "IndexParams.h"
#include "ElementManager.h"

namespace hnsw {

  class Index {

  public:
    Index(const IndexParams &params, ElementManager *elems);

  private:

    IndexParams params_;
    ElementManager *elems_;

  };
}

