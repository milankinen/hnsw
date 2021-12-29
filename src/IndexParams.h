#pragma once

#include <cstdint>

namespace hnsw {
  struct IndexParams {
    uint32_t Dimension;
    uint32_t MaxElems;
    uint32_t M;
    uint32_t M0;
  };
}