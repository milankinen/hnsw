#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace hnsw {

  struct IndexParams {
    uint32_t dimension;
    uint32_t max_elems;
    uint32_t M;
    uint32_t M0;
    double mL;
  };

  struct LayerStats {
    const float probability;
    const uint32_t bytes_per_element;
    const uint32_t estimated_n_elements;
    const uint64_t estimated_total_bytes;
  };

  class Index {

  public:
    Index(uint32_t max_elems, uint32_t dim, uint32_t M);

    //private:

    int next_random_level();

    IndexParams params_;
    std::random_device rnd_;
    std::vector<LayerStats> layer_stats_;

    void *data_;

  };
}

