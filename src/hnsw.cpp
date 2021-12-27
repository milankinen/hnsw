#include "hnsw.h"
#include <cmath>

#define BYTES_PER_LINK (4+4+4)

static auto calc_layer_stats(const hnsw::IndexParams &params) {
  std::vector<hnsw::LayerStats> stats;
  auto mL = 1. / log(params.M);
  for (int level = 0;; level++) {
    float p = exp(-level / mL) * (1. - exp(-1. / mL));
    if (p < 1e-12) {
      break;
    }
    auto header_bytes_per_element = 4;
    auto data_bytes_per_element = params.dimension * 4;
    auto link_bytes_per_element =
        BYTES_PER_LINK * params.M0 + level * BYTES_PER_LINK * params.M;
    auto bytes_per_element = header_bytes_per_element + data_bytes_per_element + link_bytes_per_element;
    auto estimated_n_elements =
        (uint32_t) (((double) p) * params.max_elems);
    stats.push_back(hnsw::LayerStats{
        .probability = p,
        .bytes_per_element = bytes_per_element,
        .estimated_n_elements = estimated_n_elements,
        .estimated_total_bytes = (uint64_t) estimated_n_elements * bytes_per_element
    });
  }
  return stats;
}

hnsw::Index::Index(uint32_t max_elements, uint32_t dimension, uint32_t M) :
    rnd_(std::random_device()),
    params_(IndexParams{
        .dimension = dimension,
        .max_elems = max_elements,
        .M = M,
        .M0 = M * 2,
        .mL = 1. / log(M)
    }),
    layer_stats_(calc_layer_stats(params_)) {
  data_ = nullptr;
}

int hnsw::Index::next_random_level() {
  std::uniform_real_distribution<double> dist(0., 1.);
  auto level = -log(dist(rnd_)) * params_.mL;
  return (int) level;
}