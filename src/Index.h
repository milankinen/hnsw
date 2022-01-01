#pragma once

#include <cstdint>
#include <random>
#include <vector>
#include "IndexParams.h"
#include "ElementManager.h"

namespace hnsw {

  class Index {

    typedef float_t (*distance_function)(float_t *, float_t *, uint32_t n);

  public:

    static const element_id_t Failure = UINT32_MAX;

    Index(const IndexParams &params, ElementManager *elems, distance_function get_distance);

    element_id_t Insert(float_t *data, uint32_t external_id);

  private:

    void
    update_entrypoint_to_nearest(float_t *point, void *&entrypoint, float_t &distance_to_entrypoint, int level) const;

    const distance_function get_distance_;
    const uint32_t dimension_;
    IndexParams params_;
    ElementManager *elems_;
    void *entrypoint_;

  };
}

