#pragma once

#include <cstdint>
#include <random>
#include <vector>
#include <unordered_set>
#include "IndexParams.h"
#include "ElementManager.h"

namespace hnsw {

  class Index {

    typedef float_t (*distance_function)(float_t *, float_t *, uint32_t n);

  public:

    static const element_id_t Failure = UINT32_MAX;

    Index(const IndexParams &params, ElementManager *elems, distance_function get_distance);

    element_id_t Insert(float_t *data, uint32_t external_id);


    struct Candidate {
      element_id_t Id;
      float_t Distance;

      static bool cmp_nearest_first(const Candidate &left, const Candidate &right) {
        return left.Distance > right.Distance;
      }

      static bool cmp_furthest_first(const Candidate &left, const Candidate &right) {
        return left.Distance < right.Distance;
      }
    };

    typedef bool (*candidate_cmp)(const Candidate &, const Candidate &);


    typedef std::unordered_set<element_id_t> VisitedElementsSet;

    class CandidateQueue : public std::priority_queue<Candidate, std::vector<Candidate>, candidate_cmp> {
    public:
      explicit CandidateQueue(candidate_cmp cmp, const std::vector<Candidate> &data)
          : std::priority_queue<Candidate, std::vector<Candidate>, candidate_cmp>(cmp, data) {
      }

      explicit CandidateQueue(candidate_cmp cmp)
          : std::priority_queue<Candidate, std::vector<Candidate>, candidate_cmp>(cmp) {

      }

      inline Candidate pop_first() {
        Candidate cand = top();
        pop();
        return cand;
      }

      const std::vector<Candidate> &get_container() const {
        return c;
      }
    };

  private:

    void update_entrypoint_to_nearest(
        float_t *point,
        element_id_t &entrypoint,
        float_t &distance_to_entrypoint,
        int level) const;

    void search_layer_update_nearest(
        float_t *point,
        CandidateQueue &nearest_elements_furthest_first,
        VisitedElementsSet &visited,
        int ef,
        int level);

    void select_neighbors(
        float_t *point,
        CandidateQueue &neighbors,
        const CandidateQueue &candidates,
        const VisitedElementsSet &visited,
        int level,
        bool extend_candidates,
        bool keep_pruned_connections
    );

    void add_links(
        float_t *point,
        element_id_t element,
        const CandidateQueue &neighbors,
        int level
    );

    const distance_function get_distance_;
    const uint32_t dimension_;
    IndexParams params_;
    ElementManager *elems_;
    element_id_t entrypoint_;

  };
}

