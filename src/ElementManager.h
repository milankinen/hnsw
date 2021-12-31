#pragma once

#include <cstdint>
#include <random>
#include "IndexParams.h"

namespace hnsw {

  typedef uint32_t id_t;

  struct ElementHeader {
    id_t ExternalId;
    uint32_t Flags;

    inline int GetLevel() const {
      return Flags;
    }
  };

  struct Link {
    // float_t Distance;
    uint32_t OutgoingId;
    uint32_t IncomingNextId;
  };

  struct LevelLinks {
    Link *data;
    int count;
  };

  class ElementManager {
  public:
    static const hnsw::id_t NoElement = 0;

    static ElementManager *Create(const IndexParams &params, size_t block_size_bytes);

    hnsw::id_t AllocateNextElement(uint32_t external_id);

    void FreeElement(hnsw::id_t id);

    inline void *GetPtr(id_t id) const;

    inline ElementHeader *GetHeader(void *ptr) const;

    inline float_t *GetData(void *ptr) const;

    inline LevelLinks GetLinks(void *ptr, int level) const;

    inline bool IsEmpty() const;

  private:

    struct DeletedElement {
      uint32_t Id;
      int32_t Level;
      uint32_t NextId;
      uint32_t PrevId;
    };

    struct Level {
      const int32_t Index;
      const double_t Probability;
      const size_t BytesPerElement;
      const size_t EstimatedTotalBytes;
      const size_t EstimatedTotalElements;
      const size_t EstimatedTotalBlocks;

      uint8_t *BlockPtr;
      size_t BlockFreeBytes;
      DeletedElement *DeletedHead;
      DeletedElement *DeletedTail;
    };

    ElementManager(const IndexParams &params, Level *levels, int n_levels, void **lookup, uint8_t *blocks,
                   size_t block_size_bytes,
                   size_t free_bytes);

    Level *next_random_level();

    void initialize_element(void *ptr, const Level *level, uint32_t external_id) const;

    std::mt19937 rnd_;
    IndexParams index_params_;
    int n_links_per_level_;
    Level *levels_;
    int n_levels_;
    uint8_t *blocks_;
    size_t block_size_bytes_;
    size_t n_free_blocks_;
    DeletedElement *deleted_head_;
    uint32_t next_elem_id_;
    uint32_t n_elements_;
    void **elem_lookup_;

  };
}

