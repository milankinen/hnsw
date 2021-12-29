#pragma once

#include <cstdint>
#include <random>
#include "IndexParams.h"

namespace hnsw {

  typedef uint32_t id_t;

  struct ElementHeader {
    id_t ExternalId;
    uint16_t NumOutgoingLinks;
    uint16_t Level;
  };

  struct Link {
    // float_t Distance;
    uint32_t OutgoingId;
    uint32_t IncomingNextId;
  };

  class ElementManager {
  public:
    static const hnsw::id_t NoElement = UINT32_MAX;

    static ElementManager *Create(const IndexParams &params, size_t block_size_bytes);

    hnsw::id_t AllocateNextElement();

    void FreeElement(hnsw::id_t id);

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

    ElementManager(const IndexParams &params, Level *levels, int n_levels, uint64_t *offsets, uint8_t *blocks,
                   size_t block_size_bytes,
                   size_t free_bytes);

    inline void *elem_ptr(hnsw::id_t id);

    Level *next_random_level();

    std::mt19937 rnd_;
    IndexParams index_params_;
    Level *levels_;
    int n_levels_;
    uint8_t *blocks_;
    size_t block_size_bytes_;
    size_t n_free_blocks_;
    DeletedElement *deleted_head_;
    uint32_t next_elem_id_;
    uint64_t *elem_offsets_;

  };
}

