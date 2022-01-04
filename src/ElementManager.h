#pragma once

#include <cstdint>
#include <random>
#include <queue>
#include "IndexParams.h"

namespace hnsw {

  typedef uint32_t element_id_t;

  struct ElementHeader {
    uint32_t ExternalId;
    uint32_t Flags;
  };

  struct Link {
    static const element_id_t NotUsed = 0;
    // float_t Distance;
    element_id_t OutgoingId;
    element_id_t IncomingNextId;
  };

  class ElementManager {
  public:
    ~ElementManager();

    static const element_id_t NoElement = 0;

    static ElementManager *Create(const IndexParams &params, size_t block_size_bytes);

    hnsw::element_id_t AllocateNextElement(uint32_t external_id);

    void FreeElement(element_id_t id);

    inline void *GetPtr(element_id_t id) const;

    static float_t *GetData(void *ptr);

    static inline int GetLevel(void *ptr);

    static inline uint32_t GetExternalId(void *ptr);

    inline int GetMaxLinks(int level) const;

    inline Link *GetLinks(void *ptr, int level) const;

    inline void SetOutgoingLink(element_id_t id, Link &link, element_id_t outgoing_id);

  private:

    struct DeletedListNode {
      uint32_t Id;
      int Level;
      uint32_t NextId;
      uint32_t PrevId;
    };

    struct Level {
      const int Index;
      const double Probability;
      const size_t BytesPerElement;
      const size_t EstimatedTotalBytes;
      const size_t EstimatedTotalElements;
      const size_t EstimatedTotalBlocks;

      uint8_t *BlockPtr;
      size_t BlockFreeBytes;
      DeletedListNode *DeletedHead;
      DeletedListNode *DeletedTail;
    };

    ElementManager(const IndexParams &params, Level *levels, int n_levels, void **lookup, uint8_t *blocks,
                   size_t block_size_bytes,
                   size_t free_bytes);

    Level *next_random_level();

    void initialize_element(void *ptr, const Level *level, uint32_t external_id) const;

    const int n_links_per_level_;
    const int data_size_bytes_;
    const int n_levels_;
    Level *levels_;
    uint8_t *blocks_;
    size_t block_size_bytes_;
    size_t n_free_blocks_;
    DeletedListNode *deleted_list_head_;
    uint32_t next_elem_id_;
    uint32_t n_elements_;
    void **elem_lookup_;
    std::mt19937 rnd_;
  };
}

