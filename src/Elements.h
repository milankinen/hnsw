#pragma once

#include <cstdint>
#include <random>
#include <queue>
#include "IndexParams.h"

namespace hnsw {

  typedef uint32_t element_id_t;

  static const element_id_t NoElement = 0;

  struct Link {
    element_id_t OutgoingId;
    element_id_t IncomingNextId;

    static inline bool IsUnused(const Link &link) {
      return link.OutgoingId == NoElement;
    }
  };

  class Elements {
  public:
    class Layer {
    private:
      const uint32_t links_offset_;
      void **lookup_;

    public:
      explicit Layer(void **lookup, int level, uint32_t links_offset, int max_neighbors)
          : lookup_(lookup),
            links_offset_(links_offset),
            Level(level),
            MaxNeighbors(max_neighbors) {}

      const int Level;
      const int MaxNeighbors;

      inline Link *GetNeighbors(element_id_t id);

      inline float_t *GetData(element_id_t id);

      void ConnectElements(element_id_t from, element_id_t to, Link &via);
    };

    ~Elements();

    static Elements *Create(const IndexParams &params, size_t block_size_bytes);

    hnsw::element_id_t AllocateNextElement(uint32_t external_id, const float_t *data);

    void FreeElement(element_id_t id);

    inline int GetElementLevel(element_id_t id);

    inline Layer *GetLayerAt(int level);

  private:

    struct ElementHeader {
      uint32_t ExternalId;
      int Level;
    };

    struct DeleteListNode {
      uint32_t DeletedElementId;
      int DeletedElementLevel;
      uint32_t NextDeletedElementId;
      uint32_t PrevDeletedElementId;
    };

    struct LayerState {
      const int Level;
      const double Probability;
      const size_t BytesPerElement;
      const size_t EstimatedTotalBytes;
      const size_t EstimatedTotalElements;
      const size_t EstimatedTotalBlocks;

      uint8_t *BlockPtr;
      size_t BlockFreeBytes;
      DeleteListNode *DeleteListHead;
      DeleteListNode *DeleteListTail;
      Layer *API;
    };

    Elements(
        const IndexParams &params,
        LayerState *layer_states,
        int n_layers,
        void **lookup,
        uint8_t *blocks,
        size_t block_size_bytes,
        size_t free_bytes);

    LayerState &select_next_random_layer();

    static void initialize_element(
        void *ptr,
        const LayerState &layer,
        uint32_t external_id,
        const float_t *data,
        uint32_t dimension);

    const IndexParams &params_;
    const int n_levels_;
    LayerState *layers_states_;
    uint8_t *blocks_;
    size_t block_size_bytes_;
    size_t n_free_blocks_;
    DeleteListNode *delete_list_head_;
    uint32_t next_elem_id_;
    void **lookup_;
    std::mt19937 rnd_;
  };
}

