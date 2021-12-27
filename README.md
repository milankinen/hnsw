# hnsw

My implementation of HNSW ANN search index described in paper
[Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/pdf/1603.09320.pdf)

Goals:

* Incremental updates and deletes without tombstones
* Lock-free [STM](https://en.wikipedia.org/wiki/Software_transactional_memory) updates
* Efficient memory layout suitable for `mmap`

Still very WIP!

## License

MIT 
