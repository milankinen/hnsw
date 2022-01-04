#include "Index.h"
#include <iostream>
#include <libc.h>
#include <sstream>

using namespace std;

/*std::string stats_str(const hnsw::LayerStats &stats) {
  std::ostringstream s;
  s << "Probability: " << stats.probability << " ";
  s << "BytesPerElement: " << stats.bytes_per_element << " ";
  s << "EstimatedTotalElements: " << stats.estimated_n_elements << " ";
  s << "estimated_total_gb: " << (stats.estimated_total_bytes / 1024) / 1024. / 1024.;
  return s.str();
}*/

void print_index_stats(uint32_t dim, uint32_t M) {
  /*auto billion = 1000000000;
  cout << "Index: " << billion << " elements, dim = " << dim << ", M = " << M << endl;
  hnsw::Index index(billion, dim, M);
  cout << "max layers: " << index.layer_stats_.size() << endl;
  cout << "Layers:: " << endl;
  for (const auto &stats: index.layer_stats_) {
    cout << stats_str(stats) << endl;
  }
  cout << endl;*/
}

std::string vec_to_str(const std::vector<int> &v) {
  std::ostringstream s;
  s << "[";
  for (int i = 0; i < v.size() - 1; i++) {
    s << v[i] << " ";
  }
  if (v.size() > 0) {
    s << v[v.size() - 1] << "]";
  }
  return s.str();
}

int main(int argc, const char **argv) {
  hnsw::Index::CandidateQueue pq(5, &hnsw::Index::Candidate::cmp_furthest_first);
  pq.push(hnsw::Index::Candidate{
      .Distance = 1,
      .Id = 1
  });
  pq.push(hnsw::Index::Candidate{
      .Distance = 3,
      .Id = 3
  });
  pq.push(hnsw::Index::Candidate{
      .Distance = 2,
      .Id = 2
  });
  pq.push(hnsw::Index::Candidate{
      .Distance = 5,
      .Id = 5
  });
  pq.push(hnsw::Index::Candidate{
      .Distance = 8,
      .Id = 8
  });
  pq.push(hnsw::Index::Candidate{
      .Distance = 9,
      .Id = 9
  });

  hnsw::Index::CandidateQueue pq2(pq, hnsw::Index::Candidate::cmp_nearest_first);


  while (!pq2.empty()) {
    cout << pq2.top().Id << endl;
    pq2.pop();
  }

  return 0;
}