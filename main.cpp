#include "hnsw.h"
#include <iostream>
#include <libc.h>
#include <sstream>

using namespace std;

std::string stats_str(const hnsw::LayerStats &stats) {
  std::ostringstream s;
  s << "probability: " << stats.probability << " ";
  s << "bytes_per_element: " << stats.bytes_per_element << " ";
  s << "estimated_n_elements: " << stats.estimated_n_elements << " ";
  s << "estimated_total_gb: " << (stats.estimated_total_bytes / 1024) / 1024. / 1024.;
  return s.str();
}

void print_index_stats(uint32_t dim, uint32_t M) {
  auto billion = 1000000000;
  cout << "Index: " << billion << " elements, dim = " << dim << ", M = " << M << endl;
  hnsw::Index index(billion, 512, 48);
  cout << "max layers: " << index.layer_stats_.size() << endl;
  cout << "Layers:: " << endl;
  for (const auto &stats: index.layer_stats_) {
    cout << stats_str(stats) << endl;
  }
  cout << endl;
}

int main(int argc, const char **argv) {
  cout << "page size: " << (size_t) sysconf(_SC_PAGESIZE) << " | " << getpagesize() << endl;
  cout << "--------" << endl << endl;
  print_index_stats(512, 8);
  print_index_stats(512, 16);
  print_index_stats(512, 48);
  print_index_stats(128, 8);
  print_index_stats(128, 16);
  print_index_stats(128, 48);
  return 0;
}