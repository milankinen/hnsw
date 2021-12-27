#include <gtest/gtest.h>
#include "hnsw.h"

TEST(ExampleTest, Greet) {
  auto msg = greet();
  EXPECT_STREQ(msg.c_str(), "tsers!");
}

