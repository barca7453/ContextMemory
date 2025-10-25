#include <catch2/catch_test_macros.hpp>
#include "ContextMemory/vector_store.h"

TEST_CASE("VectorStore can be created", "[vector_store]") {
    // Test that we can instantiate a VectorStore
    VectorStore<hnswlib::L2Space> store("test_index", 128);
    
    REQUIRE(true);
}

TEST_CASE("VectorStore can add hardcoded vectors", "[vector_store]") {
    // TODO: Write test first, then implement
}
