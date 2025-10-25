#include <catch2/catch_test_macros.hpp>
#include "ContextMemory/vector_store.h"

TEST_CASE("VectorStore can be created", "[vector_store]") {
    // Test that we can instantiate a VectorStore
    //VectorStore<hnswlib::L2Space> store("test_index", 128);
    VectorStore<L2Sim> store("test_index", 128);
    REQUIRE(store.get_dim() == 128);
    REQUIRE(store.get_max_elements() == 10000);
    REQUIRE(store.get_M() == 16);
    REQUIRE(store.get_ef_construction() == 200);
    REQUIRE(store.get_ef() == 10);
    REQUIRE(store.get_allow_replace_deleted() == true);
    
    REQUIRE(true);
}

TEST_CASE("VectorStore can add hardcoded vectors", "[vector_store]") {
    // TODO: Write test first, then implement
    VectorStore<L2Sim> store("test_index", 128);
    std::vector<float> vec = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    store.add_vector(1, vec);
    REQUIRE(store.get_index_current_count() == 1);
    REQUIRE(store.get_id_to_label().find(1) != store.get_id_to_label().end());
    REQUIRE(store.get_label_to_id()[0] == 1);
    REQUIRE(store.get_next_label() == 1);
    REQUIRE(store.get_current_resized_label_vec_size() == 1000);
    REQUIRE(store.get_id_to_label().size() == 1);

}
