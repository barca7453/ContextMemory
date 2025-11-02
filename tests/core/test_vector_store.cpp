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
    VectorStore<L2Sim> store("test_index", 10);
    std::vector<float> vec = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    store.add_vector(1, vec);
    REQUIRE(store.get_index_current_count() == 1);
    REQUIRE(store.get_id_to_label().find(1) != store.get_id_to_label().end());
    REQUIRE(store.get_label_to_id()[0] == 1);
    REQUIRE(store.get_next_label() == 1);
    REQUIRE(store.get_current_resized_label_vec_size() == 1000);
    REQUIRE(store.get_id_to_label().size() == 1);
}

TEST_CASE("VectorStore can save the mappings", "[vector_store]") {
    VectorStore<L2Sim> store("test_index", 10);
    std::vector<std::vector<float>> vec = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1},
        {1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2},
        {1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3},
        {1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4},
        {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5},
        {1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 10.6},
        {1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 9.7, 10.7},
        {1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75, 9.75, 10.75},
        {1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 10.8}};
    auto user_id = 0;
    for (const auto& v : vec) {
        store.add_vector(user_id++,v);
    }

    auto nextLabel = store.get_next_label();
    std::cout << "Next Label is " << nextLabel << std::endl;
    store.save_mappings("index_test");
    // clear the mappings in the store
    store.cleanMappings();
    std::cout << "Next Label after clearing is " << store.get_next_label() << std::endl;
    store.load_mappings("index_test");
    std::cout << "Next Label after reloading is " << store.get_next_label() << std::endl;

    // next label after loading should be the same as when the
    // store was saved
    REQUIRE (store.get_next_label() == nextLabel);

}
 
