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
    store.save_mappings("index_test");
    // clear the mappings in the store
    store.cleanMappings();
    store.load_mappings("index_test");

    // next label after loading should be the same as when the
    // store was saved
    REQUIRE (store.get_next_label() == nextLabel);

}

TEST_CASE("VectorStore can search vectors", "[vector_store]") {
    VectorStore<L2Sim> store("test_index", 10);
    
    // Add the same test vectors
    std::vector<std::vector<float>> vectors = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1},
        {1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2},
        {1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3},
        {1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4},
        {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5},
        {1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 10.6},
        {1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 9.7, 10.7},
        {1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75, 9.75, 10.75},
        {1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 10.8}
    };
    
    uint64_t user_id = 0;
    for (const auto& v : vectors) {
        store.add_vector(user_id++, v);
    }
    
    REQUIRE(store.get_index_current_count() == 10);
    
    // Test 1: Search for exact match (first vector)
    std::vector<float> query1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    auto results1 = store.search_vectors(query1, 3);
    
    std::cout << "Search results for exact match (top 3):" << std::endl;
    for (const auto& id : results1) {
        std::cout << "  User ID: " << id.user_id << std::endl;
    }
    
    // The exact match (user_id=0) should be the first result
    REQUIRE(results1.size() == 3);
    REQUIRE(results1[0].user_id == 0);
    
    // Test 2: Search for a vector very close to the last one
    std::vector<float> query2 = {1.81, 2.81, 3.81, 4.81, 5.81, 6.81, 7.81, 8.81, 9.81, 10.81};
    auto results2 = store.search_vectors(query2, 5);
    
    std::cout << "Search results for vector close to last (top 5):" << std::endl;
    for (const auto& id : results2) {
        std::cout << "  User ID: " << id.user_id << std::endl;
    }
    
    // The last vector (user_id=9) should be the closest
    REQUIRE(results2.size() == 5);
    REQUIRE(results2[0].user_id == 9);
    
    // Test 3: Search for a vector in the middle range
    std::vector<float> query3 = {1.45, 2.45, 3.45, 4.45, 5.45, 6.45, 7.45, 8.45, 9.45, 10.45};
    auto results3 = store.search_vectors(query3, 1);
    
    std::cout << "Search results for middle vector (top 1):" << std::endl;
    for (const auto& id : results3) {
        std::cout << "  User ID: " << id.user_id << std::endl;
    }
    
    // Should return exactly 1 result, likely user_id=4 or 5 (closest to middle)
    REQUIRE(results3.size() == 1);
    REQUIRE((results3[0].user_id == 4 || results3[0].user_id == 5));
    
    // Test 4: Request more results than available
    auto results4 = store.search_vectors(query1, 20);
    
    // Should return all 10 vectors even though we asked for 20
    REQUIRE(results4.size() == 10);
}


TEST_CASE("VectorStore can save and load index from file", "[vector_store][load]") {
    const std::string index_path = "test_save_load_index";
    const int dimension = 10;
    const size_t num_vectors = 10;
    
    // Step 1: Create a new index, add vectors, and save it
    {
        VectorStore<L2Sim> store(index_path, dimension);
        
        // Add test vectors
        std::vector<std::vector<float>> vectors = {
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
            {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1},
            {1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2},
            {1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3},
            {1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4},
            {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5},
            {1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 10.6},
            {1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 9.7, 10.7},
            {1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75, 9.75, 10.75},
            {1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 10.8}
        };
        
        uint64_t user_id = 100; // Start with non-zero IDs
        for (const auto& v : vectors) {
            store.add_vector(user_id++, v);
        }
        
        REQUIRE(store.get_index_current_count() == num_vectors);
        REQUIRE(store.get_next_label() == num_vectors);
        
        // Save the index
        store.save_index(index_path);
        
    } // store goes out of scope and is destroyed
    
    // Step 2: Load the index using the loading constructor
    {
        std::cout << "LOADING FROM FILE \n";
        VectorStore<L2Sim> loaded_store(index_path);
        std::cout << "LOADING FROM FILE DONE \n";
        
        // Verify dimensions and parameters
        REQUIRE(loaded_store.get_dim() == dimension);
        REQUIRE(loaded_store.get_index_current_count() == num_vectors);
        REQUIRE(loaded_store.get_next_label() == num_vectors);
        
        // Verify mappings were loaded correctly
        REQUIRE(loaded_store.get_id_to_label().size() == num_vectors);
        REQUIRE(loaded_store.get_label_to_id().size() >= num_vectors);
        
        // Verify specific user IDs exist
        auto id_to_label = loaded_store.get_id_to_label();
        for (uint64_t user_id = 100; user_id < 100 + num_vectors; ++user_id) {
            REQUIRE(id_to_label.find(user_id) != id_to_label.end());
        }
        
        // Verify search functionality works on loaded index
        std::vector<float> query = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        auto results = loaded_store.search_vectors(query, 3);
        
        REQUIRE(results.size() == 3);
        REQUIRE(results[0].user_id == 100); // Should match first vector (user_id 100)
    }
    
    // Clean up test files
    std::remove((index_path + ".hnsw").c_str());
    std::remove((index_path + ".hnsw.map").c_str());
    std::remove((index_path + ".hnsw.meta").c_str());
}

TEST_CASE("VectorStore loading constructor throws on missing index file", "[vector_store][load][error]") {
    const std::string nonexistent_path = "nonexistent_index_12345";
    
    // Should throw when trying to load non-existent index
    REQUIRE_THROWS_AS(
        VectorStore<L2Sim>(nonexistent_path),
        std::runtime_error
    );
}
#if 0
TEST_CASE("VectorStore loading constructor throws on missing mapping file", "[vector_store][load][error]") {
    const std::string index_path = "test_missing_map";
    const int dimension = 5;
    
    // ✅ FIRST: Clean up any existing files from previous runs
    std::remove((index_path + ".hnsw").c_str());
    std::remove((index_path + ".hnsw.map").c_str());
    std::remove((index_path + ".hnsw.meta").c_str());
    
    // Create and save only the index file, not the mapping
    {
        VectorStore<L2Sim> store(index_path, dimension);
        std::vector<float> vec = {1.0, 2.0, 3.0, 4.0, 5.0};
        store.add_vector(1, vec);
        
        // Save only the hnsw index and metadata
        store.save_metadata(index_path);
        store.get_index().saveIndex(index_path + ".hnsw");
        // Note: NOT saving mappings!
    }
    
    // ✅ Verify the map file doesn't exist
    std::ifstream check(index_path + ".hnsw.map");
    REQUIRE(!check.good());  // Map file should NOT exist
    
    // Should throw when trying to load without mapping file
    REQUIRE_THROWS_AS(VectorStore<L2Sim>(index_path), std::runtime_error);
    
    // Clean up
    std::remove((index_path + ".hnsw").c_str());
    std::remove((index_path + ".hnsw.meta").c_str());
}
#endif

TEST_CASE("VectorStore preserves data integrity after save/load cycle", "[vector_store][load][integrity]") {
    const std::string index_path = "test_data_integrity";
    const int dimension = 8;
    
    // Original data
    std::vector<std::pair<uint64_t, std::vector<float>>> original_data = {
        {1001, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}},
        {1002, {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f}},
        {1003, {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f}},
    };
    
    std::vector<float> query = {0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f, 0.85f};
    std::vector<uint64_t> original_results;
    
    // Step 1: Create, populate, search, and save
    {
        VectorStore<L2Sim> store(index_path, dimension);
        for (const auto& [user_id, vec] : original_data) {
            store.add_vector(user_id, vec);
        }
        
        auto results = store.search_vectors(query, 3);
        for (const auto& r : results) {
            original_results.push_back(r.user_id);
        }
        
        store.save_index(index_path);
    }
    
    // Step 2: Load and verify results are identical
    {
        VectorStore<L2Sim> loaded_store(index_path);
        
        auto loaded_results = loaded_store.search_vectors(query, 3);
        
        REQUIRE(loaded_results.size() == original_results.size());
        for (size_t i = 0; i < loaded_results.size(); ++i) {
            REQUIRE(loaded_results[i].user_id == original_results[i]);
        }
    }
    
    // Clean up
    std::remove((index_path + ".hnsw").c_str());
    std::remove((index_path + ".hnsw.map").c_str());
    std::remove((index_path + ".hnsw.meta").c_str());
}

TEST_CASE("VectorStore try_add_vector_batch adds all valid vectors", "[vector_store][batch]") {
    VectorStore<L2Sim> store("test_batch", 5);
    
    // Create a batch of valid vectors
    VectorStore<L2Sim>::VECTOR_BATCH batch = {
        {1, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}},
        {2, {1.1f, 2.1f, 3.1f, 4.1f, 5.1f}},
        {3, {1.2f, 2.2f, 3.2f, 4.2f, 5.2f}},
        {4, {1.3f, 2.3f, 3.3f, 4.3f, 5.3f}},
        {5, {1.4f, 2.4f, 3.4f, 4.4f, 5.4f}}
    };
    
    // Add batch
    auto added = store.try_add_vector_batch(batch, true);
    
    // All should be added
    REQUIRE(added.size() == 5);
    REQUIRE(store.get_index_current_count() == 5);
    REQUIRE(store.get_next_label() == 5);
    
    // Verify all user_ids were added
    for (uint64_t id = 1; id <= 5; ++id) {
        REQUIRE(std::find(added.begin(), added.end(), id) != added.end());
    }
}

TEST_CASE("VectorStore try_add_vector_batch skips duplicate user_ids", "[vector_store][batch]") {
    VectorStore<L2Sim> store("test_batch", 3);
    
    // Add one vector first
    store.add_vector(2, {1.0f, 2.0f, 3.0f});
    
    // Create batch with duplicate user_id
    VectorStore<L2Sim>::VECTOR_BATCH batch = {
        {1, {1.1f, 2.1f, 3.1f}},  // New - should succeed
        {2, {1.2f, 2.2f, 3.2f}},  // Duplicate - should skip
        {3, {1.3f, 2.3f, 3.3f}},  // New - should succeed
        {4, {1.4f, 2.4f, 3.4f}}   // New - should succeed
    };
    
    // Add batch with validation
    auto added = store.try_add_vector_batch(batch, true);
    
    // Should add 3 (skip the duplicate)
    REQUIRE(added.size() == 3);
    REQUIRE(store.get_index_current_count() == 4); // 1 original + 3 from batch
    
    // Check which were added
    REQUIRE(std::find(added.begin(), added.end(), 1) != added.end());
    REQUIRE(std::find(added.begin(), added.end(), 2) == added.end()); // Skipped
    REQUIRE(std::find(added.begin(), added.end(), 3) != added.end());
    REQUIRE(std::find(added.begin(), added.end(), 4) != added.end());
}

TEST_CASE("VectorStore try_add_vector_batch skips wrong dimension vectors", "[vector_store][batch]") {
    VectorStore<L2Sim> store("test_batch", 4);
    
    // Create batch with mixed dimensions
    VectorStore<L2Sim>::VECTOR_BATCH batch = {
        {1, {1.0f, 2.0f, 3.0f, 4.0f}},        // Correct - should succeed
        {2, {1.1f, 2.1f, 3.1f}},              // Wrong (too short) - should skip
        {3, {1.2f, 2.2f, 3.2f, 4.2f}},        // Correct - should succeed
        {4, {1.3f, 2.3f, 3.3f, 4.3f, 5.3f}}, // Wrong (too long) - should skip
        {5, {1.4f, 2.4f, 3.4f, 4.4f}}         // Correct - should succeed
    };
    
    // Add batch with validation
    auto added = store.try_add_vector_batch(batch, true);
    
    // Should add 3 (skip the 2 wrong dimensions)
    REQUIRE(added.size() == 3);
    REQUIRE(store.get_index_current_count() == 3);
    
    // Check which were added
    REQUIRE(std::find(added.begin(), added.end(), 1) != added.end());
    REQUIRE(std::find(added.begin(), added.end(), 2) == added.end()); // Skipped
    REQUIRE(std::find(added.begin(), added.end(), 3) != added.end());
    REQUIRE(std::find(added.begin(), added.end(), 4) == added.end()); // Skipped
    REQUIRE(std::find(added.begin(), added.end(), 5) != added.end());
}

TEST_CASE("VectorStore try_add_vector_batch with validation off adds invalid data", "[vector_store][batch]") {
    VectorStore<L2Sim> store("test_batch", 3);
    
    // Add one vector first
    store.add_vector(2, {1.0f, 2.0f, 3.0f});
    
    // Create batch with duplicate (but validation will be off)
    VectorStore<L2Sim>::VECTOR_BATCH batch = {
        {1, {1.1f, 2.1f, 3.1f}},  // New
        {2, {1.2f, 2.2f, 3.2f}},  // Duplicate - would normally skip, but validation is off
        {3, {1.3f, 2.3f, 3.3f}}   // New
    };
    
    // Add batch WITHOUT validation
    auto added = store.try_add_vector_batch(batch, false);
    
    // With validation off, all should be added (even duplicate)
    // Note: This overwrites the mapping for user_id 2
    REQUIRE(added.size() == 3);
    REQUIRE(store.get_index_current_count() == 4); // 1 original + 3 from batch
}

TEST_CASE("VectorStore try_add_vector_batch handles empty batch", "[vector_store][batch]") {
    VectorStore<L2Sim> store("test_batch", 3);
    
    // Empty batch
    VectorStore<L2Sim>::VECTOR_BATCH batch;
    
    // Should return empty vector without throwing
    auto added = store.try_add_vector_batch(batch, true);
    
    REQUIRE(added.empty());
    REQUIRE(store.get_index_current_count() == 0);
}

TEST_CASE("VectorStore try_add_vector_batch handles mixed valid/invalid vectors", "[vector_store][batch]") {
    VectorStore<L2Sim> store("test_batch", 5);
    
    // Add a couple vectors first
    store.add_vector(10, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    store.add_vector(20, {1.1f, 2.1f, 3.1f, 4.1f, 5.1f});
    
    // Create complex batch with various issues
    VectorStore<L2Sim>::VECTOR_BATCH batch = {
        {1, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}},        // Valid - should add
        {10, {1.1f, 2.1f, 3.1f, 4.1f, 5.1f}},       // Duplicate - should skip
        {2, {1.2f, 2.2f, 3.2f, 4.2f, 5.2f}},        // Valid - should add
        {3, {1.3f, 2.3f, 3.3f}},                    // Wrong dimension - should skip
        {4, {1.4f, 2.4f, 3.4f, 4.4f, 5.4f}},        // Valid - should add
        {20, {1.5f, 2.5f, 3.5f, 4.5f, 5.5f}},       // Duplicate - should skip
        {5, {1.6f, 2.6f, 3.6f, 4.6f, 5.6f, 6.6f}},  // Wrong dimension - should skip
        {6, {1.7f, 2.7f, 3.7f, 4.7f, 5.7f}}         // Valid - should add
    };
    
    // Add batch
    auto added = store.try_add_vector_batch(batch, true);
    
    // Should add 4 valid ones (1, 2, 4, 6)
    REQUIRE(added.size() == 4);
    REQUIRE(store.get_index_current_count() == 6); // 2 original + 4 from batch
    
    // Verify correct ones were added
    REQUIRE(std::find(added.begin(), added.end(), 1) != added.end());
    REQUIRE(std::find(added.begin(), added.end(), 2) != added.end());
    REQUIRE(std::find(added.begin(), added.end(), 4) != added.end());
    REQUIRE(std::find(added.begin(), added.end(), 6) != added.end());
    
    // Verify incorrect ones were NOT added
    REQUIRE(std::find(added.begin(), added.end(), 10) == added.end());
    REQUIRE(std::find(added.begin(), added.end(), 20) == added.end());
    REQUIRE(std::find(added.begin(), added.end(), 3) == added.end());
    REQUIRE(std::find(added.begin(), added.end(), 5) == added.end());
}

TEST_CASE("VectorStore try_add_vector_batch can search added vectors", "[vector_store][batch]") {
    VectorStore<L2Sim> store("test_batch", 4);
    
    // Create batch
    VectorStore<L2Sim>::VECTOR_BATCH batch = {
        {100, {1.0f, 2.0f, 3.0f, 4.0f}},
        {101, {1.1f, 2.1f, 3.1f, 4.1f}},
        {102, {1.2f, 2.2f, 3.2f, 4.2f}}
    };
    
    // Add batch
    auto added = store.try_add_vector_batch(batch, true);
    REQUIRE(added.size() == 3);
    
    // Search for vector close to first one
    std::vector<float> query = {1.0f, 2.0f, 3.0f, 4.0f};
    auto results = store.search_vectors(query, 2);
    
    REQUIRE(results.size() == 2);
    REQUIRE(results[0].user_id == 100); // Exact match should be first
}

TEST_CASE("VectorStore try_add_vector_batch respects capacity limit", "[vector_store][batch]") {
    // Create store with very small capacity
    VectorStore<L2Sim> store("test_batch", 3);
    // Note: max_elements is 10000 by default, so we can't easily test this
    // unless we modify the store or add many vectors
    
    // This is more of a documentation test showing the behavior
    // In real usage, if capacity is exceeded, the function stops adding
    VectorStore<L2Sim>::VECTOR_BATCH batch = {
        {1, {1.0f, 2.0f, 3.0f}},
        {2, {1.1f, 2.1f, 3.1f}},
        {3, {1.2f, 2.2f, 3.2f}}
    };
    
    auto added = store.try_add_vector_batch(batch, true);
    REQUIRE(added.size() == 3);
    REQUIRE(store.get_index_current_count() == 3);
}
