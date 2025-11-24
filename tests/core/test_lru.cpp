#include <catch2/catch_test_macros.hpp>
#include <ContextMemory/lru.h>
#include <string>

TEST_CASE("LRU Basic Put and Get", "[lru]") {
    // Create a simple LRU cache with capacity of 3
    LRU<int, std::string> cache(3);
    
    // Put a value
    cache.put(1, "value1");
    
    // Get the value back
    auto result = cache.get(1);
    
    // Verify the value is present and correct
    REQUIRE(result.has_value());
    REQUIRE(result.value() == "value1");
    
    // Verify cache contains the key
    REQUIRE(cache.contains(1));
    
    // Try to get a non-existent key
    auto missing = cache.get(999);
    REQUIRE_FALSE(missing.has_value());
    REQUIRE_FALSE(cache.contains(999));
}

TEST_CASE("LRU over capacity", "[lru]") {
    // Create a simple LRU cache with capacity of 3
    LRU<int, std::string> cache(3);
    
    // Put a value
    cache.put(1, "value1");
    cache.put(2, "value2");
    cache.put(3, "value3");
    cache.put(4, "value4");
    
    // Get the value back
    auto result = cache.get(1);
    
    // Verify the value is gone 
    REQUIRE_FALSE(result.has_value());
    
    // Verify cache has evicted the oldest key 
    REQUIRE_FALSE(cache.contains(1));
    
    auto result4 = cache.get(4);
    REQUIRE(result4.has_value());
    // Try to get a non-existent key
    auto missing = cache.get(999);
    REQUIRE_FALSE(missing.has_value());
    REQUIRE_FALSE(cache.contains(999));
}

