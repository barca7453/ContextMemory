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

