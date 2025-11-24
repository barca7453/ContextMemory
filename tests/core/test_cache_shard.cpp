#include <catch2/catch_test_macros.hpp>
#include <ContextMemory/cache_shard.h>
#include <ContextMemory/lru.h>
#include <string>

TEST_CASE("CacheShard Basic Put and Get", "[cache_shard]") {
    // Create a cache shard wrapping an LRU cache
    CacheShard<LRU<int, std::string>> shard(0, 3);  // shard_id=0, capacity=3
    
    // Put a value
    shard.put(1, "value1");
    
    // Get the value back
    auto result = shard.get(1);
    
    // Verify the value is present and correct
    REQUIRE(result.has_value());
    REQUIRE(result.value() == "value1");
    
    // Verify contains works
    REQUIRE(shard.contains(1));
    
    // Try to get a non-existent key
    auto missing = shard.get(999);
    REQUIRE_FALSE(missing.has_value());
    REQUIRE_FALSE(shard.contains(999));
    
    // Verify capacity
    REQUIRE(shard.capacity() == 3);
}

// Thread safety tests TODO
// 1. 2 threads writing series of records. Check to verify all exist
// 2. Prepopulate cache. 2 threads continuous reading. Valid reads
// 3. prepopulate cache. 3 threads - 2 readers, 1 writers going simultaneously. Verify consistency
// 4. OVERLAPPING KEYS
// 5. Cache Eviction under contention
// .     - Cache capacity: 10
// .  - Thread 1: Writes keys 1-20 (causes evictions)
// - Thread 2: Writes keys 21-40 (causes evictions)
// - Verify: Only 10 items remain, no crashes
