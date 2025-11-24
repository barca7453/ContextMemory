#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <ContextMemory/query_cache.h>
#include <string>

TEST_CASE("QueryCache basic put/get/contains flow", "[query_cache]") {
    constexpr size_t num_shards = 4;
    constexpr size_t shard_capacity = 5;
    QueryCache<size_t, std::string> cache(num_shards, shard_capacity);

    SECTION("Empty cache reports misses") {
        REQUIRE_FALSE(cache.contains(42));
        auto missing = cache.get(42);
        REQUIRE_FALSE(missing.has_value());
        REQUIRE(cache.hit_ratio() == Catch::Approx(0.0));
    }

    SECTION("Put makes key retrievable and tracked") {
        cache.put(100, "hundred");
        cache.put(200, "two-hundred");

        auto hit = cache.get(100);
        REQUIRE(hit.has_value());
        REQUIRE(hit.value() == "hundred");
        REQUIRE(cache.contains(100));

        auto miss = cache.get(999);
        REQUIRE_FALSE(miss.has_value());

        double ratio = cache.hit_ratio();
        REQUIRE(ratio >= 0.0);
        REQUIRE(ratio <= 1.0);
        REQUIRE(ratio > 0.0);  // we had at least one hit in this section
    }
}

