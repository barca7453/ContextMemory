#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>
#include "lru.h"
#include "cache_shard.h"

template <typename K, typename V>
class QueryCache {
    public:
    QueryCache(size_t num_shards, size_t shard_capacity)
        : num_shards_(num_shards), shard_capacity_(shard_capacity) {
        shard_vec_.reserve(num_shards);
        // Precreate shards for thread safety, otherwise we will need a mutex for the map, which I want to avoid.
        for (size_t n = 0; n < num_shards; ++n) {
            shard_vec_.emplace_back(std::make_unique<CacheShard<LRU<K,V>>>(n, shard_capacity_));
        }
    } 

    void put(const K& key, const V& value) {
        // hash the shard id
        size_t shard_id = std::hash<K>{}(key) % num_shards_; 
        shard_vec_[shard_id]->put(key, value);
    }

    std::optional<V> get(const K& key) {
        // hash the key to the shard id
        size_t shard_id = std::hash<K>{}(key) % num_shards_;
        return shard_vec_[shard_id]->get(key);
    }


    bool contains(const K& key) {
        size_t shard_id = std::hash<K>{}(key) % num_shards_; 
        return shard_vec_[shard_id]->contains(key);
    }

    double hit_ratio() const {
        double total_hit_ratio = 0.0;
        for (const auto& shard : shard_vec_) {
            total_hit_ratio += shard->hit_ratio();
        }
        return (total_hit_ratio / num_shards_);  // Returns 0.0-1.0
    }

    private:
        size_t num_shards_;
        size_t shard_capacity_;
        std::vector<std::unique_ptr<CacheShard<LRU<K,V>>>> shard_vec_;
};