#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <shared_mutex>

template <typename CACHE_IMPL>
struct CacheShard {
    using K = typename CACHE_IMPL::key_type;
    using V = typename CACHE_IMPL::value_type;

    CacheShard (size_t shard_id, size_t num_records) : shard_id_(shard_id), 
        internal_cache_(num_records)
    {}

    void put (const K& key, const V& value) {
        std::unique_lock lock(shard_mutex_);
        internal_cache_.put(key, value);
    }

    std::optional<V> get(const K& key) {
        std::shared_lock lock(shard_mutex_);
        return internal_cache_.get(key);
    }

    void clear() {
        std::unique_lock lock(shard_mutex_);
        internal_cache_.clear();
    }

    bool contains(const K& key) const {
        std::shared_lock lock(shard_mutex_);
        return internal_cache_.contains(key);
    }

    size_t capacity() const {
        std::shared_lock lock(shard_mutex_);  // Or maybe no lock needed if immutable?
        return internal_cache_.capacity();
    }

    double hit_ratio() const {
        std::shared_lock lock(shard_mutex_);
        return internal_cache_.hit_ratio();
    }

    private:
      size_t shard_id_;
      mutable std::shared_mutex shard_mutex_;
      CACHE_IMPL internal_cache_;
};