#pragma once
// LRU cache implementation
// Single threaded LRU core
// Pure LRU logic with no threading concerns. This is the building block.

#include <cstddef>
#include <optional>
#include <list>
#include <cassert>
#include <unordered_map>

template <typename K, typename V>
class LRU {
public:
    LRU (size_t capacity) : capacity_(capacity), hits_(0), total_gets_(0)
    {};
    
    // Key should reflect query user id and top N
    // Use a hashing function and input these 2 to get the key
    std::optional<V> get(const K& key) {
        total_gets_++;
        // if present  splice to the head
        if (auto iter = kvmap_.find(key);iter != kvmap_.end()) {
            auto list_iter = iter->second;
            lru_order_list_.splice(lru_order_list_.begin(),lru_order_list_,list_iter);
            hits_++;
            assert (list_iter->first == key);
            return list_iter->second;
        }
        return std::nullopt;
    }

    void put (const K& key, const V& val) {
        // Check if already present
        if (auto it = kvmap_.find(key);it != kvmap_.end()) {
          //  If present splice it to the beggining
          auto list_it = it->second;
          list_it->second = val;
          lru_order_list_.splice(lru_order_list_.begin(), lru_order_list_, list_it);
        } else {
            // If the cache is full, evict
            assert(lru_order_list_.size() == kvmap_.size());
            if (kvmap_.size() == capacity_) {
              // get the iterator of the last item
              auto &key = lru_order_list_.back().first;
              // remove from the map
              kvmap_.erase(key);
              // remove it from the list
              lru_order_list_.pop_back();
            }
            assert(kvmap_.size() <= capacity_);

            // New entry
            std::pair<K,V> node{key, val};
            // Add to the begining
            lru_order_list_.emplace_front(node);
            kvmap_[key] = lru_order_list_.begin();
        }
    }

    size_t capacity() const {
        return capacity_;
    }
    
    // Stats
    const double hit_ratio() const {
        if (total_gets_ == 0) {
            return 0.0;
        }
        return static_cast<double>(hits_)/total_gets_;
    }
    
    // Invalidate
    void clear() {
        lru_order_list_.clear();
        kvmap_.clear();
    }
    
    bool contains(const K& key) const {
        return kvmap_.find(key) != kvmap_.end();
    }

    using list_iterator = typename std::list<std::pair<K,V>>::iterator;
    using list_const_iterator = typename std::list<std::pair<K,V>>::const_iterator;

private:
    size_t capacity_;
    std::list<std::pair<K,V>> lru_order_list_;
    std::unordered_map<K,list_iterator> kvmap_;
    size_t hits_;
    size_t total_gets_;
};