#pragma once

#include <hnswlib/hnswlib.h>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <stdexcept>

struct CosineSim {

};

struct L2Sim {

};

template <typename SIM_POLICY>
class VectorStore {
private:
    static constexpr int LABEL_RESERVE_INCREMENT_SIZE = 1000;

    // hnswlib handles vector storage (binary, efficient)
    //std::unique_ptr<hnswlib::L2Space> space_;
    std::unique_ptr<SIM_POLICY> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;

    int dim_;
    int max_elements_;
    // HNSW parameters
    int M_; // the number of outgoing connections in the graph
    int ef_construction_; // the number of nearest neighbors to be returned as the result
    int ef_; // the number of nearest neighbors to be returned as the result
    bool allow_replace_deleted_; // whether to allow replacing of deleted elements with new added ones

    // Labels come from the embeddings
    // userId is user generated, can be anything, uuid, integer
    // the hnsw label however is dense (1,2,3,4,....)
    std::unordered_map<uint64_t, hnswlib::labeltype> id_to_label_;
    // Reverse vector: label → user_id (labels are dense 0,1,2,...)
    std::vector<uint64_t> label_to_id_;
    // the next label
    hnswlib::labeltype next_label_ = 0;
    size_t current_resized_label_vec_size_;

public:
    VectorStore(const std::string& index_path, const int dim)
        : dim_(dim), 
          max_elements_(10000), 
          M_(16), 
          ef_construction_(200), 
          ef_(10),
          allow_replace_deleted_(true), 
          current_resized_label_vec_size_(LABEL_RESERVE_INCREMENT_SIZE) {
        // Step 1: Create distance metric
        space_ = std::make_unique<hnswlib::L2Space>(dim_);
            
        // Step 2: Create index using that metric
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(),    // Pass pointer to space
            max_elements_,     // Max capacity,
            M_,
            ef_construction_,
            allow_replace_deleted_
        );
    }
    
    ~VectorStore() = default;
    
    void add_vector(uint64_t user_id, const std::vector<float>& vec) {
        if (id_to_label_.find(user_id) != id_to_label_.end()) {
            throw std::runtime_error("User ID already exists");
        }
        if (next_label_ >= max_elements_) {
            throw std::runtime_error("Exceeded max number of elements in the index");
        }
        
        // resize label_to_id_ if needed
        if (next_label_ >= current_resized_label_vec_size_) {
            current_resized_label_vec_size_ = next_label_ + LABEL_RESERVE_INCREMENT_SIZE;
            label_to_id_.resize(current_resized_label_vec_size_);
        }
        
        // Make sure addPoint succeeds before updating maps 
        index_->addPoint(vec.data(), next_label_);
        
        // Store mapping from user ID to hnsw label
        id_to_label_[user_id] = next_label_;
        label_to_id_.emplace_back(user_id);
        next_label_++;
    }
    
    std::vector<uint64_t> search_vectors(const std::vector<float>& vector, const size_t k) {
        // TODO: Implement search
        return {};
    }
    
    void save_mappings(const std::string& index_path) {
        // TODO: Implement
    }
    
    void load_mappings(const std::string& index_path) {
        // TODO: Implement
    }
};

