#include "ContextMemory/vector_store.h"

VectorStore::VectorStore(const std::string& index_path, const int dim) 
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
        max_elements_     // Max capacity
    );
}


void VectorStore::add_vector(uint64_t user_id, const std::vector<float>& vec) {
    if (id_to_label_.find(user_id) != id_to_label_.end()) {
        throw std::runtime_error("User ID already exists");
    }
    if (next_label_ >= max_elements_) {
        throw std::runtime_error("Exceeded max number of elements in the index");
    }
    
    // resize label_to_id_ if needed
    if (next_label_ >= current_resized_label_vec_size_) {
        current_resized_label_vec_size_ = next_label_ + LABEL_RESERVE_INCREMENT_SIZE ;
        label_to_id_.resize(current_resized_label_vec_size_);
    }
    
    // Make sure addPoint succeeds before updating maps 
    index_->addPoint(vec.data(), next_label_);
    
    // Store mapping from user ID to hnsw label
    id_to_label_[user_id] = next_label_;
    label_to_id_.emplace_back(user_id);
    next_label_++;
}

std::vector<uint64_t> VectorStore::search_vectors(const std::vector<float>& vector, const size_t k) {
    // TODO: Implement search
    return {};
}

void VectorStore::save_mappings(const std::string& index_path) {
    // TODO: Implement
}

void VectorStore::load_mappings(const std::string& index_path) {
    // TODO: Implement
}
