#pragma once

#include <climits>
#include <fstream>
#include <functional>
#include <tuple>
#include <hnswlib/hnswlib.h>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
//#include <thread>
#include <stdexcept>


struct L2Sim {
    using SpaceType = hnswlib::L2Space;
    using IndexType = hnswlib::HierarchicalNSW<float>;
};

struct CosineSim {
    using SpaceType = hnswlib::InnerProductSpace;  // cosine similarity
    using IndexType = hnswlib::HierarchicalNSW<float>;
};

template <typename SIM_POLICY>
class VectorStore {
private:
    static constexpr int LABEL_RESERVE_INCREMENT_SIZE = 1000;

    // hnswlib handles vector storage (binary, efficient)
    std::unique_ptr<typename SIM_POLICY::SpaceType> space_;
    std::unique_ptr<typename SIM_POLICY::IndexType> index_;

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
    std::filesystem::path mapFilePath_;

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
        space_ = std::make_unique<typename SIM_POLICY::SpaceType>(dim_);
            
        // Step 2: Create index using that metric
        index_ = std::make_unique<typename SIM_POLICY::IndexType>(
            space_.get(),    // Pass pointer to space
            max_elements_,   // Max capacity,
            M_,
            ef_construction_,
            allow_replace_deleted_
        );
    }

    VectorStore (const std::string& path) : max_elements_(10000), M_(16), ef_construction_(200), ef_(10), allow_replace_deleted_(true), mapFilePath_(path) {
        load_mappings(path);
    }
    
    ~VectorStore() = default;

        void add_vector(uint64_t user_id, const std::vector<float>& vec) {
        if (vec.size() != dim_) {
            throw std::runtime_error("The dimension and vector sizes do not match.");
        }
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

    struct SearchResult {
      uint64_t user_id;
      uint64_t distance;
    };

    std::vector<SearchResult> search_vectors(const std::vector<float>& vector, const size_t k) {
        // Mismatched dimension
        if (vector.size() != dim_) {
            throw std::runtime_error("Query vector dimension mismatch");
        }

        // search in the index
        auto queryResult = index_->searchKnnCloserFirst(vector.data(), k); // TODO there is a filter that could be used, and I am not using it.
        // form the result vector
        std::vector<SearchResult> resultVec;
        resultVec.reserve(k);
        for (auto &queryResult: queryResult) {
            SearchResult res;
            res.user_id = queryResult.second;
            res.distance = label_to_id_[queryResult.first];
            resultVec.push_back(res);
        }
        // From the result (priority Q, I should first flip it)
        return resultVec;
    }

    void save_index(const std::string& index_path) const {
        index_->saveIndex(index_path);
        save_mappings(index_path);
        save_index_metadata(index_path);
    }

    void cleanMappings() {
        label_to_id_.clear();
        id_to_label_.clear();
        next_label_ = 0;
    }
    
    void save_mappings(const std::string& index_path)  const{
        std::ofstream map_file(index_path + ".map", std::ios::binary);
        if (!map_file.is_open()) {
            throw std::runtime_error("Failed to open map file for writing");
        }

        const auto vec_size = label_to_id_.size();
        map_file.write (reinterpret_cast<const char*>(&vec_size), sizeof (vec_size));
        // write the vector itself
        map_file.write (reinterpret_cast<const char*>(label_to_id_.data()), vec_size*sizeof(uint64_t));

        // Save the map by entry
        for (const auto &[user_id, label] : id_to_label_) {
            map_file.write(reinterpret_cast<const char*>(&user_id), sizeof(user_id));
            map_file.write(reinterpret_cast<const char*>(&label), sizeof(label));
        }
        map_file.close();
    }

    void save_index_metadata(const std::string& index_path) {
        // TODO: Implement
    }

    void load_mappings(const std::string& index_path) {
        std::ifstream map_file (index_path + ".map", std::ios::in);
        if (!map_file.is_open()) {
            throw std::runtime_error("Failed to open mapping file for reading.");
        }
        // Read the size
        // This is the NUMBER OF ENTRIES in the mapping.
        uint64_t vec_size = 0;
        map_file.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t)); 
        if (vec_size == 0) {
            throw std::runtime_error("Label to id vector seems to be empty.");
        }
        // Read the vector. Guard against a 0 size vector
        std::unique_ptr<char[]> vec_buffer = std::make_unique<char[]>(vec_size);
        map_file.read(reinterpret_cast<char*>(label_to_id_.data()), vec_size * sizeof(uint64_t));
        // Read the map
        id_to_label_.clear();
        for (int i = 0;i < vec_size; ++i){
            hnswlib::labeltype label;
            uint64_t user_id;

            map_file.read(reinterpret_cast<char*>(&user_id), sizeof(uint64_t));
            map_file.read(reinterpret_cast<char*>(&label), sizeof(uint64_t));
            id_to_label_[user_id] = label;
        }
        next_label_ = vec_size;
    }


    // getters
    std::unordered_map<uint64_t, hnswlib::labeltype> get_id_to_label() const {
        return id_to_label_;
    }
    std::vector<uint64_t> get_label_to_id() const {
        return label_to_id_;
    }
    int get_dim() const {
        return dim_;
    }
    int get_max_elements() const {
        return max_elements_;
    }
    int get_M() const {
        return M_;
    }
    int get_ef_construction() const {
        return ef_construction_;
    }
    int get_ef() const {
        return ef_;
    }
    bool get_allow_replace_deleted() const {
        return allow_replace_deleted_;
    }
    size_t get_current_resized_label_vec_size() const {
        return current_resized_label_vec_size_;
    }
    size_t get_index_current_count() const {
        //return index_->getCurrentCount();
        return index_->getCurrentElementCount();
    }
    size_t get_next_label() const {
        return next_label_;
    }
};

