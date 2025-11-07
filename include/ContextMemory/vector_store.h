/**
 * @file vector_store.h
 * @brief Thread-safe vector store with HNSW indexing for similarity search
 * 
 * This file provides a high-level interface to hnswlib for approximate nearest neighbor search
 * with thread-safe operations and persistent storage.
 */

#pragma once

#include <climits>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sys/types.h>
#include <tuple>
#include <hnswlib/hnswlib.h>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
//#include <thread>
#include <shared_mutex>
#include <stdexcept>

/**
 * @brief L2 (Euclidean) distance similarity policy
 * 
 * Uses L2 distance metric for similarity computations.
 * Suitable for most general purpose vector similarity tasks.
 */
struct L2Sim {
    using SpaceType = hnswlib::L2Space;
    using IndexType = hnswlib::HierarchicalNSW<float>;
};

/**
 * @brief Cosine similarity policy
 * 
 * Uses inner product (cosine similarity) for similarity computations.
 * Particularly useful for normalized vectors and text embeddings.
 */
struct CosineSim {
    using SpaceType = hnswlib::InnerProductSpace;
    using IndexType = hnswlib::HierarchicalNSW<float>;
};

/**
 * @class VectorStore
 * @brief Thread-safe vector storage with HNSW-based similarity search
 * 
 * @tparam SIM_POLICY Similarity metric policy (L2Sim or CosineSim)
 * 
 * This class provides:
 * - Fast approximate nearest neighbor search using HNSW algorithm
 * - Thread-safe operations with read-write locking
 * - User-defined ID mapping (uint64_t user IDs to internal dense labels)
 * - Persistent storage (save/load to disk)
 * - Batch insertion for improved performance
 * 
 * Example usage:
 * @code
 *   VectorStore<L2Sim> store("my_index", 128);  // 128-dimensional vectors
 *   store.add_vector(1, my_vector);
 *   auto results = store.search_vectors(query, 5);
 *   store.save_index("my_index");
 * @endcode
 * 
 * Thread-safety:
 * - Multiple concurrent readers (search)
 * - Exclusive writer access (add/save/load)
 * - All public methods are thread-safe
 */
template <typename SIM_POLICY>
class VectorStore {
private:
    static constexpr int LABEL_RESERVE_INCREMENT_SIZE = 1000;

    mutable std::shared_mutex store_mutex_;

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
    std::filesystem::path indexPath_;

public:
    /**
     * @brief Construct a new empty vector store
     * 
     * Creates a new HNSW index with specified parameters.
     * No data is loaded; this creates an empty index ready for insertion.
     * 
     * @param index_path Logical name/path for this index (not used for new index)
     * @param dim Dimensionality of vectors to be stored
     * @param max_elements Maximum number of vectors that can be stored (default: 10,000)
     * 
     * @throws std::runtime_error If index creation fails
     * 
     * Default HNSW parameters:
     * - M: 16 (number of outgoing connections)
     * - ef_construction: 200 (search quality during construction)
     * - ef: 10 (search quality during query)
     */
    VectorStore(const std::string& index_path, const int dim, const int max_elements = 10000)
        : dim_(dim), 
          max_elements_(max_elements), 
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

    /**
     * @brief Load an existing vector store from disk
     * 
     * Loads a previously saved index including:
     * - HNSW index data (.hnsw file)
     * - ID mappings (.hnsw.map file)
     * - Metadata (.hnsw.meta file)
     * 
     * @param index_path Base path to the saved index (without extensions)
     * 
     * @throws std::runtime_error If any required file is missing or corrupted
     * 
     * Example:
     * @code
     *   VectorStore<L2Sim> store("my_index");  // Loads from my_index.hnsw, .map, .meta
     * @endcode
     */
    VectorStore (const std::string& index_path) {
        load_index_metadata(index_path);
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

        auto index_file_path = index_path + ".hnsw";
        index_->loadIndex(index_file_path, space_.get(), max_elements_);
        // load mappings
        load_mappings(index_path);
    }
    
    /**
     * @brief Destructor
     * 
     * Automatically releases all resources. Index is NOT automatically saved.
     * Call save_index() explicitly before destruction if persistence is needed.
     */
    ~VectorStore() = default;

    /**
     * @brief Add a single vector to the index
     * 
     * Adds a vector with a user-defined ID. The ID can be any uint64_t value
     * and will be mapped internally to a dense label for efficient storage.
     * 
     * @param user_id User-defined identifier (must be unique)
     * @param vec Vector data (must match dimensionality)
     * 
     * @throws std::runtime_error If:
     *   - Vector dimension doesn't match index dimension
     *   - User ID already exists
     *   - Maximum capacity exceeded
     * 
     * Thread-safety: Exclusive lock (blocks all other operations)
     * 
     * Example:
     * @code
     *   std::vector<float> vec(128, 1.0f);
     *   store.add_vector(42, vec);
     * @endcode
     */
    void add_vector(uint64_t user_id, const std::vector<float> &vec) {
      std::unique_lock lock(store_mutex_);

      if (vec.size() != dim_) {
        throw std::runtime_error(
            "The dimension and vector sizes do not match.");
      }
      if (id_to_label_.find(user_id) != id_to_label_.end()) {
        throw std::runtime_error("User ID already exists");
      }
      
      // Auto-resize HNSW index if we're approaching capacity
      if (next_label_ >= max_elements_) {
        size_t new_capacity = max_elements_ * 2;  // Double the capacity
        index_->resizeIndex(new_capacity);
        max_elements_ = new_capacity;
      }

      // resize label_to_id_ if needed
      if (next_label_ >= label_to_id_.size()) {
        label_to_id_.resize(next_label_ + LABEL_RESERVE_INCREMENT_SIZE);
      }

      // Make sure addPoint succeeds before updating maps
      index_->addPoint(vec.data(), next_label_);

      // Store mapping from user ID to hnsw label
      id_to_label_[user_id] = next_label_;
      label_to_id_[next_label_] = user_id;
      next_label_++;
    }

    /**
     * @brief Type alias for batch vector operations
     * 
     * Each element is a pair of (user_id, vector).
     */
    using VECTOR_BATCH = std::vector<std::pair<uint64_t, std::vector<float>>>;

    /**
     * @brief Add multiple vectors in a batch (best-effort)
     * 
     * Attempts to add all vectors in the batch. If validation is enabled,
     * skips vectors with duplicate IDs or wrong dimensions. Returns IDs
     * of successfully added vectors.
     * 
     * @param batch Vector of (user_id, vector) pairs to add
     * @param validate If true, validates each vector before adding (default: true)
     * 
     * @return Vector of user_ids that were successfully added
     * 
     * Behavior:
     * - Empty batch returns empty result (no error)
     * - Stops adding when capacity is reached
     * - With validation=true: skips duplicates and invalid dimensions
     * - With validation=false: attempts all (may corrupt state if duplicates exist)
     * - Catches hnswlib exceptions and continues with remaining vectors
     * 
     * Thread-safety: Exclusive lock (blocks all other operations)
     * 
     * Example:
     * @code
     *   VECTOR_BATCH batch = {
     *     {100, vec1},
     *     {101, vec2},
     *     {102, vec3}
     *   };
     *   auto added = store.try_add_vector_batch(batch);
     *   std::cout << "Added " << added.size() << " vectors" << std::endl;
     * @endcode
     */
    std::vector<uint64_t> try_add_vector_batch(const VECTOR_BATCH& batch, bool validate = true) {
      std::unique_lock lock(store_mutex_);

      if (batch.empty()) {
        return std::vector<uint64_t>();
        throw std::runtime_error("The batch is empty.");
      }

      // resize label_to_id_ if needed
      if (next_label_  >= label_to_id_.size()) {
        label_to_id_.resize(next_label_ + batch.size() + LABEL_RESERVE_INCREMENT_SIZE);
      }

      std::vector<uint64_t> added_points;
      added_points.reserve(batch.size());
      for (auto &point : batch) {
        // Auto-resize HNSW index if we're approaching capacity
        if (next_label_ >= max_elements_) {
            size_t new_capacity = max_elements_ * 2;  // Double the capacity
            try {
                index_->resizeIndex(new_capacity);
                max_elements_ = new_capacity;
            } catch (const std::exception& e) {
                // If resize fails, stop adding more vectors
                break;
            }
        }

        const auto& user_id = point.first;
        const auto& vec = point.second;
        if (validate) {
          if (id_to_label_.find(user_id) != id_to_label_.end()) {
            continue;
          }
          if (vec.size() != dim_) {
            continue;
          }
        }

        try {
          // Make sure addPoint succeeds before updating maps
          index_->addPoint(vec.data(), next_label_);
          // Store mapping from user ID to hnsw label
          id_to_label_[user_id] = next_label_;
          label_to_id_[next_label_] = user_id;
          next_label_++;
          added_points.push_back(user_id);
        } catch (...) {
            continue;
        }
      }
      return added_points;
    }

    /**
     * @struct SearchResult
     * @brief Result of a similarity search
     */
    struct SearchResult {
      uint64_t user_id;   ///< User-defined ID of the matched vector
      uint64_t distance;  ///< Distance/similarity score (lower is more similar for L2)
    };

    /**
     * @brief Search for k nearest neighbors
     * 
     * Performs approximate nearest neighbor search using HNSW algorithm.
     * Results are ordered by similarity (closest first).
     * 
     * @param vector Query vector (must match dimensionality)
     * @param k Number of nearest neighbors to return
     * 
     * @return Vector of SearchResult, ordered by similarity (best first)
     * 
     * @throws std::runtime_error If query vector dimension doesn't match
     * 
     * Notes:
     * - Returns fewer than k results if index contains fewer than k vectors
     * - Thread-safe for concurrent searches
     * - Does not modify the index
     * 
     * Thread-safety: Shared lock (allows concurrent reads)
     * 
     * Example:
     * @code
     *   std::vector<float> query(128, 0.5f);
     *   auto results = store.search_vectors(query, 5);
     *   for (const auto& r : results) {
     *     std::cout << "ID: " << r.user_id << " Distance: " << r.distance << std::endl;
     *   }
     * @endcode
     */
    std::vector<SearchResult> search_vectors(const std::vector<float> &vector,
                                             const size_t k) {
    
      std::shared_lock lock(store_mutex_);
      // Mismatched dimension
      if (vector.size() != dim_) {
        throw std::runtime_error("Query vector dimension mismatch");
      }

      // search in the index
      auto queryResult = index_->searchKnnCloserFirst(
          vector.data(), k); // TODO there is a filter that could be used, and I
                             // am not using it.
      // form the result vector
      std::vector<SearchResult> resultVec;
      resultVec.reserve(k);
      for (auto &result : queryResult) {
        SearchResult res;
        res.distance = result.first;
        res.user_id = label_to_id_[result.second];
        resultVec.push_back(res);
      }
      // From the result (priority Q, I should first flip it)
      return resultVec;
    }

    /**
     * @brief Save the complete index to disk
     * 
     * Saves all components:
     * - HNSW index data to <index_path>.hnsw
     * - ID mappings to <index_path>.hnsw.map
     * - Metadata to <index_path>.hnsw.meta
     * 
     * @param index_path Base path for saving (extensions added automatically)
     * 
     * @throws std::runtime_error If any file cannot be opened for writing
     * 
     * Thread-safety: Exclusive lock (blocks all operations during save)
     * 
     * Example:
     * @code
     *   store.save_index("my_index");
     *   // Creates: my_index.hnsw, my_index.hnsw.map, my_index.hnsw.meta
     * @endcode
     */
    void save_index(const std::string &index_path) const {
      std::unique_lock lock(store_mutex_);
      // This might need to be done in a single block
      index_->saveIndex(index_path + ".hnsw");
      save_mappings_unlocked(index_path);
      save_metadata_unlocked(index_path);
      // end block
    }

    /**
     * @brief Clear all ID mappings
     * 
     * Removes all user_id to label mappings and resets the label counter.
     * Does NOT modify the HNSW index itself - only the mapping layer.
     * 
     * Thread-safety: Exclusive lock
     * 
     * @warning After calling this, user IDs in search results will be invalid
     */
    void cleanMappings() {
      std::unique_lock lock(store_mutex_);
      // In a single block
      label_to_id_.clear();
      id_to_label_.clear();
      next_label_ = 0;
    }

    /**
     * @brief Save ID mappings to disk (with lock)
     * 
     * Public wrapper that acquires lock before saving mappings.
     * 
     * @param index_path Base path (will append .hnsw.map)
     * @throws std::runtime_error If file cannot be opened
     * 
     * Thread-safety: Exclusive lock
     */
    void save_mappings(const std::string &index_path) const {
      std::unique_lock lock(store_mutex_);
      save_mappings_unlocked(index_path);
    }

    /**
     * @brief Save ID mappings to disk (without lock)
     * 
     * Internal helper that saves mappings without acquiring lock.
     * Assumes caller holds appropriate lock.
     * 
     * @param index_path Base path (will append .hnsw.map)
     * @throws std::runtime_error If file cannot be opened
     * 
     * @note This is NOT thread-safe - for internal use only
     */
    void save_mappings_unlocked(const std::string &index_path) const {
      std::ofstream map_file(index_path + ".hnsw.map", std::ios::binary);
      if (!map_file.is_open()) {
        throw std::runtime_error("Failed to open map file for writing");
      }

      const uint64_t num_elements = next_label_;
      map_file.write(reinterpret_cast<const char *>(&num_elements),
                     sizeof(uint64_t));
      // write the vector itself
      map_file.write(reinterpret_cast<const char *>(label_to_id_.data()),
                     num_elements * sizeof(uint64_t));

      // Save the map by entry
      for (const auto &[user_id, label] : id_to_label_) {
        map_file.write(reinterpret_cast<const char *>(&user_id),
                       sizeof(user_id));
        map_file.write(reinterpret_cast<const char *>(&label), sizeof(label));
      }
      map_file.close();
    }

    /**
     * @brief Save index metadata to disk (with lock)
     * 
     * Saves dimension, capacity, and HNSW parameters.
     * 
     * @param index_path Base path (will append .hnsw.meta)
     * @throws std::runtime_error If file cannot be opened
     * 
     * Thread-safety: Exclusive lock
     */
    void save_metadata(const std::string &index_path) const {
      std::unique_lock lock(store_mutex_);
      save_metadata_unlocked(index_path);
    }

    /**
     * @brief Save metadata to disk (without lock)
     * 
     * Internal helper that saves metadata without acquiring lock.
     * Assumes caller holds appropriate lock.
     * 
     * @param index_path Base path (will append .hnsw.meta)
     * @throws std::runtime_error If file cannot be opened
     * 
     * @note This is NOT thread-safe - for internal use only
     */
    void save_metadata_unlocked(const std::string &index_path) const {
      std::ofstream metadataFile(index_path + ".hnsw.meta", std::ios::binary);
      if (!metadataFile.is_open()) {
        throw std::runtime_error("Unable to open metadata file for writing.");
      }

      metadataFile.write(reinterpret_cast<const char *>(&dim_), sizeof(uint64_t));
      metadataFile.write(reinterpret_cast<const char *>(&max_elements_),
                         sizeof(uint64_t));
      metadataFile.write(reinterpret_cast<const char *>(&M_), sizeof(uint64_t));
      metadataFile.write(reinterpret_cast<const char *>(&ef_construction_),
                         sizeof(uint64_t));
      metadataFile.write(reinterpret_cast<const char *>(&ef_), sizeof(uint64_t));
      metadataFile.write(reinterpret_cast<const char *>(&allow_replace_deleted_),
                         sizeof(char));
      metadataFile.write(
          reinterpret_cast<const char *>(&current_resized_label_vec_size_),
          sizeof(uint64_t));
      metadataFile.close();
    }

    /**
     * @brief Load index metadata from disk
     * 
     * Loads dimension, capacity, and HNSW parameters from .meta file.
     * Must be called before loading the index itself.
     * 
     * @param index_path Base path (will append .hnsw.meta)
     * 
     * @throws std::runtime_error If file cannot be opened or is corrupted
     * 
     * Thread-safety: Exclusive lock
     */
    void load_index_metadata(const std::string &index_path) {
      std::unique_lock lock(store_mutex_);

      std::ifstream metadata_file(index_path + ".hnsw.meta", std::ios::in);
      if (!metadata_file.is_open()) {
        auto message = "Unable to open file " + index_path + ".hnsw.meta for reading.";
        throw std::runtime_error(message);
      }
      metadata_file.read(reinterpret_cast<char *>(&dim_), sizeof(uint64_t));
      metadata_file.read(reinterpret_cast<char *>(&max_elements_), sizeof(uint64_t));
      metadata_file.read(reinterpret_cast<char *>(&M_), sizeof(uint64_t));
      metadata_file.read(reinterpret_cast<char *>(&ef_construction_), sizeof(uint64_t));
      metadata_file.read(reinterpret_cast<char *>(&ef_), sizeof(uint64_t));
      metadata_file.read(reinterpret_cast<char *>(&allow_replace_deleted_), sizeof(char));
      metadata_file.read(reinterpret_cast<char *>(&current_resized_label_vec_size_), sizeof(uint64_t));
      metadata_file.close();
    }

    /**
     * @brief Load ID mappings from disk
     * 
     * Loads user_id to label mappings from .map file.
     * Must be called after loading the HNSW index.
     * 
     * @param index_path Base path (will append .hnsw.map)
     * 
     * @throws std::runtime_error If:
     *   - File cannot be opened
     *   - File is empty or corrupted
     *   - Data format is invalid
     * 
     * Thread-safety: Exclusive lock
     */
    void load_mappings(const std::string &index_path) {
      std::unique_lock lock(store_mutex_);

      std::ifstream map_file(index_path + ".hnsw.map", std::ios::binary);
      if (!map_file.is_open()) {
        throw std::runtime_error("Failed to open mapping file for reading.");
      }
      // Read the size
      // This is the NUMBER OF ENTRIES in the mapping.
      uint64_t num_elements = 0;
      map_file.read(reinterpret_cast<char *>(&num_elements), sizeof(uint64_t));
      if (num_elements == 0) {
        throw std::runtime_error("Label to id vector seems to be empty.");
      }
      // Read the vector. Guard against a 0 size vector
      label_to_id_.resize(num_elements);
      map_file.read(reinterpret_cast<char *>(label_to_id_.data()),
                    num_elements * sizeof(uint64_t));
      // Read the map
      id_to_label_.clear();
      for (int i = 0; i < num_elements; ++i) {
        hnswlib::labeltype label;
        uint64_t user_id;

        map_file.read(reinterpret_cast<char *>(&user_id), sizeof(uint64_t));
        map_file.read(reinterpret_cast<char *>(&label), sizeof(uint64_t));
        id_to_label_[user_id] = label;
      }
      next_label_ = num_elements;
      map_file.close();
    }

    
    //=============================================================================
    // Setters - All thread-safe with shared locks (allow concurrent reads)
    //=============================================================================
    
    /**
     * @brief Set the ef parameter for search operations
     * 
     * The ef parameter controls the size of the dynamic candidate list during search.
     * Higher values improve recall (accuracy) but make searches slower.
     * 
     * @param ef New ef value (typical range: 10-500)
     * 
     * Trade-offs:
     * - ef=10:  Fast searches, low recall (~55-60% for large indices)
     * - ef=50:  Balanced, medium recall (~85-90%)
     * - ef=100: Slower, high recall (~95-98%)
     * - ef=200+: Very slow, very high recall (~99%+)
     * 
     * Note: This only affects search speed/quality, not the index itself.
     * Can be adjusted dynamically without rebuilding the index.
     * 
     * Thread-safety: Exclusive lock
     */
    void set_ef(size_t ef) {
        std::unique_lock lock(store_mutex_);
        ef_ = ef;
        index_->setEf(ef);  // Update both member and HNSW index
    }

    //=============================================================================
    // Getters - All thread-safe with shared locks (allow concurrent reads)
    //=============================================================================

    /**
     * @brief Get copy of user_id to label mapping
     * @return Map of user IDs to internal dense labels
     * Thread-safety: Shared lock
     */
    std::unordered_map<uint64_t, hnswlib::labeltype> get_id_to_label() const {
      std::shared_lock lock(store_mutex_);
      return id_to_label_;
    }

    /**
     * @brief Get copy of label to user_id mapping
     * @return Vector where index is label, value is user_id
     * Thread-safety: Shared lock
     */
    std::vector<uint64_t> get_label_to_id() const {
      std::shared_lock lock(store_mutex_);
      return label_to_id_;
    }

    /**
     * @brief Get vector dimensionality
     * @return Number of dimensions per vector
     * Thread-safety: Shared lock
     */
    int get_dim() const {
      std::shared_lock lock(store_mutex_);
      return dim_;
    }

    /**
     * @brief Get maximum capacity
     * @return Maximum number of vectors that can be stored
     * Thread-safety: Shared lock
     */
    int get_max_elements() const {
      std::shared_lock lock(store_mutex_);
      return max_elements_;
    }

    /**
     * @brief Get HNSW M parameter
     * @return Number of bi-directional links per node
     * Thread-safety: Shared lock
     */
    int get_M() const {
      std::shared_lock lock(store_mutex_);
      return M_;
    }

    /**
     * @brief Get ef_construction parameter
     * @return Number of candidates considered during index construction
     * Thread-safety: Shared lock
     */
    int get_ef_construction() const {
      std::shared_lock lock(store_mutex_);
      return ef_construction_;
    }

    /**
     * @brief Get ef parameter
     * @return Number of candidates considered during search
     * Thread-safety: Shared lock
     */
    int get_ef() const {
      std::shared_lock lock(store_mutex_);
      return ef_;
    }

    /**
     * @brief Check if deleted element replacement is allowed
     * @return True if new elements can replace deleted ones
     * Thread-safety: Shared lock
     */
    bool get_allow_replace_deleted() const {
      std::shared_lock lock(store_mutex_);
      return allow_replace_deleted_;
    }

    /**
     * @brief Get current label vector allocated size
     * @return Current capacity of label_to_id_ vector
     * Thread-safety: Shared lock
     */
    size_t get_current_resized_label_vec_size() const {
      std::shared_lock lock(store_mutex_);
      return current_resized_label_vec_size_;
    }

    /**
     * @brief Get number of vectors currently in index
     * @return Current element count
     * Thread-safety: Shared lock
     */
    size_t get_index_current_count() const {
      std::shared_lock lock(store_mutex_);
        //return index_->getCurrentCount();
      return index_->getCurrentElementCount();
    }

    /**
     * @brief Get next available label
     * @return Next internal label to be assigned
     * Thread-safety: Shared lock
     */
    size_t get_next_label() const {
      std::shared_lock lock(store_mutex_);
      return next_label_;
    }

    /*
     * Non-const get_index() removed to prevent bypassing thread-safety.
     * Exposing mutable reference to internal state breaks encapsulation
     * and allows callers to modify the index without proper locking.
     */

    /**
     * @brief Get const reference to underlying HNSW index
     * 
     * @return Const reference to the HNSW index
     * @throws std::runtime_error If index is not initialized
     * 
     * @warning Use with caution. The returned reference is only valid
     *          while the VectorStore exists. The lock is released after
     *          this function returns.
     * 
     * Thread-safety: Shared lock (released after return)
     * 
     * @note For internal/testing use. Prefer using the public API methods.
     */
    const typename SIM_POLICY::IndexType& get_index() const {
      std::shared_lock lock(store_mutex_);
      if (!index_) {
        throw std::runtime_error("Index is not initialized.");
      }
      return *index_;
    } 
};

