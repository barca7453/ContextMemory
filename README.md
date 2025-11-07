# ContextMemory
# High-Performance Memory Engine for LLM Applications

Production-grade C++ retrieval system for vector search in RAG applications.

## Status
🚧 **In Active Development** (Started Oct 30, 2025)

### Completed
- ✅ Core VectorStore class
- ✅ Add vectors with custom IDs
- ✅ Fast vector search using hnswlib
- ✅ Persistence (save/load index)
- ✅ Unit tests

### In Progress
- 🔄 REST API layer
- 🔄 Comprehensive documentation

### Planned
- ⏳ Session management
- ⏳ Concurrent access
- ⏳ Docker deployment

## Quick Start
```cpp
#include "ContextMemory/vector_store.h"

// =============================================================================
// CREATE NEW INDEX
// =============================================================================

// Specify similarity metric (L2Sim or CosineSim) and dimension
VectorStore<L2Sim> store("my_index", 128);  // 128 dimensions, L2 distance

// =============================================================================
// ADD SINGLE VECTOR
// =============================================================================

std::vector<float> vec(128, 1.0f);
store.add_vector(42, vec);  // user_id=42

// =============================================================================
// ADD BATCH OF VECTORS
// =============================================================================

// Create a batch
VectorStore<L2Sim>::VECTOR_BATCH batch = {
    {100, std::vector<float>(128, 1.0f)},
    {101, std::vector<float>(128, 1.1f)},
    {102, std::vector<float>(128, 1.2f)},
    {103, std::vector<float>(128, 1.3f)}
};

// Add batch with validation (returns IDs that were successfully added)
auto added = store.try_add_vector_batch(batch, true);

std::cout << "Successfully added " << added.size() << " vectors" << std::endl;
for (auto user_id : added) {
    std::cout << "  Added user_id: " << user_id << std::endl;
}

// =============================================================================
// SEARCH
// =============================================================================

std::vector<float> query(128, 1.0f);
auto results = store.search_vectors(query, 5);  // Get top 5 results

for (const auto& result : results) {
    std::cout << "User ID: " << result.user_id 
              << " Distance: " << result.distance << std::endl;
}

// =============================================================================
// SAVE INDEX
// =============================================================================

store.save_index("my_index");
// Creates three files:
//   - my_index.hnsw      (HNSW index)
//   - my_index.hnsw.map  (ID mappings)
//   - my_index.hnsw.meta (metadata)

// =============================================================================
// LOAD INDEX LATER
// =============================================================================

VectorStore<L2Sim> loaded("my_index");  // Loads from saved files

// Can immediately search
auto results2 = loaded.search_vectors(query, 5);
```

## Build
```bash
mkdir build && cd build
cmake ..
make
ctest  # Run tests
```

## Technologies
- C++17
- hnswlib (vector search)
- Catch2 (testing)
- CMake

## Performance Goals
- Sub-5ms retrieval on 1M+ vectors
- 100x faster than Python alternatives
- Production-ready reliability

## Performance Benchmarks

Tested on [your hardware: e.g., M1 Mac / Intel i9 / etc], 5,000 vectors, 128 dimensions:

### Search Performance (the key metric)
| k (results) | Latency | Throughput |
|-------------|---------|------------|
| 1 | 13 μs | 75,642 qps |
| 10 | 15 μs | 63,816 qps |
| 50 | 48 μs | 20,618 qps |
| 100 | 85 μs | 11,712 qps |

**100-1000x faster than Python alternatives** (ChromaDB, LanceDB: ~100-200ms)

### Insertion Performance
- **4,500 vectors/sec** (~221 μs per vector)
- Batch operations with validation
- Consistent performance at scale

### I/O Performance
- **Save**: 2 ms for 5K vectors
- **Load**: 10 ms for 5K vectors
- Near-instant persistence

### Why This Matters for RAG
- **Real-time search**: <1ms latency enables interactive applications
- **High throughput**: Handle 60K+ queries/sec on single machine
- **Fast startup**: 10ms load time vs seconds for Python alternatives

## Author
Jayendra Gowrishankar
