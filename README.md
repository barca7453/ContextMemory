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
#include "VectorStore.hpp"

// Create new index
VectorStore store(128, 1000000);  // 128 dims, 1M capacity

// Add vectors
std::vector vec(128, 1.0f);
store.add_vector(42, vec);

// Search
auto results = store.search_vectors(query, 5);
for (const auto& result : results) {
    std::cout << "ID: " << result.id 
              << " Distance: " << result.distance << std::endl;
}

// Save
store.save("my_index");

// Load later
VectorStore loaded("my_index");
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

## Author
Jayendra Gowrishankar
