# Build your library
add_library(memory_engine
    core/index_manager.cpp
    # ... more files
    # Note: vector_store.cpp is not compiled as VectorStore is now a header-only template class
)

target_include_directories(memory_engine
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(memory_engine
    PUBLIC hnswlib::hnswlib
)
