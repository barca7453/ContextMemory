# Build your library
add_library(memory_engine
    core/vector_store.cpp
    core/index_manager.cpp
    # ... more files
)

target_include_directories(memory_engine
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(memory_engine
    PRIVATE hnswlib::hnswlib
)
