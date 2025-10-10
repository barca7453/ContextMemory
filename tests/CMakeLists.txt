add_executable(memory_engine_tests
    test_main.cpp
    core/test_vector_store.cpp
    # ... more test files
)

target_link_libraries(memory_engine_tests
    PRIVATE memory_engine
    PRIVATE Catch2::Catch2
)

# Enable CTest
include(CTest)

# Add Catch2 module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/third_party/Catch2/extras")
include(Catch)
catch_discover_tests(memory_engine_tests)
