cmake_minimum_required(VERSION 3.15)
project(MemoryEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDE integration (clangd, IntelliSense)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(third_party)
add_subdirectory(src)
add_subdirectory(tests)
