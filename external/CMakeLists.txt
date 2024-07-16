include(FetchContent)

# Declare
FetchContent_Declare(metal-cpp
    GIT_REPOSITORY "https://github.com/bkaradzic/metal-cpp"
    GIT_TAG a63bd172ddcba73a3d87ca32032b66ad41ddb9a6) # metal-cpp_macOS14.2_iOS17.2

# Fetch
message(STATUS "Fetching metal-cpp...")
FetchContent_MakeAvailable(metal-cpp)

# Configure
add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE ${metal-cpp_SOURCE_DIR})
