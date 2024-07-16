cmake_minimum_required(VERSION 3.11)

project(raytracer-metal)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external)

add_executable(hello-metal src/main.cpp src/cocoa_bridge.mm)
target_include_directories(hello-metal PRIVATE ${METAL_CPP_INCLUDE_DIR})
target_link_libraries(hello-metal
    metal-cpp
    glfw)
target_link_libraries(hello-metal
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore")
