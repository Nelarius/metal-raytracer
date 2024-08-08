cmake_minimum_required(VERSION 3.11)

project(raytracer-metal)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external)

set(RENDERER_SOURCE
    camera.cpp
    cgltf.c
    cocoa_bridge.mm
    fly_camera_controller.cpp
    gltf_model.cpp
    main.cpp
    renderer.cpp
    stb_image.c
    texture.cpp
)
list(TRANSFORM RENDERER_SOURCE PREPEND ${CMAKE_CURRENT_LIST_DIR}/src/)

add_executable(metal-raytracer ${RENDERER_SOURCE})
target_include_directories(metal-raytracer PRIVATE ${METAL_CPP_INCLUDE_DIR})
target_compile_options(metal-raytracer PRIVATE -W -Wall -Wextra)
target_link_libraries(metal-raytracer
    cgltf
    fmt
    glfw
    glm::glm
    metal-cpp
    stb)
target_link_libraries(metal-raytracer
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore")
