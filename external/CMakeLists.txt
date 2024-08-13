include(FetchContent)

# Declare
FetchContent_Declare(cgltf
    GIT_REPOSITORY "https://github.com/jkuhlmann/cgltf"
    GIT_TAG c887e783d0f11046adda8848d5bd27aed5f9cb9b) # v.1.13

FetchContent_Declare(metal-cpp
    GIT_REPOSITORY "https://github.com/bkaradzic/metal-cpp"
    GIT_TAG a63bd172ddcba73a3d87ca32032b66ad41ddb9a6) # metal-cpp_macOS14.2_iOS17.2

FetchContent_Declare(fmt
    GIT_REPOSITORY "https://github.com/fmtlib/fmt"
    GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1

FetchContent_Declare(glfw
    GIT_REPOSITORY "https://github.com/glfw/glfw"
    GIT_TAG 7b6aead9fb88b3623e3b3725ebb42670cbe4c579) # 3.4

FetchContent_Declare(glm GIT_REPOSITORY "https://github.com/g-truc/glm"
    GIT_TAG bf71a834948186f4097caa076cd2663c69a10e1e) # v.0.9.9.8

FetchContent_Declare(imgui GIT_REPOSITORY "https://github.com/ocornut/imgui"
    GIT_TAG 8199457a7d9e453f8d3d9cadc14683fb54a858b5) # v1.91.0

FetchContent_Declare(stb
    GIT_REPOSITORY "https://github.com/nothings/stb"
    GIT_TAG beebb24b945efdea3b9bba23affb8eb3ba8982e7) # Oct 12, 2023

# Fetch
message(STATUS "Fetching cgltf...")
FetchContent_MakeAvailable(cgltf)

message(STATUS "Fetching metal-cpp...")
FetchContent_MakeAvailable(metal-cpp)

message(STATUS "Fetching fmt...")
FetchContent_MakeAvailable(fmt)

message(STATUS "Fetching glfw...")
FetchContent_MakeAvailable(glfw)

message(STATUS "Fetching glm...")
FetchContent_MakeAvailable(glm)

message(STATUS "Fetcing imgui...")
FetchContent_MakeAvailable(imgui)

message(STATUS "Fetching stb...")
FetchContent_MakeAvailable(stb)

# Configure
add_library(cgltf INTERFACE)
target_include_directories(cgltf INTERFACE ${cgltf_SOURCE_DIR})

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_metal.mm
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
)
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_METAL_CPP)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PRIVATE glfw metal-cpp)

add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE ${metal-cpp_SOURCE_DIR})

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})
