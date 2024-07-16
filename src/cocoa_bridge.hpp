#ifndef COCOA_BRIDGE
#define COCOA_BRIDGE
#include <Metal/Metal.hpp>

// Source: https://stackoverflow.com/questions/76776751/how-to-use-metal-cpp-with-glfw

struct GLFWwindow;

namespace CA
{
class MetalLayer;
}

namespace nlrs
{
void addLayerToGlfwWindow(GLFWwindow* window, CA::MetalLayer* layer);
}
#endif
