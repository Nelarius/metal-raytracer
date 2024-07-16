#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <GLFW/glfw3.h>

#include "cocoa_bridge.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>

int main(int, char**)
try
{
    if (!glfwInit())
    {
        throw std::runtime_error("glfwInit failed");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(640, 480, "Hello, Metal", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }

    // NOTE: an autoreleasepool is not necessary here, because the NS::SharedPtr is calls release
    // deterministically
    // auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    auto layer = NS::TransferPtr(CA::MetalLayer::layer());
    layer->setDevice(device.get());
    layer->setFramebufferOnly(true);
    layer->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm);
    nlrs::addLayerToGlfwWindow(window, layer.get());

    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        auto                     framePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        CA::MetalDrawable* const nextDrawable = layer->nextDrawable();
        assert(nextDrawable);
        // TODO: is nextDrawable already marked for autorelease?
        // nextDrawable->autorelease();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
catch (const std::exception& e)
{
    std::printf("Error: %s\n", e.what());
}
