#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <fmt/core.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "cocoa_bridge.hpp"
#include "fly_camera_controller.hpp"
#include "gltf_model.hpp"
#include "render_config.hpp"
#include "renderer.hpp"
#include "ui_renderer.hpp"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string_view>

constexpr int              WIDTH = 640;
constexpr int              HEIGHT = 480;
constexpr std::string_view WINDOW_TITLE = "metal-raytracer";

namespace fs = std::filesystem;

void printHelp() { std::printf("Usage: metal-raytracer <input.glb>\n"); }

int main(int argc, char** argv)
try
{
    if (argc != 2)
    {
        printHelp();
        return 0;
    }

    fs::path gltfPath = argv[1];
    if (!fs::exists(gltfPath))
    {
        fmt::print(stderr, "File {} does not exist\n", gltfPath.string());
        return 1;
    }

    if (!glfwInit())
    {
        throw std::runtime_error("glfwInit failed");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, WINDOW_TITLE.data(), nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }

    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    auto layer = NS::TransferPtr(CA::MetalLayer::layer());
    layer->setDevice(device.get());
    layer->setDrawableSize(CGSizeMake(static_cast<float>(WIDTH), static_cast<float>(HEIGHT)));
    layer->setFramebufferOnly(true);
    layer->setPixelFormat(nlrs::COLOR_ATTACHMENT_FORMAT);
    nlrs::addLayerToGlfwWindow(window, layer.get());

    nlrs::GltfModel           model(gltfPath);
    nlrs::Renderer            renderer(device, model);
    nlrs::UiRenderer          uiRenderer(device);
    nlrs::FlyCameraController cameraController;
    cameraController.lookAt(glm::vec3(0.0f, 0.0f, 0.0f));

    int currentWidth = WIDTH;
    int currentHeight = HEIGHT;

    glfwMakeContextCurrent(window);
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        {
            // Handle resize
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            if (width != currentWidth || height != currentHeight)
            {
                currentWidth = width;
                currentHeight = height;
                layer->setDrawableSize(
                    CGSizeMake(static_cast<float>(width), static_cast<float>(height)));
            }
        }

        cameraController.update(window, 0.016f);

        {
            auto                     pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
            CA::MetalDrawable* const nextDrawable = layer->nextDrawable();
            renderer.draw(
                cameraController.getCamera(),
                static_cast<std::uint32_t>(currentWidth),
                static_cast<std::uint32_t>(currentHeight));
            uiRenderer.draw(nextDrawable, renderer.texture());
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
catch (const std::exception& e)
{
    std::printf("Error: %s\n", e.what());
}
