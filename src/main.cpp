#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <fmt/core.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#define IMGUI_IMPL_METAL_CPP
#include <imgui_impl_metal.h>

#include "cocoa_bridge.hpp"
#include "fly_camera_controller.hpp"
#include "gltf_model.hpp"
#include "render_config.hpp"
#include "renderer.hpp"
#include "ui_renderer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numeric>
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

    IMGUI_CHECKVERSION();
    ImGuiContext* const imguiContext = ImGui::CreateContext();
    if (imguiContext == nullptr)
    {
        throw std::runtime_error("ImGui::CreateContext failed");
    }
    ImGui::SetCurrentContext(imguiContext);
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOther(window, true);
    ImGui_ImplMetal_Init(device.get());

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

        uiRenderer.newFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            struct RayTracingStats
            {
                float averageMillis;
                float p50Millis;
                float p75Millis;
            };
            auto       stats = RayTracingStats{0.f, 0.f, 0.f};
            const auto timings = renderer.rayTracingTimings();
            if (!timings.empty())
            {
                std::vector<float> sortedTimings(timings.begin(), timings.end());
                std::sort(sortedTimings.begin(), sortedTimings.end());
                stats.averageMillis =
                    std::accumulate(sortedTimings.begin(), sortedTimings.end(), 0.0f) /
                    static_cast<float>(sortedTimings.size());
                stats.p50Millis = sortedTimings[sortedTimings.size() / 2];
                stats.p75Millis = sortedTimings[sortedTimings.size() * 3 / 4];
            }

            ImGui::Begin("Rendering stats");
            ImGui::PlotHistogram(
                "Raytracing times (ms)",
                timings.data(),
                static_cast<int>(timings.size()),
                0,
                nullptr,
                0.0f,
                30.0f,
                ImVec2(0, 40));
            ImGui::Text(
                "Average: %.2f ms (%.2f FPS)", stats.averageMillis, 1000.0f / stats.averageMillis);
            ImGui::Text(
                "50th percentile: %.2f ms (%.2f FPS)", stats.p50Millis, 1000.0f / stats.p50Millis);
            ImGui::Text(
                "75th percentile: %.2f ms (%.2f FPS)", stats.p75Millis, 1000.0f / stats.p75Millis);
            ImGui::End();
        }

        if (!ImGui::GetIO().WantCaptureMouse)
        {
            cameraController.update(window, 1.0f / 60.0f);
        }

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

    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(imguiContext);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
catch (const std::exception& e)
{
    std::printf("Error: %s\n", e.what());
}
