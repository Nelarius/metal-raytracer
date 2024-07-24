#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <GLFW/glfw3.h>
#include <simd/simd.h>
#include <glm/glm.hpp>

#include "cocoa_bridge.hpp"
#include "fly_camera_controller.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>

constexpr MTL::PixelFormat COLOR_ATTACHMENT_FORMAT = MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB;
constexpr int              WIDTH = 640;
constexpr int              HEIGHT = 480;
constexpr std::string_view WINDOW_TITLE = "Hello, Metal";

namespace nlrs
{
namespace shader_types
{
struct Uniforms
{
    simd::float4x4 viewProjectionMatrix;
};
} // namespace shader_types

class Renderer
{
public:
    Renderer(NS::SharedPtr<MTL::Device> device)
        : mDevice(std::move(device)),
          mCommandQueue(NS::TransferPtr(mDevice->newCommandQueue())),
          mPSO(),
          mVertexPositionsBuffer(),
          mVertexColorsBuffer(),
          mUniformsBuffer()
    {
        if (!mDevice)
        {
            throw std::runtime_error("Device was null");
        }

        if (!mCommandQueue)
        {
            throw std::runtime_error("Failed to create command queue");
        }

        {
            using NS::StringEncoding::UTF8StringEncoding;

            const char* shaderSrc = R"(
                #include <metal_stdlib>
                using namespace metal;

                struct VertexOutput
                {
                    float4 position [[position]];
                    half3 color;
                };

                struct Uniforms {
                    float4x4 viewProjectionMatrix;
                };

                VertexOutput vertex vertexMain( uint vertexId [[vertex_id]],
                                    device const float4* positions [[buffer(0)]],
                                    device const float4* colors [[buffer(1)]],
                                    constant const Uniforms& uniforms [[buffer(2)]] )
                {
                    VertexOutput out;
                    out.position = uniforms.viewProjectionMatrix * positions[ vertexId ]; 
                    out.color = half3 ( colors[ vertexId ].xyz );
                    return out;
                }

                half4 fragment fragmentMain( VertexOutput in [[stage_in]] )
                {
                    return half4( in.color, 1.0 );
                }
            )";

            NS::Error* error = nullptr;
            auto       library = NS::TransferPtr(mDevice->newLibrary(
                NS::String::string(shaderSrc, UTF8StringEncoding), nullptr, &error));
            if (!library)
            {
                throw std::runtime_error(error->localizedDescription()->utf8String());
            }

            auto vertexFn = NS::TransferPtr(
                library->newFunction(NS::String::string("vertexMain", UTF8StringEncoding)));
            auto fragmentFn = NS::TransferPtr(
                library->newFunction(NS::String::string("fragmentMain", UTF8StringEncoding)));
            auto pipelineDesc = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
            pipelineDesc->setVertexFunction(vertexFn.get());
            pipelineDesc->setFragmentFunction(fragmentFn.get());
            pipelineDesc->colorAttachments()->object(0)->setPixelFormat(COLOR_ATTACHMENT_FORMAT);

            mPSO = NS::TransferPtr(mDevice->newRenderPipelineState(pipelineDesc.get(), &error));
            if (!mPSO)
            {
                throw std::runtime_error(error->localizedDescription()->utf8String());
            }
        }

        {
            constexpr std::size_t NUM_VERTICES = 3;

            simd::float4 positions[NUM_VERTICES] = {
                {-0.8f, 0.8f, 0.0f, 1.0f}, {0.0f, -0.8f, 0.0f, 1.0f}, {+0.8f, 0.8f, 0.0f, 1.0f}};

            simd::float4 colors[NUM_VERTICES] = {
                {1.0, 0.3f, 0.2f, 1.0f}, {0.8f, 1.0, 0.0f, 1.0f}, {0.8f, 0.0f, 1.0f, 1.0f}};

            const std::size_t positionsDataSize = NUM_VERTICES * sizeof(simd::float4);
            const std::size_t colorDataSize = NUM_VERTICES * sizeof(simd::float4);

            mVertexPositionsBuffer = NS::TransferPtr(
                mDevice->newBuffer(positionsDataSize, MTL::ResourceStorageModeManaged));
            mVertexColorsBuffer =
                NS::TransferPtr(mDevice->newBuffer(colorDataSize, MTL::ResourceStorageModeManaged));
            std::memcpy(mVertexPositionsBuffer->contents(), positions, positionsDataSize);
            std::memcpy(mVertexColorsBuffer->contents(), colors, colorDataSize);
            // synchronize modified buffer sections to the GPU
            mVertexPositionsBuffer->didModifyRange(
                NS::Range::Make(0, mVertexPositionsBuffer->length()));
            mVertexColorsBuffer->didModifyRange(NS::Range::Make(0, mVertexColorsBuffer->length()));

            mUniformsBuffer = NS::TransferPtr(mDevice->newBuffer(
                sizeof(shader_types::Uniforms), MTL::ResourceStorageModeManaged));
        }
    }

    void draw(CA::MetalDrawable* drawable, const glm::mat4& viewProjectionMat)
    {
        {
            auto uniforms = reinterpret_cast<shader_types::Uniforms*>(mUniformsBuffer->contents());
            uniforms->viewProjectionMatrix =
                *reinterpret_cast<const simd::float4x4*>(&viewProjectionMat[0][0]);
            mUniformsBuffer->didModifyRange(NS::Range::Make(0, sizeof(shader_types::Uniforms)));
        }

        MTL::RenderPassDescriptor* const renderPassDesc =
            MTL::RenderPassDescriptor::alloc()->init();
        // Objects created with new, alloc, Create, copy, mutableCopy should be managed either by
        // NS::SharedPtr, or using an autoreleasepool.
        renderPassDesc->autorelease();
        MTL::RenderPassColorAttachmentDescriptor* const colorAttachmentDesc =
            renderPassDesc->colorAttachments()->object(0);
        colorAttachmentDesc->setTexture(drawable->texture());
        colorAttachmentDesc->setLoadAction(MTL::LoadActionClear);
        colorAttachmentDesc->setClearColor(MTL::ClearColor::Make(0.2f, 0.25f, 0.3f, 1.0));
        colorAttachmentDesc->setStoreAction(MTL::StoreActionStore);

        MTL::CommandBuffer* const        commandBuffer = mCommandQueue->commandBuffer();
        MTL::RenderCommandEncoder* const renderEncoder =
            commandBuffer->renderCommandEncoder(renderPassDesc);

        renderEncoder->setRenderPipelineState(mPSO.get());
        renderEncoder->setVertexBuffer(mVertexPositionsBuffer.get(), 0, 0);
        renderEncoder->setVertexBuffer(mVertexColorsBuffer.get(), 0, 1);
        renderEncoder->setVertexBuffer(mUniformsBuffer.get(), 0, 2);
        renderEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));

        renderEncoder->endEncoding();
        commandBuffer->presentDrawable(drawable);
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }

private:
    NS::SharedPtr<MTL::Device>              mDevice;
    NS::SharedPtr<MTL::CommandQueue>        mCommandQueue;
    NS::SharedPtr<MTL::RenderPipelineState> mPSO;
    NS::SharedPtr<MTL::Buffer>              mVertexPositionsBuffer;
    NS::SharedPtr<MTL::Buffer>              mVertexColorsBuffer;
    NS::SharedPtr<MTL::Buffer>              mUniformsBuffer;
};
} // namespace nlrs

int main(int, char**)
try
{
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
    layer->setPixelFormat(COLOR_ATTACHMENT_FORMAT);
    nlrs::addLayerToGlfwWindow(window, layer.get());

    nlrs::Renderer            renderer(device);
    nlrs::FlyCameraController cameraController;
    cameraController.lookAt(glm::vec3(0.0f, 0.0f, 0.0f));

    glfwMakeContextCurrent(window);
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        cameraController.update(window, 0.016f);

        {
            auto                     pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
            CA::MetalDrawable* const nextDrawable = layer->nextDrawable();
            renderer.draw(nextDrawable, cameraController.viewProjectionMatrix());
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
