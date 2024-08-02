#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <cgltf.h>
#include <fmt/core.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <simd/simd.h>

#include "assert.hpp"
#include "cocoa_bridge.hpp"
#include "fly_camera_controller.hpp"
#include "gltf_model.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

constexpr MTL::PixelFormat COLOR_ATTACHMENT_FORMAT = MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB;
constexpr int              WIDTH = 640;
constexpr int              HEIGHT = 480;
constexpr std::string_view WINDOW_TITLE = "metal-raytracer";

namespace fs = std::filesystem;

namespace nlrs
{
namespace shader_types
{
struct Camera
{
    simd::float3 origin;
    simd::float3 lowerLeftCorner;
    simd::float3 horizontal;
    simd::float3 vertical;
    simd::float3 up;
    simd::float3 right;
};

struct Uniforms
{
    Camera camera;
};
} // namespace shader_types

void printHelp() { std::printf("Usage: metal-raytracer <input.glb>\n"); }

class Renderer
{
public:
    Renderer(NS::SharedPtr<MTL::Device> device, const GltfModel& model)
        : mDevice(std::move(device)),
          mCommandQueue(NS::TransferPtr(mDevice->newCommandQueue())),
          mHeap(),
          mPSO(),
          mVertexPositionsBuffer(),
          mUniformsBuffer(),
          mAccelerationStructure()
    {
        if (!mDevice)
        {
            throw std::runtime_error("Device was null");
        }

        if (!mDevice->supportsFamily(MTL::GPUFamilyMetal3))
        {
            throw std::runtime_error("Device does not support Metal 3");
        }

        if (MTL::ArgumentBuffersTier2 != mDevice->argumentBuffersSupport())
        {
            throw std::runtime_error("Device is not a tier 2 device for argument buffers");
        }

        if (!mCommandQueue)
        {
            throw std::runtime_error("Failed to create command queue");
        }

        {
            auto heapDesc = NS::TransferPtr(MTL::HeapDescriptor::alloc()->init());
            heapDesc->setStorageMode(MTL::StorageModePrivate);
            heapDesc->setHazardTrackingMode(MTL::HazardTrackingModeTracked);
            heapDesc->setType(MTL::HeapTypeAutomatic);
            heapDesc->setSize(1 << 30);
            mHeap = NS::TransferPtr(mDevice->newHeap(heapDesc.get()));
        }

        {
            using NS::StringEncoding::UTF8StringEncoding;

            const char* shaderSrc = R"(
                #include <metal_stdlib>
                #include <metal_geometric>
                #include <metal_raytracing>

                using namespace metal;

                struct VertexOutput
                {
                    float4 position [[position]];
                    float2 uv;
                };

                VertexOutput vertex vertexMain( uint vertexId [[vertex_id]],
                                    device const float2* positions [[buffer(0)]] )
                {
                    const float2 pos = positions[vertexId].xy;
                    const float2 uv = pos * float2(0.5, -0.5) + float2(0.5, 0.5);
                    VertexOutput out;
                    out.position = float4(pos, 0.f, 1.0);
                    out.uv = uv;
                    return out;
                }

                struct Camera {
                    float3 origin;
                    float3 lowerLeftCorner;
                    float3 horizontal;
                    float3 vertical;
                    float3 up;
                    float3 right;
                };

                struct Uniforms {
                    Camera camera;
                };

                raytracing::ray generateCameraRay(constant const Camera& camera, const float u, const float v) {
                    float3 origin = camera.origin;
                    float3 direction = normalize(camera.lowerLeftCorner + u * camera.horizontal + v * camera.vertical - origin);
                    return raytracing::ray(origin, direction);
                }

                half4 fragment fragmentMain( VertexOutput in [[stage_in]],
                                    constant const Uniforms& uniforms [[buffer(0)]],
                                    raytracing::acceleration_structure<> accelerationStructure [[buffer(1)]] )
                {
                    raytracing::intersector<raytracing::triangle_data> intersector;
                    const raytracing::ray ray = generateCameraRay(uniforms.camera, in.uv.x, 1.0 - in.uv.y);
                    typename raytracing::intersector<raytracing::triangle_data>::result_type intersection = intersector.intersect(ray, accelerationStructure);
                    if (intersection.type == raytracing::intersection_type::triangle) {
                        return half4(1.0, 0.0, 1.0, 1.0);
                    }
                    return half4(0.0, 0.0, 0.0, 1.0);
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
            constexpr std::size_t NUM_VERTICES = 6;

            simd::float2 quadPositions[NUM_VERTICES] = {
                // clang-format off
                {-1.0, -1.0,},
                {1.0, -1.0,},
                {1.0, 1.0,},
                {1.0, 1.0,},
                {-1.0, 1.0,},
                {-1.0, -1.0,},
                // clang-format on
            };
            const std::size_t positionsDataSize = NUM_VERTICES * sizeof(simd::float2);
            mVertexPositionsBuffer = NS::TransferPtr(
                mDevice->newBuffer(positionsDataSize, MTL::ResourceStorageModeManaged));
            std::memcpy(mVertexPositionsBuffer->contents(), quadPositions, positionsDataSize);
            // synchronize modified buffer sections to the GPU
            mVertexPositionsBuffer->didModifyRange(
                NS::Range::Make(0, mVertexPositionsBuffer->length()));
        }

        {
            mUniformsBuffer = NS::TransferPtr(mDevice->newBuffer(
                sizeof(shader_types::Uniforms), MTL::ResourceStorageModeManaged));
        }

        {
            // Build triangle geometry buffers

            std::vector<std::pair<NS::SharedPtr<MTL::Buffer>, NS::SharedPtr<MTL::Buffer>>> buffers;
            for (const auto& mesh : model.meshes)
            {
                const std::size_t numVertices = mesh.positions.size();
                const std::size_t numIndices = mesh.indices.size();

                const std::size_t vertexDataSize = numVertices * sizeof(glm::vec3);
                const std::size_t indexDataSize = numIndices * sizeof(std::uint32_t);

                auto vertexBuffer = NS::TransferPtr(
                    mDevice->newBuffer(vertexDataSize, MTL::ResourceStorageModeManaged));
                auto indexBuffer = NS::TransferPtr(
                    mDevice->newBuffer(indexDataSize, MTL::ResourceStorageModeManaged));

                std::memcpy(vertexBuffer->contents(), mesh.positions.data(), vertexDataSize);
                std::memcpy(indexBuffer->contents(), mesh.indices.data(), indexDataSize);

                vertexBuffer->didModifyRange(NS::Range::Make(0, vertexBuffer->length()));
                indexBuffer->didModifyRange(NS::Range::Make(0, indexBuffer->length()));

                buffers.emplace_back(std::move(vertexBuffer), std::move(indexBuffer));
            }

            // Build geometry descriptors

            std::vector<const NS::Object*> geometryDescriptors;
            geometryDescriptors.reserve(buffers.size());
            for (const auto& [vertexBuffer, indexBuffer] : buffers)
            {
                MTL::AccelerationStructureTriangleGeometryDescriptor* const triangleDesc =
                    MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
                triangleDesc->setVertexBuffer(vertexBuffer.get());
                triangleDesc->setVertexFormat(MTL::AttributeFormatFloat3);
                triangleDesc->setIndexBuffer(indexBuffer.get());
                triangleDesc->setIndexType(MTL::IndexTypeUInt32);
                triangleDesc->setTriangleCount(indexBuffer->length() / (sizeof(std::uint32_t) * 3));

                const NS::Object* const triangleDescObj =
                    static_cast<const NS::Object*>(triangleDesc);
                geometryDescriptors.push_back(triangleDescObj);
            }

            // Build primitive acceleration structure

            // TODO: are the pointers managed by the array or are they leaked?
            auto geometryDescriptorsArray = NS::TransferPtr(
                NS::Array::array(geometryDescriptors.data(), geometryDescriptors.size()));
            auto primitiveAccelerationStructureDesc =
                NS::TransferPtr(MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init());
            primitiveAccelerationStructureDesc->setGeometryDescriptors(
                geometryDescriptorsArray.get());

            const MTL::AccelerationStructureSizes sizes =
                mDevice->accelerationStructureSizes(primitiveAccelerationStructureDesc.get());
            auto scratchBuffer = NS::TransferPtr(
                mDevice->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModeManaged));
            // TODO: can this also be created using a device? What is the difference?
            mAccelerationStructure =
                NS::TransferPtr(mHeap->newAccelerationStructure(sizes.accelerationStructureSize));

            // TODO: autorelease pool?
            MTL::CommandBuffer* const commandBuffer = mCommandQueue->commandBuffer();
            MTL::AccelerationStructureCommandEncoder* const cmdEncoder =
                commandBuffer->accelerationStructureCommandEncoder();
            cmdEncoder->buildAccelerationStructure(
                mAccelerationStructure.get(),
                primitiveAccelerationStructureDesc.get(),
                scratchBuffer.get(),
                0);
            cmdEncoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        }
    }

    void draw(CA::MetalDrawable* drawable, const Camera& camera)
    {
        {
            auto* uniforms = reinterpret_cast<shader_types::Uniforms*>(mUniformsBuffer->contents());
            uniforms->camera = {
                .origin = {camera.origin.x, camera.origin.y, camera.origin.z},
                .lowerLeftCorner =
                    {camera.lowerLeftCorner.x, camera.lowerLeftCorner.y, camera.lowerLeftCorner.z},
                .horizontal = {camera.horizontal.x, camera.horizontal.y, camera.horizontal.z},
                .vertical = {camera.vertical.x, camera.vertical.y, camera.vertical.z},
                .up = {camera.up.x, camera.up.y, camera.up.z},
                .right = {camera.right.x, camera.right.y, camera.right.z}};
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
        renderEncoder->setFragmentBuffer(mUniformsBuffer.get(), 0, 0);
        renderEncoder->setFragmentAccelerationStructure(mAccelerationStructure.get(), 1);
        renderEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(6));

        renderEncoder->endEncoding();
        commandBuffer->presentDrawable(drawable);
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }

private:
    NS::SharedPtr<MTL::Device>                mDevice;
    NS::SharedPtr<MTL::CommandQueue>          mCommandQueue;
    NS::SharedPtr<MTL::Heap>                  mHeap;
    NS::SharedPtr<MTL::RenderPipelineState>   mPSO;
    NS::SharedPtr<MTL::Buffer>                mVertexPositionsBuffer;
    NS::SharedPtr<MTL::Buffer>                mUniformsBuffer;
    NS::SharedPtr<MTL::AccelerationStructure> mAccelerationStructure;
};
} // namespace nlrs

int main(int argc, char** argv)
try
{
    if (argc != 2)
    {
        nlrs::printHelp();
        return 0;
    }

    fs::path gltfPath = argv[1];
    if (!fs::exists(gltfPath))
    {
        // std::printf("Error: %s does not exist\n", gltfPath.c_str());
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
    layer->setPixelFormat(COLOR_ATTACHMENT_FORMAT);
    nlrs::addLayerToGlfwWindow(window, layer.get());

    nlrs::GltfModel           model(gltfPath);
    nlrs::Renderer            renderer(device, model);
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
            renderer.draw(nextDrawable, cameraController.getCamera());
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
