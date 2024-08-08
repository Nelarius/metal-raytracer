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
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <tuple>
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

struct TextureDescriptor
{
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t offset;
    std::uint32_t padding;
};

struct PrimitiveData
{
    simd::float2      uv0;
    simd::float2      uv1;
    simd::float2      uv2;
    TextureDescriptor textureDescriptor;
};
} // namespace shader_types

void printHelp() { std::printf("Usage: metal-raytracer <input.glb>\n"); }

std::tuple<
    std::vector<Texture::BgraPixel>,
    std::vector<shader_types::PrimitiveData>,
    std::vector<std::uint32_t>>
buildPrimitiveData(const GltfModel& model)
{
    std::vector<std::uint32_t> primitiveDataOffsets;
    primitiveDataOffsets.reserve(model.meshes.size());

    std::vector<shader_types::PrimitiveData> primitiveData;
    primitiveData.reserve(1 << 10);

    std::vector<Texture::BgraPixel> textureData;
    textureData.reserve(1 << 20);
    // map base color textures to texture descriptors
    std::vector<shader_types::TextureDescriptor> textureDescriptors;
    textureDescriptors.reserve(model.baseColorTextures.size());
    for (const auto& texture : model.baseColorTextures)
    {
        const auto        dimensions = texture.dimensions();
        const auto        pixels = texture.pixels();
        const std::size_t offset = textureData.size();

        textureData.resize(textureData.size() + pixels.size());
        std::memcpy(textureData.data() + offset, pixels.data(), pixels.size_bytes());

        textureDescriptors.push_back({
            .width = dimensions.width,
            .height = dimensions.height,
            .offset = static_cast<std::uint32_t>(offset),
            .padding = 0,
        });
    }

    for (const auto& mesh : model.meshes)
    {
        const std::uint32_t primitiveIdxOffset = static_cast<std::uint32_t>(primitiveData.size());
        primitiveDataOffsets.push_back(primitiveIdxOffset);

        const shader_types::TextureDescriptor textureDescriptor =
            textureDescriptors[mesh.baseColorTextureIndex];

        for (std::size_t i = 0; i < mesh.indices.size(); i += 3)
        {
            const std::uint32_t i0 = mesh.indices[i + 0];
            const std::uint32_t i1 = mesh.indices[i + 1];
            const std::uint32_t i2 = mesh.indices[i + 2];

            const glm::vec2 uv0 = mesh.texCoords[i0];
            const glm::vec2 uv1 = mesh.texCoords[i1];
            const glm::vec2 uv2 = mesh.texCoords[i2];

            primitiveData.push_back({
                .uv0 = {uv0.x, uv0.y},
                .uv1 = {uv1.x, uv1.y},
                .uv2 = {uv2.x, uv2.y},
                .textureDescriptor = textureDescriptor,
            });
        }
    }

    return std::make_tuple(
        std::move(textureData), std::move(primitiveData), std::move(primitiveDataOffsets));
}

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
          mTextureBuffer(),
          mPrimitiveBuffer(),
          mPrimitiveBufferOffsets(),
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

                struct TextureDescriptor {
                    uint32_t width;
                    uint32_t height;
                    uint32_t offset;
                    uint32_t padding;
                };

                struct PrimitiveData {
                    float2 uv0;
                    float2 uv1;
                    float2 uv2;
                    TextureDescriptor textureDescriptor;
                };

                raytracing::ray generateCameraRay(constant const Camera& camera, const float u, const float v) {
                    float3 origin = camera.origin;
                    float3 direction = normalize(camera.lowerLeftCorner + u * camera.horizontal + v * camera.vertical - origin);
                    return raytracing::ray(origin, direction);
                }

                half3 textureLookup(device const uint32_t* textures, TextureDescriptor desc, float2 uv) {
                    const float u = fract(uv.x);
                    const float v = fract(uv.y);

                    const uint32_t j = uint32_t(u * float(desc.width));
                    const uint32_t i = uint32_t(v * float(desc.height));
                    const uint32_t idx = i * desc.width + j;

                    const uint32_t bgra = textures[desc.offset + idx];
                    const float3 srgb = float3(float((bgra >> 16u) & 0xffu), float((bgra >> 8u) & 0xffu), float(bgra & 0xffu)) / 255.0f;
                    const float3 linearRgb = pow(srgb, float3(2.2f));
                    return half3(linearRgb);
                }

                half4 fragment fragmentMain( VertexOutput in [[stage_in]],
                                    constant const Uniforms& uniforms [[buffer(0)]],
                                    raytracing::acceleration_structure<> accelerationStructure [[buffer(1)]],
                                    device const uint32_t* textureData [[buffer(2)]],
                                    device const PrimitiveData* primitiveData [[buffer(3)]],
                                    device const uint32_t* primitiveDataOffsets [[buffer(4)]] )
                {
                    half4 color = half4(0.0, 0.0, 0.0, 1.0);
                    raytracing::intersector<raytracing::triangle_data> intersector;
                    const raytracing::ray ray = generateCameraRay(uniforms.camera, in.uv.x, 1.0 - in.uv.y);
                    typename raytracing::intersector<raytracing::triangle_data>::result_type intersection = intersector.intersect(ray, accelerationStructure);
                    if (intersection.type == raytracing::intersection_type::triangle) {
                        const uint32_t primitiveIdx = intersection.primitive_id;
                        const uint32_t geometryIdx = intersection.geometry_id;
                        const uint32_t primitiveBufferOffset = primitiveDataOffsets[geometryIdx];
                        const float2 barycentricCoord = intersection.triangle_barycentric_coord;   // raytracing::triangle_data tag
                        device const PrimitiveData& primitive = primitiveData[primitiveBufferOffset + primitiveIdx];
                        const float2 uv0 = primitive.uv0;
                        const float2 uv1 = primitive.uv1;
                        const float2 uv2 = primitive.uv2;
                        const float2 uv = uv1 * barycentricCoord.x + uv2 * barycentricCoord.y + uv0 * (1.0 - barycentricCoord.x - barycentricCoord.y);
                        const half3 rgb = textureLookup(textureData, primitive.textureDescriptor, uv);
                        color = half4(rgb, 1.0);
                    }
                    return color;
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
            const auto [textureData, primitiveData, primitiveDataOffsets] =
                buildPrimitiveData(model);
            const std::size_t textureDataSize = textureData.size() * sizeof(Texture::BgraPixel);
            const std::size_t primitiveDataSize =
                primitiveData.size() * sizeof(shader_types::PrimitiveData);
            mTextureBuffer = NS::TransferPtr(
                mDevice->newBuffer(textureDataSize, MTL::ResourceStorageModeManaged));
            std::memcpy(mTextureBuffer->contents(), textureData.data(), textureDataSize);
            mTextureBuffer->didModifyRange(NS::Range::Make(0, mTextureBuffer->length()));
            mPrimitiveBuffer = NS::TransferPtr(
                mDevice->newBuffer(primitiveDataSize, MTL::ResourceStorageModeManaged));
            std::memcpy(mPrimitiveBuffer->contents(), primitiveData.data(), primitiveDataSize);
            mPrimitiveBuffer->didModifyRange(NS::Range::Make(0, mPrimitiveBuffer->length()));
            mPrimitiveBufferOffsets = NS::TransferPtr(mDevice->newBuffer(
                primitiveDataOffsets.size() * sizeof(std::uint32_t),
                MTL::ResourceStorageModeManaged));
            std::memcpy(
                mPrimitiveBufferOffsets->contents(),
                primitiveDataOffsets.data(),
                primitiveDataOffsets.size() * sizeof(std::uint32_t));
            mPrimitiveBufferOffsets->didModifyRange(
                NS::Range::Make(0, mPrimitiveBufferOffsets->length()));
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
        renderEncoder->setFragmentBuffer(mTextureBuffer.get(), 0, 2);
        renderEncoder->setFragmentBuffer(mPrimitiveBuffer.get(), 0, 3);
        renderEncoder->setFragmentBuffer(mPrimitiveBufferOffsets.get(), 0, 4);
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
    NS::SharedPtr<MTL::Buffer>                mTextureBuffer;
    NS::SharedPtr<MTL::Buffer>                mPrimitiveBuffer;
    NS::SharedPtr<MTL::Buffer>                mPrimitiveBufferOffsets;
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
