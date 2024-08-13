#include "camera.hpp"
#include "gltf_model.hpp"
#include "renderer.hpp"
#include "shader_types.hpp"
#include "render_config.hpp"
#include "texture.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nlrs
{
namespace
{
constexpr NS::UInteger SAMPLE_BUFFER_COUNT = 2;

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
} // namespace

Renderer::Renderer(NS::SharedPtr<MTL::Device> device, const GltfModel& model)
    : mDevice(std::move(device)),
      mCommandQueue(NS::TransferPtr(mDevice->newCommandQueue())),
      mTexture(),
      mHeap(),
      mPso(),
      mTimerSampleBuffer(),
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

        const char* const shaderSrc = R"(
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

        mPso = NS::TransferPtr(mDevice->newRenderPipelineState(pipelineDesc.get(), &error));
        if (!mPso)
        {
            throw std::runtime_error(error->localizedDescription()->utf8String());
        }
    }

    {
        const MTL::CounterSet* counterSetTimestamp = nullptr;
        const NS::Array&       counterSets = *mDevice->counterSets();
        for (NS::UInteger i = 0; i < counterSets.count(); ++i)
        {
            const MTL::CounterSet* const counterSet =
                static_cast<const MTL::CounterSet*>(counterSets.object(i));

            if (counterSet->name()->isEqualToString(MTL::CommonCounterSetTimestamp))
            {
                counterSetTimestamp = counterSet;
                break;
            }
        }
        if (counterSetTimestamp == nullptr)
        {
            throw std::runtime_error("Timestamp counter not supported by device");
        }

        const MTL::Counter* counterTimestamp = nullptr;
        const NS::Array&    counters = *counterSetTimestamp->counters();
        for (NS::UInteger i = 0; i < counters.count(); ++i)
        {
            const MTL::Counter* const counter =
                static_cast<const MTL::Counter*>(counters.object(i));
            if (counter->name()->isEqualToString(MTL::CommonCounterTimestamp))
            {
                counterTimestamp = counter;
                break;
            }
        }
        if (counterTimestamp == nullptr)
        {
            throw std::runtime_error("Timestamp counter set does not contain timestamp counter");
        }

        auto counterSampleBufferDesc =
            NS::TransferPtr(MTL::CounterSampleBufferDescriptor::alloc()->init());
        counterSampleBufferDesc->setCounterSet(counterSetTimestamp);
        counterSampleBufferDesc->setStorageMode(MTL::StorageModeShared);
        counterSampleBufferDesc->setSampleCount(SAMPLE_BUFFER_COUNT);

        NS::Error* error = nullptr;
        mTimerSampleBuffer =
            NS::TransferPtr(mDevice->newCounterSampleBuffer(counterSampleBufferDesc.get(), &error));
        if (error != nullptr)
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
        mVertexPositionsBuffer =
            NS::TransferPtr(mDevice->newBuffer(positionsDataSize, MTL::ResourceStorageModeManaged));
        std::memcpy(mVertexPositionsBuffer->contents(), quadPositions, positionsDataSize);
        // synchronize modified buffer sections to the GPU
        mVertexPositionsBuffer->didModifyRange(
            NS::Range::Make(0, mVertexPositionsBuffer->length()));
    }

    {
        mUniformsBuffer = NS::TransferPtr(
            mDevice->newBuffer(sizeof(shader_types::Uniforms), MTL::ResourceStorageModeManaged));
    }

    {
        const auto [textureData, primitiveData, primitiveDataOffsets] = buildPrimitiveData(model);
        const std::size_t textureDataSize = textureData.size() * sizeof(Texture::BgraPixel);
        const std::size_t primitiveDataSize =
            primitiveData.size() * sizeof(shader_types::PrimitiveData);
        mTextureBuffer =
            NS::TransferPtr(mDevice->newBuffer(textureDataSize, MTL::ResourceStorageModeManaged));
        std::memcpy(mTextureBuffer->contents(), textureData.data(), textureDataSize);
        mTextureBuffer->didModifyRange(NS::Range::Make(0, mTextureBuffer->length()));
        mPrimitiveBuffer =
            NS::TransferPtr(mDevice->newBuffer(primitiveDataSize, MTL::ResourceStorageModeManaged));
        std::memcpy(mPrimitiveBuffer->contents(), primitiveData.data(), primitiveDataSize);
        mPrimitiveBuffer->didModifyRange(NS::Range::Make(0, mPrimitiveBuffer->length()));
        mPrimitiveBufferOffsets = NS::TransferPtr(mDevice->newBuffer(
            primitiveDataOffsets.size() * sizeof(std::uint32_t), MTL::ResourceStorageModeManaged));
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
            auto indexBuffer =
                NS::TransferPtr(mDevice->newBuffer(indexDataSize, MTL::ResourceStorageModeManaged));

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

            const NS::Object* const triangleDescObj = static_cast<const NS::Object*>(triangleDesc);
            geometryDescriptors.push_back(triangleDescObj);
        }

        // Build primitive acceleration structure

        // TODO: are the pointers managed by the array or are they leaked?
        auto geometryDescriptorsArray = NS::TransferPtr(
            NS::Array::array(geometryDescriptors.data(), geometryDescriptors.size()));
        auto primitiveAccelerationStructureDesc =
            NS::TransferPtr(MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init());
        primitiveAccelerationStructureDesc->setGeometryDescriptors(geometryDescriptorsArray.get());

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

void Renderer::draw(const Camera& camera, const std::uint32_t width, const std::uint32_t height)
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

    if (!mTexture || mTexture->width() != width || mTexture->height() != height)
    {
        auto textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
            MTL::PixelFormatBGRA8Unorm, width, height, false);
        textureDesc->setPixelFormat(COLOR_ATTACHMENT_FORMAT);
        textureDesc->setStorageMode(MTL::StorageModePrivate);
        textureDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget);
        mTexture = NS::TransferPtr(mDevice->newTexture(textureDesc));
    }

    MTL::RenderPassDescriptor* const renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    // Objects created with new, alloc, Create, copy, mutableCopy should be managed either by
    // NS::SharedPtr, or using an autoreleasepool.
    renderPassDesc->autorelease();
    MTL::RenderPassColorAttachmentDescriptor* const colorAttachmentDesc =
        renderPassDesc->colorAttachments()->object(0);
    colorAttachmentDesc->setTexture(mTexture.get());
    colorAttachmentDesc->setLoadAction(MTL::LoadActionClear);
    colorAttachmentDesc->setClearColor(MTL::ClearColor::Make(0.2f, 0.25f, 0.3f, 1.0));
    colorAttachmentDesc->setStoreAction(MTL::StoreActionStore);
    MTL::RenderPassSampleBufferAttachmentDescriptor* const sampleBufferDesc =
        renderPassDesc->sampleBufferAttachments()->object(0);
    sampleBufferDesc->setSampleBuffer(mTimerSampleBuffer.get());
    sampleBufferDesc->setStartOfFragmentSampleIndex(0);
    sampleBufferDesc->setEndOfFragmentSampleIndex(1);

    MTL::CommandBuffer* const commandBuffer = mCommandQueue->commandBuffer();
    MTL::Timestamp            cpuStartTime, gpuStartTime;
    MTL::Timestamp            cpuEndTime, gpuEndTime;
    mDevice->sampleTimestamps(&cpuStartTime, &gpuStartTime);
    commandBuffer->addCompletedHandler([this, &cpuEndTime, &gpuEndTime](MTL::CommandBuffer*) {
        mDevice->sampleTimestamps(&cpuEndTime, &gpuEndTime);
    });
    MTL::RenderCommandEncoder* const renderEncoder =
        commandBuffer->renderCommandEncoder(renderPassDesc);

    renderEncoder->setRenderPipelineState(mPso.get());
    renderEncoder->setVertexBuffer(mVertexPositionsBuffer.get(), 0, 0);
    renderEncoder->setFragmentBuffer(mUniformsBuffer.get(), 0, 0);
    renderEncoder->setFragmentAccelerationStructure(mAccelerationStructure.get(), 1);
    renderEncoder->setFragmentBuffer(mTextureBuffer.get(), 0, 2);
    renderEncoder->setFragmentBuffer(mPrimitiveBuffer.get(), 0, 3);
    renderEncoder->setFragmentBuffer(mPrimitiveBufferOffsets.get(), 0, 4);
    renderEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(6));
    renderEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    NS::Data* const resolvedData =
        mTimerSampleBuffer->resolveCounterRange(NS::Range::Make(0, SAMPLE_BUFFER_COUNT));
    const MTL::Timestamp* const timestamps =
        static_cast<const MTL::Timestamp*>(resolvedData->mutableBytes());

    const auto absoluteTimeInUs = [gpuStartTime, gpuEndTime, cpuStartTime, cpuEndTime](
                                      MTL::Timestamp gpuTimestamp) -> double {
        // See:
        // https://developer.apple.com/documentation/metal/gpu_counters_and_counter_sample_buffers/converting_gpu_timestamps_into_cpu_time?language=objc
        const double gpuReferenceTimespan = static_cast<double>(gpuEndTime - gpuStartTime);
        const double cpuReferenceTimespan = static_cast<double>(cpuEndTime - cpuStartTime);
        const double normalizedGpuTime =
            static_cast<double>(gpuTimestamp - gpuStartTime) / gpuReferenceTimespan;

        // Convert GPU time to CPU time
        const double nanoseconds =
            normalizedGpuTime * cpuReferenceTimespan + static_cast<double>(cpuStartTime);
        return nanoseconds / 1e3;
    };
    const double fragmentStartUs = absoluteTimeInUs(timestamps[0]);
    const double fragmentEndUs = absoluteTimeInUs(timestamps[1]);
    std::printf("Fragment time: %.2f ms\n", 0.001 * (fragmentEndUs - fragmentStartUs));
}
} // namespace nlrs
