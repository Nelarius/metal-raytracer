#pragma once

#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalDrawable.hpp>

namespace nlrs
{
struct Camera;
struct GltfModel;

class Renderer
{
public:
    Renderer(NS::SharedPtr<MTL::Device>, const GltfModel&);

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) noexcept = default;
    Renderer& operator=(Renderer&&) noexcept = default;

    void draw(CA::MetalDrawable* drawable, const Camera& camera);

    static constexpr MTL::PixelFormat COLOR_ATTACHMENT_FORMAT =
        MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB;

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
