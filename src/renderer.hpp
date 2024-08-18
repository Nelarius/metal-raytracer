#pragma once

#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalDrawable.hpp>

#include <vector>

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

    void draw(const Camera& camera, std::uint32_t width, std::uint32_t height);

    inline const MTL::Texture*    texture() const { return mTexture.get(); }
    inline std::span<const float> rayTracingTimings() const { return mFragmentMillis; }

private:
    NS::SharedPtr<MTL::Device>                mDevice;
    NS::SharedPtr<MTL::CommandQueue>          mCommandQueue;
    NS::SharedPtr<MTL::Texture>               mTexture;
    NS::SharedPtr<MTL::Heap>                  mHeap;
    NS::SharedPtr<MTL::RenderPipelineState>   mPso;
    NS::SharedPtr<MTL::CounterSampleBuffer>   mTimerSampleBuffer;
    NS::SharedPtr<MTL::Buffer>                mVertexPositionsBuffer;
    NS::SharedPtr<MTL::Buffer>                mUniformsBuffer;
    NS::SharedPtr<MTL::Buffer>                mTextureBuffer;
    NS::SharedPtr<MTL::Buffer>                mPrimitiveBuffer;
    NS::SharedPtr<MTL::Buffer>                mPrimitiveBufferOffsets;
    NS::SharedPtr<MTL::AccelerationStructure> mAccelerationStructure;
    std::vector<float>                        mFragmentMillis;
};
} // namespace nlrs
