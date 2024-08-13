#pragma once

#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalDrawable.hpp>

namespace nlrs
{
class UiRenderer
{
public:
    UiRenderer(NS::SharedPtr<MTL::Device> device);

    UiRenderer(const UiRenderer&) = delete;
    UiRenderer& operator=(const UiRenderer&) = delete;
    UiRenderer(UiRenderer&&) noexcept = default;
    UiRenderer& operator=(UiRenderer&&) noexcept = default;

    void draw(const CA::MetalDrawable* target, const MTL::Texture* source);

private:
    NS::SharedPtr<MTL::Device>              mDevice;
    NS::SharedPtr<MTL::CommandQueue>        mCommandQueue;
    NS::SharedPtr<MTL::RenderPipelineState> mPso;
    NS::SharedPtr<MTL::Buffer>              mVertexPositionsBuffer;
};
} // namespace nlrs
