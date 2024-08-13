#include "ui_renderer.hpp"
#include "render_config.hpp"

#include <fmt/core.h>
#include <Foundation/Foundation.hpp>
#define IMGUI_IMPL_METAL_CPP
#include <imgui_impl_metal.h>
#include <simd/simd.h>

#include <exception>
#include <stdexcept>

namespace nlrs
{
UiRenderer::UiRenderer(NS::SharedPtr<MTL::Device> device)
    : mDevice{std::move(device)},
      mCommandQueue{NS::TransferPtr(mDevice->newCommandQueue())},
      mPso{},
      mVertexPositionsBuffer{},
      mRenderPassDescriptor{}
{
    if (!mDevice)
    {
        throw std::runtime_error("Device is nullptr");
    }

    if (!mCommandQueue)
    {
        throw std::runtime_error("Failed to create command queue");
    }

    {
        using NS::StringEncoding::UTF8StringEncoding;

        const char* const shaderSource = R"(
        #include <metal_stdlib>
        
        using namespace metal;

        struct VertexOutput
        {
            float4 position [[position]];
            float2 uv;
        };

        VertexOutput vertex vmain(uint32_t vertexId [[vertex_id]],
                                   device const float2* positions [[buffer(0)]])
        {
            const float2 pos = positions[vertexId].xy;
            const float2 uv = pos * float2(0.5, -0.5) + float2(0.5, 0.5);
            VertexOutput out;
            out.position = float4(pos, 0.f, 1.0);
            out.uv = uv;
            return out;
        }

        float4 fragment fmain(VertexOutput input [[stage_in]],
                            texture2d<float> texture [[texture(0)]])
        {
            const uint2 texCoord = uint2(input.uv * float2(texture.get_width(), texture.get_height()));
            const uint lod = 0;
            const float4 color = texture.read(texCoord, lod);
            return color;
        }
        )";

        NS::Error* error = nullptr;
        auto       library = NS::TransferPtr(mDevice->newLibrary(
            NS::String::string(shaderSource, UTF8StringEncoding), nullptr, &error));
        if (!library)
        {
            throw std::runtime_error(fmt::format(
                "Failed to create ui renderer shader library: {}",
                error->localizedDescription()->utf8String()));
        }

        auto vertexFn =
            NS::TransferPtr(library->newFunction(NS::String::string("vmain", UTF8StringEncoding)));
        auto fragmentFn =
            NS::TransferPtr(library->newFunction(NS::String::string("fmain", UTF8StringEncoding)));
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
        mVertexPositionsBuffer->didModifyRange(
            NS::Range::Make(0, mVertexPositionsBuffer->length()));
    }

    mRenderPassDescriptor = NS::TransferPtr(MTL::RenderPassDescriptor::alloc()->init());
    MTL::RenderPassColorAttachmentDescriptor* const colorAttachmentDesc =
        mRenderPassDescriptor->colorAttachments()->object(0);
    colorAttachmentDesc->setLoadAction(MTL::LoadActionClear);
    colorAttachmentDesc->setClearColor(MTL::ClearColor::Make(0.2f, 0.25f, 0.3f, 1.0));
    colorAttachmentDesc->setStoreAction(MTL::StoreActionStore);
}

void UiRenderer::newFrame() { ImGui_ImplMetal_NewFrame(mRenderPassDescriptor.get()); }

void UiRenderer::draw(const CA::MetalDrawable* const drawable, const MTL::Texture* const source)
{
    MTL::RenderPassColorAttachmentDescriptor* const colorAttachmentDesc =
        mRenderPassDescriptor->colorAttachments()->object(0);
    colorAttachmentDesc->setTexture(drawable->texture());

    MTL::CommandBuffer* const        commandBuffer = mCommandQueue->commandBuffer();
    MTL::RenderCommandEncoder* const encoder =
        commandBuffer->renderCommandEncoder(mRenderPassDescriptor.get());

    encoder->setRenderPipelineState(mPso.get());
    encoder->setVertexBuffer(mVertexPositionsBuffer.get(), 0, 0);
    encoder->setFragmentTexture(source, 0);
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(6));
    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, encoder);
    encoder->endEncoding();

    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}
} // namespace nlrs
