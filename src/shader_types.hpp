#pragma once

#include <simd/simd.h>
#include <cstdint>

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
} // namespace nlrs
