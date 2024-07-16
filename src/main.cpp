#include <cstdint>
#include <cstdio>

#include <Metal/Metal.hpp>

int main()
{
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    std::printf("device: %i\n", device);
    device->release();

    return 0;
}
