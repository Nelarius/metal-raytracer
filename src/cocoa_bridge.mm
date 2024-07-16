#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>
#import <QuartzCore/CAMetalLayer.h>
#import "cocoa_bridge.hpp"

namespace nlrs
{
void addLayerToGlfwWindow(GLFWwindow* const window, CA::MetalLayer* const layer)
{
    NSWindow*     cocoaWindow = glfwGetCocoaWindow(window);
    CAMetalLayer* nativeLayer = (__bridge CAMetalLayer*)layer;
    [[cocoaWindow contentView] setLayer:nativeLayer];
    [nativeLayer setMaximumDrawableCount:2];
    [[cocoaWindow contentView] setWantsLayer:YES];
    [[cocoaWindow contentView] setNeedsLayout:YES];
}
} // namespace nlrs
