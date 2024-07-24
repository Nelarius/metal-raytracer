#pragma once

#include "camera.hpp"
#include "extent.hpp"
#include "units/angle.hpp"

#include <glm/glm.hpp>

#include <optional>

struct GLFWwindow;

namespace nlrs
{
struct FlyCameraController
{
public:
    struct Orientation
    {
        glm::vec3 forward;
        glm::vec3 right;
        glm::vec3 up;
    };

    FlyCameraController() = default;

    Camera    getCamera() const;
    glm::mat4 viewProjectionMatrix() const;

    void lookAt(const glm::vec3& p);
    void update(GLFWwindow* window, float dt);

    // Accessors

    inline float&           speed() { return mSpeed; }
    inline Angle&           vfov() { return mVfov; }
    inline float&           aperture() { return mAperture; }
    inline float&           focusDistance() { return mFocusDistance; }
    inline const glm::vec3& position() const { return mPosition; }
    inline Angle            yaw() const { return mYaw; }
    inline Angle            pitch() const { return mPitch; }
    inline Orientation      orientation() const { return cameraOrientation(); }

private:
    // Camera orientation and physical characteristics

    glm::vec3 mPosition = glm::vec3(0.f, 0.f, -1.f);
    Angle     mYaw = Angle::degrees(0.f);
    Angle     mPitch = Angle::degrees(0.f);
    Angle     mVfov = Angle::degrees(80.0f);
    float     mAperture = 0.f;
    float     mFocusDistance = 10.0f;

    // Input state

    float mSpeed = 1.0f;
    bool  mLeftPressed = false;
    bool  mRightPressed = false;
    bool  mForwardPressed = false;
    bool  mBackwardPressed = false;
    bool  mUpPressed = false;
    bool  mDownPressed = false;
    bool  mMouseLookPressed = false;

    // Window state

    struct MousePos
    {
        // Mouse position coordinates are given in screen coordinates not pixel coordinates.
        double x = 0.0;
        double y = 0.0;
    };

    // TODO: discriminate between window size and framebuffer size to avoid accidents with mouse
    // cursor uv-coordinates
    Extent2i                mWindowSize = Extent2i(0, 0);
    std::optional<MousePos> mLastMousePos = std::nullopt;

    Orientation cameraOrientation() const;
    glm::vec3   generateCameraRayDir(const Orientation&, const MousePos&) const;
};
} // namespace nlrs
