#include <libfreenect2/registration.h>

namespace Freenect2Driver {
  class Registration {
  private:
    libfreenect2::Freenect2Device* dev;
    libfreenect2::Registration* reg;
    libfreenect2::Frame* lastDepthFrame;
    bool enabled;

  public:
    Registration(libfreenect2::Freenect2Device* dev);
    ~Registration();

    void depthFrame(libfreenect2::Frame* frame);
    void colorFrameRGB888(libfreenect2::Frame* srcFrame, libfreenect2::Frame* dstFrame);
    void setEnable(bool enable = true) { enabled = enable; }
    bool isEnabled() { return enabled; }
  };
}
