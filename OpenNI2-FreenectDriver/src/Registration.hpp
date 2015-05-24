#include <libfreenect2/registration.h>

namespace Freenect2Driver {
  class Registration {
  private:
    libfreenect2::Freenect2Device* dev;
    libfreenect2::Registration* reg;
    static const int depthWidth = 512;
    static const int depthHeight = 424;
    static const float invalidDepth = 0.0;
    float depth[depthWidth * depthHeight];

  public:
    Registration(libfreenect2::Freenect2Device* dev);
    ~Registration();

    void depthFrame(OniFrame* frame);
    void colorFrameRGB888(uint8_t* data, OniFrame* frame);
  };
}
