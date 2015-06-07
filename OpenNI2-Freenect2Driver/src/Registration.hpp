#include <libfreenect2/registration.h>

namespace Freenect2Driver {
  class Registration {
  private:
    libfreenect2::Freenect2Device* dev;
    libfreenect2::Registration* reg;
    static const int depthWidth = 512;
    static const int depthHeight = 424;
    static const int colorWidth = 1920;
    static const int colorHeight = 1080;
    static const float invalidDepth;
    static const float infiniteDepth;
    float depth[depthWidth * depthHeight];
    float colorDepth[colorWidth * colorHeight];

  public:
    Registration(libfreenect2::Freenect2Device* dev);
    ~Registration();

    void depthFrame(libfreenect2::Frame* frame);
    void colorFrameRGB888(libfreenect2::Frame* srcFrame, OniFrame* dstFrame);
  };
}
