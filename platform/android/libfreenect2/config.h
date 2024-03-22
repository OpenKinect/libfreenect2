#ifndef LIBFREENECT2_CONFIG_H
#define LIBFREENECT2_CONFIG_H

#define LIBFREENECT2_VERSION "0.2.0"
#define LIBFREENECT2_API_VERSION ((0 << 16) | 2)

#define LIBFREENECT2_PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))

#define LIBFREENECT2_API __attribute__((visibility("default")))

#define LIBFREENECT2_WITH_TURBOJPEG_SUPPORT
#define LIBFREENECT2_THREADING_STDLIB
#define LIBFREENECT2_WITH_CXX11_SUPPORT

#endif // LIBFREENECT2_CONFIG_H
