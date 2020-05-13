/* build with:
   swig2.0 -c++ -DLIBFREENECT2_API -java freenect2.i
   javac ...
*/

%module(directors=1) libfreenect2

/* Includes that will be added to the generated xxx_wrap.cpp
   wrapper file. They will not be interpreted by SWIG */

%{
#include <string>
#include "../include/libfreenect2/libfreenect2.hpp"
%}

%feature("director") Freenect2Device;
%feature("director") FrameListener;
%include "std_string.i"

%include ../include/libfreenect2/libfreenect2.hpp
%include ../include/libfreenect2/frame_listener.hpp
%include ../include/libfreenect2/registration.h

%pragma(java) jniclasscode=%{
  static {
    try {
      System.loadLibrary("freenect2");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native library freenect2 failed to load.\n" + e);
      System.exit(1);
    }
  }
%}
