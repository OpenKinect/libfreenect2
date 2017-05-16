INCLUDE(CheckCXXSourceCompiles)
INCLUDE(CheckCSourceCompiles)

SET(CMAKE_REQUIRED_INCLUDES "${MY_DIR}/include/internal" ${OpenCL_INCLUDE_DIRS})
SET(CMAKE_REQUIRED_LIBRARIES ${OpenCL_LIBRARIES})
CHECK_C_SOURCE_COMPILES("
#include <CL/cl.h>
int main() {
  clGetPlatformIDs(0, 0, 0);
  return 0;
}" OpenCL_C_WORKS)
CHECK_CXX_SOURCE_COMPILES("
#include <CL/cl.hpp>
int main() {
  cl::Context context;
  cl::Platform platform;
  cl::Device device;
  return 0;
}" OpenCL_CXX_WORKS)
SET(CMAKE_REQUIRED_INCLUDES)
SET(CMAKE_REQUIRED_LIBRARIES)
