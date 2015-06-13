##############################################################################
# OpenNI2-Freenect2Driver
##############################################################################

set(LIBFREENECT2_DIR ../..)
set(PROTONECT_DIR ${LIBFREENECT2_DIR}/examples/protonect)
set(MY_DIR ${LIBFREENECT2_DIR}/drivers/OpenNI2-Freenect2Driver)
LIST(APPEND CMAKE_MODULE_PATH ${PROTONECT_DIR}/cmake_modules)
# setup threading
INCLUDE(SetupLibfreenect2Threading)

file(GLOB HEADERS src/*.hpp src/*.h)
file(GLOB SOURCES src/*.cpp)
add_library(Freenect2Driver SHARED ${HEADERS} ${SOURCES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/OpenNI2-Freenect2Driver)
set_target_properties(Freenect2Driver PROPERTIES
  VERSION ${PROJECT_VER}
  SOVERSION ${PROJECT_APIVER}
  OUTPUT_NAME Freenect2Driver)
  
add_definitions(-DPROJECT_VER="${PROJECT_VER}")

include_directories(${LIBFREENECT2_DIR}/include)
include_directories(${PROTONECT_DIR}/include)
include_directories(${MY_DIR}/extern/OpenNI-Linux-x64-2.2.0.33/Include)
include_directories(${MY_DIR}/src)
include_directories(${PROTONECT_DIR}/${LIBFREENECT2_THREADING_INCLUDE_DIR})

FIND_LIBRARY(FREENECT2_LIB freenect2 ${PROTONECT_DIR}/lib)
target_link_libraries(Freenect2Driver ${FREENECT2_LIB} ${MATH_LIB})

install (TARGETS Freenect2Driver
  DESTINATION "${PROJECT_LIBRARY_INSTALL_DIR}/OpenNI2-Freenect2Driver")
