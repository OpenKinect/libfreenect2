##############################################################################
# OpenNI2-Freenect2Driver
##############################################################################

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

include_directories(extern/OpenNI-Linux-x64-2.2.0.33/Include)
include_directories(${PROJECT_SOURCE_DIR}/../include)
include_directories(${PROJECT_SOURCE_DIR}/../examples/protonect/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/wrappers/cpp)

target_link_libraries(Freenect2Driver freenect2 ${MATH_LIB})

install (TARGETS Freenect2Driver
  DESTINATION "${PROJECT_LIBRARY_INSTALL_DIR}/OpenNI2-Freenect2Driver")
