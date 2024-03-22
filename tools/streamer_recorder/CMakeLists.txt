CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.1)

if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
  endif()
endif()

IF(NOT DEFINED CMAKE_BUILD_TYPE)
  # No effect for multi-configuration generators (e.g. for Visual Studio)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose: RelWithDebInfo Release Debug MinSizeRel None")
ENDIF()

PROJECT(libfreenect2_tools_SR)

SET(MY_DIR ${libfreenect2_tools_SR_SOURCE_DIR})
SET(DEPENDS_DIR "${MY_DIR}/../../../depends" CACHE STRING "Dependency directory")

OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)

# The build system could be standalone if these files are copied instead of being referenced here.
SET(freenect2_ROOT_DIR ${MY_DIR}/../..)
SET(flextGL_SOURCES ${freenect2_ROOT_DIR}/src/flextGL.cpp)
SET(flextGL_INCLUDE_DIRS ${freenect2_ROOT_DIR}/src) # for flextGL.h
SET(example_INCLUDE_DIRS ${freenect2_ROOT_DIR}/examples) # for protonect viewer

FIND_PACKAGE(PkgConfig)    # try find PKGConfig as it will be used if found
LIST(APPEND CMAKE_MODULE_PATH ${freenect2_ROOT_DIR}/cmake_modules) # FindGLFW3.cmake

IF(TARGET freenect2)
  MESSAGE(STATUS "Using in-tree freenect2 target")
  SET(freenect2_LIBRARIES freenect2)
  SET(freenect2_DLLS ${LIBFREENECT2_DLLS})
ELSE()
  FIND_PACKAGE(freenect2 REQUIRED)
  # Out-of-tree build will have to have DLLs manually copied.
ENDIF()

INCLUDE_DIRECTORIES(
  ${freenect2_INCLUDE_DIR}
  ${example_INCLUDE_DIRS}
  ${MY_DIR}/include
)

SET(ProtonectSR_src
  ProtonectSR.cpp
  PracticalSocket.cpp
  streamer.cpp
  recorder.cpp
)

SET(ProtonectSR_LIBRARIES
  ${freenect2_LIBRARIES}
)

SET(ProtonectSR_DLLS
  ${freenect2_DLLS}
)

# Add OpenCV package dependency for udp-image-streaming
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LIST(APPEND ProtonectSR_LIBRARIES
  ${OpenCV_LIBS}
)
# OpenCV requires exception.
IF(UNIX AND NOT APPLE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
ENDIF()

IF(ENABLE_OPENGL)
  FIND_PACKAGE(GLFW3)
  FIND_PACKAGE(OpenGL)
  IF(GLFW3_FOUND AND OPENGL_FOUND)
    INCLUDE_DIRECTORIES(
      ${GLFW3_INCLUDE_DIRS}
      ${flextGL_INCLUDE_DIRS}
    )

    LIST(APPEND ProtonectSR_DLLS ${GLFW3_DLL})
    LIST(APPEND ProtonectSR_src
      ${example_INCLUDE_DIRS}/viewer.cpp
      ${flextGL_SOURCES}
    )
    LIST(APPEND ProtonectSR_LIBRARIES
      ${GLFW3_LIBRARIES}
      ${OPENGL_gl_LIBRARY}
    )
    ADD_DEFINITIONS(-DEXAMPLES_WITH_OPENGL_SUPPORT=1)
  ENDIF()
ENDIF(ENABLE_OPENGL)

ADD_EXECUTABLE(ProtonectSR
  ${ProtonectSR_src}
)

TARGET_LINK_LIBRARIES(ProtonectSR
  ${ProtonectSR_LIBRARIES}
)

if(EXISTS ProtonectSR)
  configure_file(build/ProtonectSR freenect2-record COPYONLY)
  configure_file(build/ProtonectSR freenect2-replay COPYONLY)
  configure_file(build/ProtonectSR freenect2-stream COPYONLY)
endif()

file(MAKE_DIRECTORY build/recordings)
file(MAKE_DIRECTORY build/recordings/depth)
file(MAKE_DIRECTORY build/recordings/regist)

IF(WIN32)
  INSTALL(TARGETS ProtonectSR DESTINATION bin)
  LIST(REMOVE_DUPLICATES ProtonectSR_DLLS)
  FOREACH(FILEI ${ProtonectSR_DLLS})
    ADD_CUSTOM_COMMAND(TARGET ProtonectSR POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILEI} $<TARGET_FILE_DIR:ProtonectSR>
    )
  ENDFOREACH(FILEI)
  INSTALL(FILES ${ProtonectSR_DLLS} DESTINATION bin)
ENDIF()
