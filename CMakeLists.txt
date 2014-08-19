CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)

SET(PROJECTNAME "freenect2")
PROJECT(${PROJECTNAME} C CXX)

#Suppressing CMAKE 3.0 warnings
if(POLICY CMP0043)
cmake_policy(SET CMP0043 OLD)
endif()

SET(CMAKE_BUILD_TYPE RelWithDebInfo)

# additional cmake modules
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)

# setup threading
INCLUDE(SetupLibfreenect2Threading)
INCLUDE_DIRECTORIES(${LIBFREENECT2_THREADING_INCLUDE_DIR})
# setup GenerateResources
INCLUDE(GenerateResources)

#-----------------------------
# Define default destinations
#-----------------------------

SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

#-----------------------------
# Enable Relative path
#-----------------------------

#don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# Enable Shared libs
SET(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

# dependencies
# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

# LibUSB
FIND_PACKAGE(LibUSB REQUIRED)
INCLUDE_DIRECTORIES(${LibUSB_INCLUDE_DIRS})

# GLEW
FIND_PACKAGE(GLEW REQUIRED)
INCLUDE_DIRECTORIES(${GLEW_INCLUDE_PATH})
FIND_PACKAGE(GLEWMX REQUIRED)
INCLUDE_DIRECTORIES(${GLEWMX_INCLUDE_DIRS})
ADD_DEFINITIONS(-DGLEW_MX -DGLEW_STATIC)

# GLFW
FIND_PACKAGE(GLFW REQUIRED)
SET(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs")
SET(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs")
SET(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation")
INCLUDE_DIRECTORIES(${GLFW_INCLUDE_DIR})

# TurboJPEG
if(UNIX)
FIND_PACKAGE(TurboJPEG REQUIRED)
INCLUDE_DIRECTORIES(${TJPEG_INCLUDE_DIR})
endif()

#-----------------------------
# Adding sources
#-----------------------------

SET(RESOURCES_INC_FILE "${CMAKE_SOURCE_DIR}/src/resources.inc")
GENERATE_RESOURCES(${RESOURCES_INC_FILE} ${CMAKE_SOURCE_DIR} 
  resources/11to16.bin 
  resources/xTable.bin 
  resources/zTable.bin 
  src/shader/debug.fs
  src/shader/default.vs
  src/shader/filter1.fs
  src/shader/filter2.fs
  src/shader/stage1.fs
  src/shader/stage2.fs
)
ADD_DEFINITIONS(-DRESOURCES_INC)

file(GLOB_RECURSE PUBLIC_HEADERS include/*.h)
file(GLOB PROJECT_HEADERS include/*.h include/*.hpp)
file(GLOB_RECURSE PROJECT_SRC src/*.cpp)

INCLUDE_DIRECTORIES(include)

#-----------------------------
# Generating Target
#-----------------------------

ADD_LIBRARY(${PROJECTNAME} SHARED
  ${PUBLIC_HEADERS}
  ${PROJECT_HEADERS}
  ${PROJECT_SRC}
  ${LIBFREENECT2_THREADING_SOURCE}
  ${RESOURCES_INC_FILE}
)

#-----------------------------
# Define exports (API)
#-----------------------------

set_target_properties(${PROJECTNAME} PROPERTIES
  PUBLIC_HEADER "${PUBLIC_HEADERS}")

set(${PROJECTNAME}_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
    CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

#-----------------------------
# Linking libraries
#-----------------------------

TARGET_LINK_LIBRARIES(${PROJECTNAME}
  ${LibUSB_LIBRARIES}
  ${OpenCV_LIBS}
  ${TJPEG_LIBRARY}
  ${LIBFREENECT2_THREADING_LIBRARIES}
  ${GLEW_LIBRARY}
  ${GLEWMX_LIBRARIES}
  ${GLFW_LIBRARIES}
)

#-----------------------------
# Install Phase
#-----------------------------

INSTALL(TARGETS ${PROJECTNAME}
   EXPORT ${PROJECTNAME}Targets
   LIBRARY DESTINATION lib
   RUNTIME DESTINATION bin
   PUBLIC_HEADER DESTINATION include
)


ADD_EXECUTABLE(test_opengl
  test/test_opengl_depth_packet_processor.cpp
)

TARGET_LINK_LIBRARIES(test_opengl
  ${PROJECTNAME}
)
  
ADD_EXECUTABLE(Protonect
  test/Protonect.cpp
)
  
TARGET_LINK_LIBRARIES(Protonect
  ${PROJECTNAME}
)

INSTALL(TARGETS Protonect
  DESTINATION bin
)
