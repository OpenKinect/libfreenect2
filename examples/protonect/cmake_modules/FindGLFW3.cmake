# - Try to find GLFW3
#
# Will define the following:
# GLFW3_FOUND
# GLFW3_INCLUDE_DIRS
# GLFW3_LIBRARIES

include(FindPackageHandleStandardArgs)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  find_path(GLFW3_INCLUDE_DIRS glfw/glfw3.h DOC "GLFW include directory " HINTS $ENV{GLFW_ROOT}/include)
  
  find_library(GLFW3_LIBRARIES NAMES glfw3dll.lib HINTS $ENV{GLFW_ROOT}/lib/)
ENDIF()

find_package_handle_standard_args(GLFW3 "Could not find GLFW3 - try adding GLFW_ROOT in enviroment variables." GLFW3_INCLUDE_DIRS GLFW3_LIBRARIES)