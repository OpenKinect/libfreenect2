# FindTurboJPEG.cmake
# Uses environment variable TurboJPEG_ROOT as backup
# - TurboJPEG_FOUND
# - TurboJPEG_INCLUDE_DIRS
# - TurboJPEG_LIBRARIES
# - TurboJPEG_LIBRARIES_STATIC

FIND_PATH(TurboJPEG_INCLUDE_DIRS
  turbojpeg.h
  DOC "Found TurboJPEG include directory"
  PATHS
    "${DEPENDS_DIR}/libjpeg_turbo"
    "${DEPENDS_DIR}/libjpeg-turbo64"
    "/usr/local/opt/jpeg-turbo" # homebrew
    "/opt/local" # macports
    "C:/libjpeg-turbo64"
    "/opt/libjpeg-turbo"
    ENV TurboJPEG_ROOT
  PATH_SUFFIXES
    include
)

FIND_LIBRARY(TurboJPEG_LIBRARIES
  NAMES libturbojpeg.so.0 turbojpeg.so
  DOC "Found TurboJPEG library path"
  PATHS
    "${DEPENDS_DIR}/libjpeg_turbo"
    "${DEPENDS_DIR}/libjpeg-turbo64"
    "/usr/local/opt/jpeg-turbo" # homebrew
    "/opt/local" # macports
    "C:/libjpeg-turbo64"
    "/opt/libjpeg-turbo"
    ENV TurboJPEG_ROOT
  PATH_SUFFIXES
    lib
    lib64
)

FIND_LIBRARY(TurboJPEG_LIBRARIES_STATIC
  NAMES libturbojpeg.a.0 turbojpeg.a
  DOC "Found TurboJPEG library path"
  PATHS
    "${DEPENDS_DIR}/libjpeg_turbo"
    "${DEPENDS_DIR}/libjpeg-turbo64"
    "/usr/local/opt/jpeg-turbo" # homebrew
    "/opt/local" # macports
    "C:/libjpeg-turbo64"
    "/opt/libjpeg-turbo"
    ENV TurboJPEG_ROOT
  PATH_SUFFIXES
    lib
    lib64
)

INCLUDE(CheckCSourceCompiles)
if(MSVC)
  set(CMAKE_REQUIRED_DEFINITIONS -MT)
endif()
set(CMAKE_REQUIRED_INCLUDES ${TurboJPEG_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${TurboJPEG_LIBRARIES})
check_c_source_compiles("#include <turbojpeg.h>\nint main(void) { tjhandle h=tjInitCompress(); return 0; }" TURBOJPEG_WORKS)
set(CMAKE_REQUIRED_DEFINITIONS)
set(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_REQUIRED_LIBRARIES)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TurboJPEG DEFAULT_MSG TurboJPEG_LIBRARIES TurboJPEG_INCLUDE_DIRS TURBOJPEG_WORKS)
