# - Find the GLEWMX includes and library
#
# This module accepts the following environment variables:
#  GLEWMX_ROOT - Specify the location of libglewmx
#
# This module defines
#  GLEWMX_FOUND - If false, do not try to use libglewmx.
#  GLEWMX_INCLUDE_DIRS - Where to find the headers.
#  GLEWMX_LIBRARIES - The libraries to link against to use libglewmx.

INCLUDE(FindPkgConfig OPTIONAL)

IF(PKG_CONFIG_FOUND)
    PKG_CHECK_MODULES(GLEWMX glewmx)
ELSE(PKG_CONFIG_FOUND)
    FIND_PATH(GLEWMX_INCLUDE_DIRS GL/glew.h
        $ENV{GLEWMX_ROOT}/include
        $ENV{GLEWMX_ROOT}
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/include
        /usr/include
        /sw/include # Fink
        /opt/local/include # DarwinPorts
        /opt/csw/include # Blastwave
        /opt/include
        /usr/freeware/include
    )
    FIND_LIBRARY(GLEWMX_LIBRARIES 
        NAMES GLEWmx libGLEWmx glew32mx glew32mxs
        PATHS
        $ENV{GLEWMX_ROOT}/lib
        $ENV{GLEWMX_ROOT}
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/lib
        /usr/lib
        /sw/lib
        /opt/local/lib
        /opt/csw/lib
        /opt/lib
        /usr/freeware/lib64
    )
    SET(GLEWMX_FOUND "NO")
    IF(GLEWMX_LIBRARIES AND GLEWMX_INCLUDE_DIRS)
        SET(GLEWMX_FOUND "YES")
    ENDIF(GLEWMX_LIBRARIES AND GLEWMX_INCLUDE_DIRS)
ENDIF(PKG_CONFIG_FOUND)