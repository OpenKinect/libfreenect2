INCLUDE(CheckCXXSourceCompiles)

CHECK_CXX_SOURCE_COMPILES("
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

int main(int argc, char** argv) {
  std::thread thread;
  std::mutex mutex;
  std::lock_guard<std::mutex> lock_guard(mutex);
  std::unique_lock<std::mutex> unique_lock(mutex);
  std::condition_variable condition_variable;
  thread_local int i;
  
  return 0;
}

" LIBFREENECT2_THREADING_STDLIB)

IF(LIBFREENECT2_THREADING_STDLIB)
  SET(LIBFREENECT2_THREADING "stdlib")
  SET(LIBFREENECT2_THREADING_INCLUDE_DIR "")
  SET(LIBFREENECT2_THREADING_SOURCE "")
  SET(LIBFREENECT2_THREADING_LIBRARIES "")
  ADD_DEFINITIONS(-DLIBFREENECT2_THREADING_STDLIB)
ELSE(LIBFREENECT2_THREADING_STDLIB)
  SET(LIBFREENECT2_THREADING "tinythread")
  SET(LIBFREENECT2_THREADING_INCLUDE_DIR "src/tinythread/")
  SET(LIBFREENECT2_THREADING_SOURCE "src/tinythread/tinythread.cpp")
  IF(NOT WIN32)
    SET(LIBFREENECT2_THREADING_LIBRARIES "pthread")
  ELSE(NOT WIN32)
    SET(LIBFREENECT2_THREADING_LIBRARIES "")
  ENDIF(NOT WIN32)
  ADD_DEFINITIONS(-DLIBFREENECT2_THREADING_TINYTHREAD)
ENDIF(LIBFREENECT2_THREADING_STDLIB)

MESSAGE(STATUS "using ${LIBFREENECT2_THREADING} as threading library")
