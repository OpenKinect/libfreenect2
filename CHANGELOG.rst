^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package libfreenect2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Works on my machine...
* Update FindTegraJPEG.cmake (`#999 <https://github.com/LCAS/libfreenect2/issues/999>`_)
* Fix headline/anchor for MacOS instructions (`#994 <https://github.com/LCAS/libfreenect2/issues/994>`_)
* cmake: Fix OSX RPATH handling
* docs: Update TurboJPEG instructions on Ubuntu
  turbojpeg.h, libturbojpeg.{a,so} are moved from libjpeg-turbo8-dev in xenial to libturbojpeg0-dev in artful.
  Fix `#984 <https://github.com/LCAS/libfreenect2/issues/984>`_.
* Update DOI URL
* docs: Update OpenNI2 tap for Mac OS
  Fix `#968 <https://github.com/LCAS/libfreenect2/issues/968>`_.
* usb: Select UsbDk with libusb_set_option
* cmake: Remove UsbDk build configs
* remove command for builsing libusbdk as it's now in libusb master
* windows: Build upstream libusb only
  libusbK patches have been merged upstream.
* docs: Be more precise about cd ..
* cmake: Make streamer-recorder build in-tree.
  - [streamer][recorder][cmake] adjust for refactorings and comments
  - [streamer][recorder][cmake] the main executable does not exist sometimes
* docs: Primarily add proper copyright headers and minor editorializing.
* docs: Document ProtonectSR more, its options, and basic build steps.
* ProtonectSR: Add replay and exe option handling.
  add replay code invocation and option processing, but for now
  commenting out for merge. Make OS-indepdenent opendir(); to test.
* ProtonectSR: Sync ProtonectSR.cpp with Protonect.cpp.
* ProtonectSR: Make initial implementation of streamer-recorder.
  @PyrApple:
  - first implementation of the real-time streamer:
  send kinect image via UDP socket to third party
  (e.g. to use in python).
  ## ONLY DEPTH FRAME SENT FOR NOW ##
  added example Blender (BGE) scene applying
  received image to object texture.
  - first implementation of the frame recorder:
  record kinect captured images to disk
  + time stamp for video creation.
  - ProtonectSR standalone toolbox creation: moved all code related
  to streamer and recorder to ./tools directory, reset of all
  changes made to other parts of the original libfreenect code.
  - ProtonectSR standalone toolbox creation:
  finished reset of original libfreenect code.
  @smokhov:
  - Remove .keep files, per @xlz.
  - [streamer] move .cpp our of include
  - [streamer][recorder] primarily copy-edited headers to match better
  project conventions; move data members to private sections
  - [docs] arrange include better with a comment
  - [sockets] add necessary include
  - [recorder] format better with project's coding conventions
* Remove usage of std::bind
  It is deprecated after C++11.
* threading: Add the missing header for std::bind
  Fixes `#945 <https://github.com/LCAS/libfreenect2/issues/945>`_.
* docs: Add notes about missing helper_math.h
* Clean up Freenect2Replay coding style
* Add Freenect2Replay API and depth implementation
* docs: Clarify SyncMultiFrameListener API usage
  Fix `#851 <https://github.com/LCAS/libfreenect2/issues/851>`_.
* fix memory leak in DumpDepthPacketProcessor
* docs: Update Ubuntu 14.04 OpenGL instructions
  `apt-get install libgl1-mesa-dri-lts-vivid` creates more confusion than its benefit. Remove it.
* Added TegraJpeg sources path for L4T 28.1
* add osx library path for brew install
* Changed MIN GL Version from 3 to 2
  changed glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); to glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2); in order to get compatibility with older opengl versions
* Fixes memory leak
  When listener doesn't take ownership it will leak the frame
* fix the cmake conditional expression error
* docs: Fix a typo
* docs: Update timestamp unit
  Thanks to @caselitz's testing
  Close `#849 <https://github.com/LCAS/libfreenect2/issues/849>`_.
* opencl: Suppress warning on ocl-icd 2.2.10+
  A hack to compare versions. Since we're only dealing with
  versions of 2.y.z and 1.x and only providing informative
  warnings this hack should do the work for now without
  introducing complicated version comparison routines.
* Fix version check when trying to find freenect2 with find_package
* usb: Retry a few times with libusb_open()
  On some platforms (UsbDk) libusb_open() will return errors for a
  short while immediately after libusb_close(). Retry a few times.
  Also, fix libusb device reference handling across open and close.
  Fix `#812 <https://github.com/LCAS/libfreenect2/issues/812>`_.
* cmake: Clean up L4T URLs
* Update README.md
  Take . off command coz it breaks it
* cmake: Update CUDA C++11 build flags
  nvcc can't use only -Xcompiler -std=c++11, it needs -std=c++11 in the whole
  pipeline.
  The real fix is in CMake 3.3.
  https://github.com/Kitware/CMake/commit/99abebdea01b9ef73e091db5594553f7b1694a1b
  Fix `#818 <https://github.com/LCAS/libfreenect2/issues/818>`_.
* docs: Update README.md to CommonMark
* Merge pull request `#815 <https://github.com/LCAS/libfreenect2/issues/815>`_ from eric-schleicher/patch-1
  typo preventing correct display of markdown
* type preventing correct display of markdown
  The ### linux section trapped inside \`\`\` block
* cmake: Work around CUDA 8 overriding OpenCL path
  Fix `#804 <https://github.com/LCAS/libfreenect2/issues/804>`_.
* cudakde: Fix restrict keyword order
* Merge pull request `#765 <https://github.com/LCAS/libfreenect2/issues/765>`_ from xlz/disable-default-constructors
  Disable default copy and assignment constructors
* opencl: Disable useless -Wignored-attributes on gcc6
* Disable default copy and assignment constructors
  To prevent locks from being copied.
  Close `#677 <https://github.com/LCAS/libfreenect2/issues/677>`_.
* openni2: Add serial parameter to uri
  The URI format is now this:
  freenect2://0?serial=0123456789&depth-size=123x456
  Opening OpenNI2 by serial number is not implemented now
  so the serial parameter in the URI is only informative for
  distinguishing between devices. depth-size is still optional.
  Close `#762 <https://github.com/LCAS/libfreenect2/issues/762>`_
* openni2: Add VideoStream::convertDepthToColorCoordinates
  Close `#760 <https://github.com/LCAS/libfreenect2/issues/760>`_
* cmake: Update Tegra gstjpeg download paths
  Also split find_library() for libjpeg.so and libnvjpeg.so.
* Add KDE depth unwrapping algorithms
  This implements kernel density estimation based phase unwrapping
  procedure. It shows improved depth imaging, especially for large depth and
  outdoors scenes. The method was presented on ECCV 2016, see paper for more
  information.
  http://users.isy.liu.se/cvl/perfo/abstracts/jaremo16.html
  The algorithms are added as OpenCL and CUDA processors. OpenCLKde and CudaKde
  pipelines are also added as APIs.
* cmake: Update Windows CUDA 8 sample path
* Merge pull request `#655 <https://github.com/LCAS/libfreenect2/issues/655>`_ from imatge-upc/pkg_config_path
  Avoid overriding the PKG_CONFIG_PATH environment variable
* opencl: Use 1.0f float to avoid llvm errors
  Beignet recommends:
  If you use 1.0 in the kernel, LLVM 3.6 will treat it as 1.0f, a
  single float, because the project doesn't support double float.
  but LLVM 3.8 will treat it as 1.0, a double float, at the last
  it may cause error.  So we recommend using 1.0f instead of 1.0
  if you don't need double float.
* Merge pull request `#745 <https://github.com/LCAS/libfreenect2/issues/745>`_ from Delicode/fix_openni2_enumeration
  Fix OpenNI2 enumeration softlocking sensors
* Fix OpenNI2 enumeration softlocking sensors
* Merge pull request `#739 <https://github.com/LCAS/libfreenect2/issues/739>`_ from xiekuncn/master
  Added TegraJPEG supporting for TK1 L4T r21.5.
* Added TegraJPEG supporting for TK1 L4T r21.5.
  add downloading tegra jpeg at L4T r21.5.
  you also can download the file from http://developer.download.nvidia.com/embedded/L4T/r21_Release_v5.0/source/gstjpeg_src.tbz2 to folder ${srouce_root}/depends/gstjpeg/
* Merge pull request `#734 <https://github.com/LCAS/libfreenect2/issues/734>`_ from RealRecon/fix_cmake
  Fixed typo in CUDA related part in the CMake file
* Fixed typo in CUDA related part in the CMake file
* - avoid overriding the PKG_CONFIG_PATH environment variable
* Update author list for 0.2 release
* docs: Update API descriptions
* docs: Document environment variables
* windows: Update release files
* logging: Lower rgb stream message level
* Add envvar LIBFREENECT2_PIPELINE to select pipeline
* depends: Update i965 driver path
  Fix `#631 <https://github.com/LCAS/libfreenect2/issues/631>`_
* depends: Do not download libva debs for non-x86
* depends: Fix libva debs version
* docs: Update UsbDk instructions
* cmake: Check USB device driver
  Check UsbDk device driver. If not found, fall back to libusbK.
  If libusbK device driver is not found, bail.
  Fix `#621 <https://github.com/LCAS/libfreenect2/issues/621>`_
* api: Specify Freenect2Device::Config::Config()
  This function was not exported from Freenect2Device for MSVC.
* openni2: Fix msvc warning
* Merge pull request `#614 <https://github.com/LCAS/libfreenect2/issues/614>`_ from hanyazou/status_1024
  Add 5 seconds limit to the status 0x90000 checking loop
* usb: Add 5 seconds limit to the status 0x90000 checking loop
* Merge pull request `#612 <https://github.com/LCAS/libfreenect2/issues/612>`_ from hanyazou/wait_new_frames_timeout
  Protonect: Add timeout arg for waitForNewFrame()
* Protonect: Add timeout arg for waitForNewFrame()
* tegra: Fix typo
* docs: How to switch to libusbk backend
* usb: Use less transfers for multi-Kinect setup
  Windows can only poll() 64 fds at once.
* docs: Update Beignet ppa
* Add error propagation for processors
  The new internal API policy:
  Packet processors should report internal errors by setting
  good() to false, and pass the last frame to the user with
  status set to 1.
  Currently CUDA, OpenCL, Tegra, and VAAPI have been added
  with the error propagation. CPU, OpenGL, and VT have no
  error checking in place so they do not report errors.
  TurboJPEG seems to produce non-fatal errors so it also
  does not propagate errors.
  The user should check the received frame's status
  for errors. If there are errors, the user should stop the
  device and exit.
  When good() is false, the processor->process()
  will no longer be called, and if the user continues to
  call waitForNewFrame(), it will hang.
* frame: Update format definitions
* usb: Use envvars to control transfer pool size
  LIBFREENECT2_RGB_TRANSFER_SIZE (default 0x4000)
  LIBFREENECT2_RGB_TRANSFERS (default 20)
  LIBFREENECT2_IR_PACKETS (default 8)
  LIBFREENECT2_IR_TRANSFERS (default 60)
* cmake: Add Linux4Tegra 23.2 link
* usb: Issue reboot command on Mac OS X
  Without the ShutdownCommand, the Kinect still disappears randomly
  on Mac OS X. Painstaking effort did not determine the cause.
  So take the suboptimal way and shut it down explicitly.
  Fixed `#539 <https://github.com/LCAS/libfreenect2/issues/539>`_.
* docs: VAAPI is supported by Ivy Bridge and newer
* build: Fix Tegra tarball URL
* cuda: Fix wrong write combined flag
  The buffer sent to CUDA needs write combined flag.  The buffer send
  to the user does not need this.
  This flag made Registration::apply() very slow in its memory read.
* docs: Add instructions on building with UsbDk
* build: Use usbdk for libusb on Windows
* Merge pull request `#592 <https://github.com/LCAS/libfreenect2/issues/592>`_ from fran6co/vt_10.8
  Mac OS X 10.8 compatibility
* 10.8 compatibility, if the system supports hardware acceleration it's should be enabled by default
* docs: How to let CMake find libfreenect2
* Protonect: Add '-frames' option
* cuda: Use memory pooling for frames
* build: Update libusb build script for VS2013
* Create ISSUE_TEMPLATE.md
* opencl: Use a different profiling macro
  Enabling profiling in OpenCL effects the performance, so for
  profiling libfreenect2s processors, it should be disabled and only
  used when testing improvements of the OpenCL code itself.
* opencl: Add recommended changes
  Usage of LIBFREENECT2_WITH_PROFILING.
  Changed CHECK_CL macros.
  OpenCLAllocator can now be used for input and output buffers.
  OpenCLFrame now uses OpenCLBuffer from allocator.
  IMAGE_SIZE and LUT_SIZE as static const.
  Added Allocators for input and output buffers.
  Moved allocate_opencl to top.
  Added good() method.
* opencl: Use more concise error checking macro
  Changed filling methods to return a bool on success, making macro
  LOG_CL_ERROR obsolete.
* opencl: Add optional profiling
  Added (optional) profiling of OpenCL kernels.
  Reverted back to calculating sine and cosine on the GPU.
* opencl: Use pinned memory buffers and frames
* opencl: allocate OpenCL buffers on initialization
  Removed arrays for tables and allocated OpenCL buffers on
  initialization.
  loadXZTables, loadLookupTable and loadP0TablesFromCommandResponse
  will now directly write to the OpenCL buffers.
* opencl: Use precomputed sin/cos tables
  Instead of computing the sine and cosine for the p0 table and the
  phases on the GPU, they are now precomputed once on the CPU.
  Details: Replaced sin(a+b) by sin(a)*cos(b)+cos(a)*sin(b), where
  sin(a),cos(b),cos(a),sin(b) are stored in a LUT.  Simplyfied
  processPixelStage1 code and removed processMeasurementTriple.
  Moved one if from decodePixelMeasurement to processPixelStage1.
  Removed the first part of `valid && any(...)` because valid has been
  checked before.
* Merge pull request `#583 <https://github.com/LCAS/libfreenect2/issues/583>`_ from fran6co/vt_10.9
  Using 10.9 available API for VideoToolbox
* Using 10.9 available API for VideoToolbox
* logging: Add an option to collect profiling
  Use cmake -DENABLE_PROFILING=ON (OFF by default).
* threading: Set thread names for perf
* Protonect: Add argument to select GPU
* cmake: Build CUDA 6.5 object without C++11
  CUDA 7.0 is the first version that supports C++11.
  Though linking C++11 objects with non-C++11 ones is problematic.
* logging: Remove std::string from internal API
  The internal logging API is used by the CUDA processor.
  For CUDA 6.5 and -DENABLE_CXX11=ON, the cuda object is compiled
  with C++98 and other objects with C++11. Thus remove std::string
  for being incompatible ABI across C++98 and C++11.
* docs: Add instructions about Jetson and others
* tegra: Add build support
* tegra: Add Tegra JPEG decoder
* Merge pull request `#575 <https://github.com/LCAS/libfreenect2/issues/575>`_ from fran6co/patch-1
  Error when using C++11 std threading
* Error when using C++11 std threading
* allocator: Use unique_lock for condvar
  Fix a FTBFS with C++11.
* vaapi: Fix a missed vaUnmap
* docs: Add CUDA instructions
* cmake: Fix path separator being escaped on Windows
* cuda: Use zerocopy pinned memory
* cuda: Optimize math
* cuda: Add build support
* cuda: Add CUDA depth processor
* docs: Update Windows OpenCL download
  Intel OpenCL SDK 2016 is available for download
* build: Update libusb build script
  Josh Blake's winiso is now broken by merge conflicts.
  Provide a new libusb winiso branch to solve the conflicts.
* usb: Do not reboot
  Freenect2Device::close() issues ShutdownCommand which reboots
  the device and makes it disappear for 3 seconds.
  Do not do that.
* Fix a memory leak
* vaapi: Use zerocopy memory pool for frames
* sync listener: discard new frames if not released
  Before the user releases the frame map, SyncMultiFrameListener
  saves the frame within. SyncMultiFrameListener also discards
  new frames after it already saves one frame. This effectively
  creates a triple buffer, and is not supported by PoolAllocator
  of size 2.
  To remove the triple buffer, now SyncMultiFrameListener returns
  false and does not save any frames before the user releases
  the frame map.
* allocator: Handle unordered allocate()/free()
  Due to the frame listener API, its exchange of frames will be
  unordered unlike that between stream parsers and processors.
  `lock(); next = !next` cannot handle unordered allocate()/
  free(). `try_lock(); lock();` will waste time on the second
  when the first becomes available shortly after.
  Use a conditional variable to handle this.
* cmake: Print feature list
* docs: Add VAAPI dependency instructions
* vaapi: Use more zero-copy operations
  Provide memory-mapped packet buffers allocated by VA-API to the
  RGB stream parser to save a 700KB malloc & memcpy.
  Reuse decoding results from the first JPEG packet for all
  following packets, assuming JPEG coding parameters do not change
  based on some testing.
* vaapi: Remove a 8MB memcpy
* vaapi: Add build support
* vaapi: Add VA-API JPEG decoder
* Refactor DoubleBuffer with memory pools
* Change *RgbPacketProcessor::process() to public
  It was somehow protected accidentally.
* Merge pull request `#574 <https://github.com/LCAS/libfreenect2/issues/574>`_ from hanyazou/delay_start_stream
  Delay start stream in OpenNI2 driver
* openni2: Delay start streaming
* openni2: Add Freenect2Driver::DriverImpl class
* Fix zero length resources array
* vt: Remove incorrectly marked API
* docs: Rewrite README build instructions
* usb: Fix typos in error reporting
  The typos made iai_kinect2 hang.
  Fixes `#570 <https://github.com/LCAS/libfreenect2/issues/570>`_
* Set 0.2 version (in development, not released)
* cmake: Fix old find_package UPPERCASE_FOUND
  We use OriginalCase_FOUND to detect package presence,
  but old CMake only provides UPPERCASE_FOUND.
  Use FOUND_VAR to specify OriginalCase_FOUND.
* cmake: Detect missing rgb processor at build time
  Users get segfaults when they built the new code with
  the old CMake cache, which has no support macro of TurboJPEG.
* Remove test_opengl_depth_packet_processor.cpp
  Dumping of raw USB data and device tables is now provided by
  Dump Processors.
* usb: Add more error checking
  Except in Freenect2Device::stop(), which tries the best to stop.
* usb: Move byte parsing code to response.h
  Out from libfreenect2.cpp
  Also unify the response variable type in parsing functions
  to std::vector from (const unsigned char *, int).
* usb: Check CommandTransaction received length
* usb: Add error reporting to CommandTransaction
  Fix memory management with std::vector
* Add ability to disable RGB or depth stream
  Users want to save USB bandwidth and CPU if they don't use
  RGB or depth.
  Add new `startStreams(bool rgb, bool depth)` to Freenect2Device
  Add options `-norgb -nodepth` to Protonect
* Revert "Fallback is always TurboJPEG"
  This reverts commit c3f9aaeac19be3c19f543881e32696ff7f1ba7bc.
  I changed the original commit to use TurboJpegRgbPacketProcessor
  as the fallback always without checking its macro. It would FTBFS
  when TurboJPEG is not enabled.
* Missing frame parameters
* Fallback is always TurboJPEG
* Merge pull request `#365 <https://github.com/LCAS/libfreenect2/issues/365>`_ from fran6co/vt_rgb
  New VideoToolbox rgb packet processor
* New VideoToolbox rgb packet processor
  Mac OS X >= 10.8 has hardware accelerated jpeg decoding (a bit hidden)
* Merge pull request `#549 <https://github.com/LCAS/libfreenect2/issues/549>`_ from matthieu-ft/master
  registration: Add depth-only methods
* registration: Add depth-only methods
  - undistortDepth() is the equivalent for apply() but without color
  - getPointXYZ() is the equivalent for getPointXYZRGB() without color
  This commit enables to work only with the depth without having to process the color image.
  Indeed, the implementation forces you so far to register the color image if you want
  to compute any 3D Point associated with a pixel value. This is time consuming and
  critical for applications that require to be run in real time.
* Merge pull request `#554 <https://github.com/LCAS/libfreenect2/issues/554>`_ from brendandburns/master
  dump: Add accessors for the various depth tables.
* Add accessors for the various depth tables.
* Merge pull request `#551 <https://github.com/LCAS/libfreenect2/issues/551>`_ from brendandburns/master
  Add a dump depth processor.  Reactivate the RGB dump processor.
* Add a dump depth processor.  Reactivate the RGB dump processor.
  Add a dump pipeline.
* Add Zenodo DOI badge
* docs: Provide a PPA for OpenNI2 on trusty
* cpu: Split case of r1yi bigger than 352
  Due to known range of the x coordinate, "rizi >> 4" cannot go beyond 352.
  The only way to get there is due to having an out-of-bound pixel (x, y) coordinate.
  Therefore, "return lut11to16[0]" happens only for a true boolean condition.
* cpu: Merge booleans, eliminate bfi and r4wi
* cpu: Move 'data' access function
  To the point where it is needed.
* cpu: Refactor processMeasurementTriple
* Adding in CLI -help option and -version option
* Merge pull request `#523 <https://github.com/LCAS/libfreenect2/issues/523>`_ from xlz/openni2
  OpenNI2 driver
* tools: Add mkcontrib.py
* openni2: Fix compiler warnings and extra headers
* openni2: Move method definitions out of headers
* openni2: Add build instructions
* openni2: Refactor setVideoMode() and getSensorInfo() in VideoStream class
* openni2: Add OpenKinect Project's license headers
* openni2: Use OpenNI2 logging functions/classes
* openni2: Add timestamp on the frames
* openni2: Add registration
  @HenningJ has the following contribution to this commit:
  Change copying of color images to reflect the change from BGR
  to BGRX color format.
* openni2: Add IrStream class
* openni2: Add proper build system
  make install to copy libfreenect2-openni2* to lib/OpenNI2/Drivers.
  make install-openni2 to cmake -E copy_directory OpenNI2/Drivers
* openni2: Adapt to libfreenect2 API
  Test with /opt/OpenNI2/Tools/NiViewer.
* openni2: Copy OpenNI2-FreenectDriver
  From libfreenect 89f77f6d2c23876936af65766a4c140898bc3ea8
* Add a maintainer
* Merge pull request `#530 <https://github.com/LCAS/libfreenect2/issues/530>`_ from xlz/release-cleanup
  Release cleanup, fix memleaks, packaging helpers.
* Merge pull request `#520 <https://github.com/LCAS/libfreenect2/issues/520>`_ from xlz/macosx-opengl32
  opengl: Lower version to 3.2 for older Mac OSX
* Add windows packaging script and text
* Merge pull request `#526 <https://github.com/LCAS/libfreenect2/issues/526>`_ from xlz/libusb-msvc2015
  Update libusb build script for msvc 2015
* Update libusb build script for msvc 2015
  libusb upstream has merged msvc 2015 support.
* Merge pull request `#521 <https://github.com/LCAS/libfreenect2/issues/521>`_ from xlz/usb-troubleshooting
  Usb troubleshooting docs, closes `#516 <https://github.com/LCAS/libfreenect2/issues/516>`_.
* cmake: Fix a typo in FindLibUSB.cmake
  This typo made it unclear why libusb is not found.
  Reported in `#459 <https://github.com/LCAS/libfreenect2/issues/459>`_, `#512 <https://github.com/LCAS/libfreenect2/issues/512>`_, `#458 <https://github.com/LCAS/libfreenect2/issues/458>`_, `#495 <https://github.com/LCAS/libfreenect2/issues/495>`_.
* docs: Mitigate memory fragmentation
  Reported in `#516 <https://github.com/LCAS/libfreenect2/issues/516>`_.
* usb: Suggest LIBUSB_DEBUG=3 for troubleshooting
  LIBUSB_DEBUG=4 is too verbose and mostly useless.
* opengl: Lower version to 3.2 for older Mac OSX
  Proposed by @robozo in `#519 <https://github.com/LCAS/libfreenect2/issues/519>`_.
* Update README.md
* Add missing comment about onNewFrame return value
  Discussion in `#353 <https://github.com/LCAS/libfreenect2/issues/353>`_
* Update README.md
* Update README.md
* typo fix
* extend TOC
* typo fix
* add TOC with link to API docs
* Plug some memory leaks
  viewer.{h,cpp} are ignored this time.
* Fix up coding style to suppress -Wall warnings
* cmake: Add release versioning variables
  Also use shared library versioning .so.x.y.z
  To create a new release, edit the main CMakeLists.txt and change
  PROJECT_VER_PATCH, _MINOR, or _MAJOR.
  CMake's builtin PROJECT_VERSION\_* variables are not backward
  compatible and not used here.
* Organize miscellaneous platform specific files
* docs: Remove GPL Doxyfile comments
  These comments come from Doxygen code and are licensed under GPL
  only. To avoid incompatibility with Apache license, remove them.
* docs: Organize docs and doxygen files together
* Merge pull request `#507 <https://github.com/LCAS/libfreenect2/issues/507>`_ from xlz/preemptive-api-expansion
  Preemptive API expansion
* Merge pull request `#499 <https://github.com/LCAS/libfreenect2/issues/499>`_ from RyanGordon/viewer_memory_leak_fix
  Fix Memory Leak in Viewer.cpp
* Merge pull request `#494 <https://github.com/LCAS/libfreenect2/issues/494>`_ from xlz/mostly-usb-fixes
  Mostly usb fixes
* api: Add status and pixel format fields to Frame
* api: Add return values to Freenect2Device methods
* examples: Show how to pause
* Deallocate VAO and VBO in viewer.cpp so that memory doesn't leak within the GL library
* usb: Add proper warmup sequence
* usb: Request exact size in bulk transfers
  To avoid a lot of
  WARN Event TRB for slot 1 ep 2 with no TDs queued?
  in dmesg.
* usb: Print correct firmware version number
  Blob `#3 <https://github.com/LCAS/libfreenect2/issues/3>`_ is the main one in the firmware's 7 blobs, and should
  represent version of other blobs, except the bootloader blobs
  which is never updated and not to be bothered with about their
  versions.
  The official SDK uses only blob `#3 <https://github.com/LCAS/libfreenect2/issues/3>`_ to report the version. Use it
  for the version number here.
* opencl: Make Beignet to work by default
  Beignet performs self-test and fails for Haswell and kernel 4.0-.
  These environment variables override the self-test.
  Set the variables by default:
  export OCL_IGNORE_SELF_TEST=1
  export OCL_STRICT_CONFORMANCE=0
* Merge pull request `#486 <https://github.com/LCAS/libfreenect2/issues/486>`_ from RyanGordon/bug/protonect_fullwindow_render
  Viewer Scaling Fix
* Fixing slight cropping within viewer
* Patch for viewer scaling in retina displays, contributed by @pookiefoof
* Merge pull request `#490 <https://github.com/LCAS/libfreenect2/issues/490>`_ from xlz/msvc-symbol-resolving
  Fix MSVC FTBFS, closes `#489 <https://github.com/LCAS/libfreenect2/issues/489>`_
* api: Revert workaround in cdd4f06
  The workaround broke MSVC building. MSVC refuses to resolve the
  symbol because the return type is different, which was the
  point of the workaround.
  Alternative workarounds would make it more a mess. I have sent a
  patch to iai_kinect2 directly to use new API.
* Fixing width/height calculation so that each of the 4 viewports has a equal share of the viewer
* Merge branch 'master' into bug/protonect_fullwindow_render
* Merge pull request `#477 <https://github.com/LCAS/libfreenect2/issues/477>`_ from xlz/api-docs
  API documentation
* docs: Remove duplicate comments in the code
  Some comments in the code are duplicate of those in the headers.
* docs: Add all API documentation
  Also fix a few inconsistencies in the code.
* docs: use cmake to configure doxyfile
* api: Follow up refactoring in Registration
* Merge pull request `#484 <https://github.com/LCAS/libfreenect2/issues/484>`_ from ludiquechile/patch-1
  registration.cpp merge fix
* registration.cpp merge fix
  https://github.com/OpenKinect/libfreenect2/pull/441
* Merge pull request `#441 <https://github.com/LCAS/libfreenect2/issues/441>`_ from giacomodabisias/master
  add external allocation parameter for color offset map
* Merge pull request `#479 <https://github.com/LCAS/libfreenect2/issues/479>`_ from xlz/frame-api
  Forward ABI compatibility of Frame
* api: Allow Frame to use external memory
  Frame allocates memory with new[] by default. Provide a way to not
  do that.
* Merge pull request `#476 <https://github.com/LCAS/libfreenect2/issues/476>`_ from xlz/api-cleanup
  API cleanup/refactoring
* cmake: add freenect2_INCLUDE_DIRS
  iai_kinect2 expects this.
* api: Work around setConfiguration in iai_kinect2
  iai_kinect2 used p->getDepthPacketProcessor()->setConfiguration()
  to configure the device. This is deprecated, but here provides
  compatibility for such usage.
* api: Hide private functions in Registation
  Registration class is marked as API. Private functions in
  Registration got exported as symbols.
  Avoid that.
* api: Hide protected function in Freenect2
  Freenect2 class is marked as API. A protected function in
  Freenect2 got exported as a symbol.
  Avoid that.
* api: Remove the abstract class PacketPipeline
  It is a useless duplicate of BasePacketPipeline.
* api: Add a function to configure depth processors
  Since direct access to depth processors is removed, add
  Freenect2Device::setConfiguration() to allow users to
  configure depth processors. This design is consistent with
  IrCameraParams also being processed in Freenect2Device.
* api: Remove packet processors from public API
  Packet processors should not appear in public API. Users never
  directly interact with these classes.
* api: Move packet processor headers to internal
  File moving only.
  Prepare to remove packet processor classes from public API.
* Merge pull request `#465 <https://github.com/LCAS/libfreenect2/issues/465>`_ from stfuchs/feature/camera-settings
  Feature/camera settings
* Merge pull request `#472 <https://github.com/LCAS/libfreenect2/issues/472>`_ from xlz/opencl-platforms
  Add some OpenCL instructions to README
* docs: OpenCL instructions for Mali, Intel etc.
* Merge pull request `#469 <https://github.com/LCAS/libfreenect2/issues/469>`_ from rahulraw/master
  quick README fix
* Merge pull request `#470 <https://github.com/LCAS/libfreenect2/issues/470>`_ from vinouz/patch-1
  Changed gaussian kernel coefficients so that total is 1.0f (was 0.9999999f)
* Update depth_packet_processor.cpp
  Changed gaussian kernel coefficients to have a sum equal to 1.0f
* Update depth_packet_processor.cpp
  Just a check, like in cocktails with 4 thirds....
* quick README fix
* Fixed logic to render the 4 frames in the full window. Also handle window resizing.
* changed default values to 0
* checkout libusb 1.0.20 for manual install, closes `#466 <https://github.com/LCAS/libfreenect2/issues/466>`_
* changed default exposure to 30
* added doxygen comments
* store camera settings in Frame
  Conflicts:
  include/libfreenect2/frame_listener.hpp
  include/libfreenect2/rgb_packet_processor.h
* Merge pull request `#463 <https://github.com/LCAS/libfreenect2/issues/463>`_ from RyanGordon/update_readme
  Updating README to remove no-longer relevant section
* Updating README to remove no-longer relevant section
* typo fix
* Merge pull request `#450 <https://github.com/LCAS/libfreenect2/issues/450>`_ from alberth/cmake_doxygen_config
  Add doxygen configuration and target to cmake
* Merge pull request `#429 <https://github.com/LCAS/libfreenect2/issues/429>`_ from xlz/build-cleanup
  Assorted fixes and cleanup for 0.1
* Merge pull request `#435 <https://github.com/LCAS/libfreenect2/issues/435>`_ from fran6co/fix-apple
  Fixes missing subpackets in OS X
* Add doxygen configuration and target to cmake
  After generating the Makefile, documentation is generated by issueing "make
  doc", and ends up in the "doc" sub-directory in the build directory.
* logging: Fix cerr/cout according to level
  Previously the logging level was reversed for adding a None level,
  but the selection of cerr or cout was not reversed. Fix that.
* examples: Output usage by default
* docs: update README.md
* Fixes "subpacket too large", "not all subsequences received" and LIBUSB_ERROR_OTHER errors for OS X
* fixes wrong function parameter comment
* docs: Fix installation scripts
  Mac OSX users should use package managers to install libusb
  and glfw3. cl.hpp no longer needs downloading.
  Fix install_ubuntu.sh to download debs properly for ARM users.
* cmake: Fix MSVC warnings
* opencl: Improve compatibility
  Add a copy FindOpenCL.cmake from CMake 3.1.0 verbatim except the
  CMake BSD license header, and a path edit.
  Check if libOpenCL.so is compatible with CL/cl.h. If not, issue
  a warning, and revert to OpenCL 1.1 for the processor. Otherwise
  use OpenCL 1.2.
  This should provide a proper solution to the issue in `#167 <https://github.com/LCAS/libfreenect2/issues/167>`_.
* opencl: Add a copy of cl.hpp 1.2 from khronos.org
  opencl-headers of Debian stretch+ and Ubuntu wily+ no longer carry
  cl.hpp. Mac OSX Xcode also does not have cl.hpp.
  Use a local copy to avoid asking users to download cl.hpp which
  requires root to install and may break API beyond control of
  libfreenect2.
  This updated local copy will also solve compiling errors
  "_mm_mfence not declared" in `#139 <https://github.com/LCAS/libfreenect2/issues/139>`_ and `#250 <https://github.com/LCAS/libfreenect2/issues/250>`_.
* cmake: Require libusb 1.0.20 on Linux
  Tell users at configure time libusb 1.0.19 does not work.
  But do not enforce this on Windows or Mac OSX.
* opengl: Fix OpenGL 3.1 support on Windows
  Properly check version and report error in the viewer.
  In OpenGL processor, FBOs must have read buffer properly set up.
  It's possible viewer's shader version 330 needs to be ported to
  version 140, but no bugs were encountered at the moment.
* cmake: Copy DLLs with executables on Windows
  Subsumes PR `#282 <https://github.com/LCAS/libfreenect2/issues/282>`_.
* cmake: Use proper output directories
  EXECUTABLE_OUTPUT_PATH and LIBRARY_OUTPUT_PATH are deprecated
  by CMake. Use proper variables and also set up output path
  for DLLs.
* logging: Improve packet loss messages
  Avoid flooding of packet loss messages on Windows because the
  console is very slow.
  Fix packet loss counting.
* cmake: Fix rebuilding error with stale cache files
  check_c_source_compiles would generate wrong files if the user
  does not set correct variables initially even given correct values
  later. Protect against this scenario.
  This should fix `#418 <https://github.com/LCAS/libfreenect2/issues/418>`_.
  Also remove "-MT" flags for MSVC which seems to do no good here.
* cmake: Improve Visual Studio 2015 support
  Add VS 2015 detection.
  Add scripts for building libusb with VS2013/2015 (in a Git Shell).
  Check MS64/dll paths for libusb, following the official binary
  release file structure.
* cmake: Improve find_library and link usage
  According to CMake docs, "link_directories() is rarely necessary".
  Therefore remove link_directories(), and use find_library()
  after pkg_check_modules() to obtain full paths of libraries.
  Because of policy change of CMP0063, only set visibility properties
  for freenect2. Do not make them global.
* cmake: Simplify export.h usage
  Rename it from "libfreenect2/libfreenect2_export.h" to
  <libfreenect2/export.h>.
* fixes memory deallocation
* Merge pull request `#379 <https://github.com/LCAS/libfreenect2/issues/379>`_ from xlz/remove-hardcode
  Generate depth tables with camera parameters
* makes the map for storing the color offset for each depth pixel a function parameter in order to make the user decide the allocation policy
* Merge pull request `#440 <https://github.com/LCAS/libfreenect2/issues/440>`_ from giacomodabisias/master
  fixes missing std::string include in libfreenect2.hpp
* fixes missing std::string include
* Generate depth tables with camera parameters
  The xtable, ztable, and 11to16 LUT can now be generated with
  camera parameters at runtime according to analysis in `#144 <https://github.com/LCAS/libfreenect2/issues/144>`_.
  The tables are generated during Freenect2Device::start(), and
  passed to depth processors.
  Users can provide custom camera parameters at runtime with new
  API: setIrCameraParams(), and setColorCameraParams(), and depth
  processors will use those instead of USB queried parameters.
  File loading functions in depth processors are removed.
  Hardcoded table binary files are removed.
* Merge pull request `#402 <https://github.com/LCAS/libfreenect2/issues/402>`_ from OpenKinect/floe-no-devtype-custom
  Get rid of CL_DEVICE_TYPE_CUSTOM
* Merge branch 'master' into floe-no-devtype-custom
* Merge pull request `#376 <https://github.com/LCAS/libfreenect2/issues/376>`_ from xlz/megarefactor
  0.1 release build system restructuring
* Get rid of CL_DEVICE_TYPE_CUSTOM
* Update README about restructuring
* Use CMake to generate LIBFREENECT2_API macro
* Separate public and internal API
  Several LIBFREENECT_API macros are removed from identifiers that
  are no longer public. Several headers are moved to internal
  directory and no longer exported.
  Build for Protonect out-of-tree with public API only. This provides
  a demo on how to use the public API.
  Protonect will be built by default in libfreenect2, controlled with
  BUILD_EXAMPLES.
* Do not generate resources in source tree
  Move generated config.h and resources.inc.h to build directory.
* Fix libfreenect2 build paths
  Remove Protonect definitions from the main CMakeLists.txt
  to `examples` directory.
  Fix *.bin paths.
  A few line-end whitespace deletions.
* Update .gitignore to new paths
  example/protonect is no more.
* Code restructuring
  Renaming only commit. Will not build.
* Remove old libfreenect2.h
  It can be found in commit history.
* Raise CMake version requirement to 2.18.12.1
  User reported error with 2.18.12 in `#363 <https://github.com/LCAS/libfreenect2/issues/363>`_. It seems before
  2.18.12.1 transitive dependencies are not correctly resolved.
* Allow custom RPATH settings
  Package distributors can use RPATH to specify local libusb.
* Use BUILD_SHARED_LIBS to control library type
  Right now both shared and static libraries are built at once
  without options for configuration.
  Use CMake standard variable BUILD_SHARED_LIBS to control the build
  type. Reusing shared library objects for static one is a bad idea
  because -fPIC results in slower static code with more bloat. Thus
  the option to build both at once is not provided. Users are free
  to rebuild with -DBUILD_SHARED_LIBS=OFF.
  This implements requests in `#292 <https://github.com/LCAS/libfreenect2/issues/292>`_ and `#263 <https://github.com/LCAS/libfreenect2/issues/263>`_, but reverting `#276 <https://github.com/LCAS/libfreenect2/issues/276>`_.
* Merge pull request `#397 <https://github.com/LCAS/libfreenect2/issues/397>`_ from Tabjones/master
  First prototype of computeCoordinates of point cloud
* converted rgb to float, to suit PointXYZRGB pcl structure
* updated getPointXYZRGB function, to compute a single point at a time
* first prototype of computeCoordinates, to be tested
* add comment about problems with PCI-E x1 slots
* Merge pull request `#393 <https://github.com/LCAS/libfreenect2/issues/393>`_ from xlz/macosx-opengl
  Fix GLFW setup on Mac OSX, closes `#386 <https://github.com/LCAS/libfreenect2/issues/386>`_
* opengl: Fix GLFW setup on Mac OSX
  Fix user reported error in `#386 <https://github.com/LCAS/libfreenect2/issues/386>`_.
  On Mac OSX, GLFW must be set up with OpenGL 3.2+, AND forward
  compatible, AND with core profile.
* Merge pull request `#391 <https://github.com/LCAS/libfreenect2/issues/391>`_ from xlz/null-filename
  Check NULL filename in the custom logger
* examples: Check NULL filename in the custom logger
  User reported error of opening NULL filename with debug profile.
* Merge pull request `#372 <https://github.com/LCAS/libfreenect2/issues/372>`_ from fran6co/stdlib
  stdlib threading is only available for c++11
* Merge pull request `#385 <https://github.com/LCAS/libfreenect2/issues/385>`_ from xlz/pr383fixed
  Minor bugfixes (logger, freestore handling), closes `#383 <https://github.com/LCAS/libfreenect2/issues/383>`_
* Fix mem free bug and null pointer error
  When exiting libfreenect2::CpuDepthPacketProcessor::process() is
  called but listener\_ pointer is NULL. Adding checking to listener\_.
  First time deleting not alloced mem pointer buffer\_ will fail.
  When creating Mat buffer\_ set it to NULL.
* Add logger.h and logging.h declaration to CMakeLists.txt
  Remove LOG\_* in external code in viewer.h to fix link error
  Add return to logging.cpp's stopTiming function to fix compile error
* Merge pull request `#380 <https://github.com/LCAS/libfreenect2/issues/380>`_ from alberth/add_doxydocs
  Add: Doxygen documentation comment for many of the classes.
* Add: Doxygen documentation comment for many of the classes.
* Merge pull request `#368 <https://github.com/LCAS/libfreenect2/issues/368>`_ from xlz/intel-opengl
  Intel Mesa OpenGL bug fixes and cleanup
* Output less warnings in depth stream parser
  Assembly errors and lost packets should not flood the log output.
* usb: Improve error reporting
* opengl: Clean up flextGL definitions
  Remove commented definitions. They can be found in commit history.
  Move OpenGL version check out of flextGL, and use LOG\_* macros
  for error reporting.
* opengl: Add error reporting at major positions
* opengl: Work around buggy booleans in Mesa
  Mesa 10.2.9 and older versions are oblivious to a behavior change
  in the CMP instruction on Intel CPU SandyBridge and newer.
  On SandyBridge and newer ones, CMP instruction sets all bits to one
  in dst register (-1) as boolean true value. Before that, only the
  LSB is set to one with other bits being undefined.
  Mesa 10.2.9 and older use XOR instruction on the LSB for the logical
  not operator, which produces -2 as boolean value for !true.
  The value is then used by SEL instruction in mix(), which compares
  the value with zero and does not clear high bits before that,
  selecting wrong components.
  A macro MESA_BUGGY_BOOL_CMP is added to forcibly convert -1 to 1
  for Mesa 10.2.9 and older before logical not result is used for
  mix(). The rest of comparison operators and conditionals are safe
  from this behavior.
  I could not independently reproduce this behavior in a seperate
  standalone problem. It is possibly because instruction generation
  varies from optimization.
  This behavior was fixed in Mesa upstream
  2e51dc838be177a09f60958da7d1d904f1038d9c, only appearing in 10.3+.
* opengl: Fix unsupported F32C3 format on Intel/Mesa
  F32C3 format is not supported on Intel/Mesa making FBOs incomplete.
  Just change F32C3 to F32C4, and vec3 output automatically expands
  to vec4.
  Also add completeness checks to each FBO.
* opengl: Limit texture size to 4k on Intel
  Intel/Mesa has GL_MAX_RECTANGLE_TEXTURE_SIZE=4096, but this was
  asking for 424*10.
  Drop the 10th frame which seems useless now, so the texture size
  works for Intel/Mesa.
* changed minimal opengl version to 3.1
* Merge pull request `#364 <https://github.com/LCAS/libfreenect2/issues/364>`_ from xlz/logging
  Logging refactoring continued
* Work around buggy OpenCL ICD loader
  ocl-icd under 2.2.3 calls dlopen() in its library constructor
  and accesses a thread local variable in the process. This causes
  all subsequent access to any other thread local variables to
  deadlock.
  The bug is fixed in ocl-icd 2.2.4, which is not in stable releases
  in Ubuntu or Debian. Thus this provides a workaround given buggy
  ocl-icd.
  To avoid access to thread local variable, errno, std::ostream
  with unitbuf, and exception handling in libstdc++ cannot be used.
  This commit checks ocl-icd version, and refactor the OpenCL
  processor to not use exceptions. Then disable unitbuf on std::cerr
  and disable all exceptions with -fno-exceptions (when available).
  This commit and the ocl-icd bug do not affect Mac OS X or Windows.
* Allow Protonect to run without a viewer
* Add an example on how to create custom logger
  Also export level2str() in Logger for external use.
* Move timing code into logging system
  Also implement a WithPerfLogging class based on timing code to
  remove duplicate timing code in several processors.
* Use LOG\_* macros in remaining classes
* Separate internal logging.h and API logger.h
  Also add a "None" logging level
  Thus remove NoopLogger, and sort logging levels by verbosity.
* Convert to a global static logger
  Before this commit, logger pointers get passed around through
  inheritance and manually constructed dependency assignment lists.
  The manual management is hard to scale with logging calls which
  can appear anywhere in the code.
  This commit implements a single global static logger for all
  Freenect2 contexts. It still can be replaced by different
  loggers, but only one at a time.
  Now it is the responsibility of each logging point to include
  libfreenect2/logging.h, which is not automatically included.
* Use LOG\_* macros in all classes except packet processors
* Changed LOG\_* macros to prepend function signature
* Initial log api definition
  fixed WithLogImpl::setLog; removed global ConsoleLog instance;
  updated Freenect2 to manage lifetime of Log instance
  renamed Log to Logger
  added LIBFREENECT2_API macro to logging classes
  added environment variable LIBFREENECT2_LOGGER_LEVEL to change
  default logger level, possible values
  'debug','info','warning','error'
  made logger level immutable
* Merge pull request `#374 <https://github.com/LCAS/libfreenect2/issues/374>`_ from fran6co/win32
  Fixes Windows compilation, closes `#373 <https://github.com/LCAS/libfreenect2/issues/373>`_
* Fixes Windows compilation
* stdlib threading is only available for c++11
  Mac OSX doesn't support thread_local, but libfreneect is not using it
* Merge pull request `#362 <https://github.com/LCAS/libfreenect2/issues/362>`_ from xlz/remove-opencv-docs
  Update OpenCV docs
* Remove README.depends.txt
  Total duplicate content from README.md
* Remove OpenCV references from README.md
* Merge pull request `#360 <https://github.com/LCAS/libfreenect2/issues/360>`_ from larshg/master
  Add postfix to have both debug and release libraries.
* Merge pull request `#361 <https://github.com/LCAS/libfreenect2/issues/361>`_ from fran6co/glviewer
  Removes opencv dependency, add OpenGL viewer & own timer class
* Removes Opencv for good
* Creates a timer class
* Fixes some compilation issues on Mac
* Added viewer to Protonect
  Added define for opencv to be able to use either opencv or opengl.
  Removed dublicate of flextGL .c/.h
* removed most of the opencv dependencies
  fixed compilation; fixed segfaults in CpuDepthPacketProcessor; disabled timing
* Merge pull request `#357 <https://github.com/LCAS/libfreenect2/issues/357>`_ from goldhoorn/fix_libusb_find_script
  Corrected handling of DEPENDS_DIR and extended description of it
* Add postfix for havng both debug and release libraries.
* Corrected handling of DEPENDS_DIR and extended description of it
* Merge pull request `#351 <https://github.com/LCAS/libfreenect2/issues/351>`_ from goldhoorn/fix_libusb_find_script
  Correct find_scrpipt for libusb
* Correct find_scrpipt for libusb
  The DEPENDS is only set for a local installation.
  Otherwise the system (global) one should used.
  Furthermore the check if libusb was actually found
  (even reuqired) was broken
* Merge pull request `#345 <https://github.com/LCAS/libfreenect2/issues/345>`_ from AliShug/master
  Remove `roundf()` use from Registration
* Merge pull request `#341 <https://github.com/LCAS/libfreenect2/issues/341>`_ from larshg/master
  Exit on opengl errors
* Remove `roundf()` use from Registration
  Replaces use of `roundf()` function in registration.cpp with `(int)(x +
  0.5f)` to allow compiling on older versions of MSVC.
* Exit on opengl (3.3) error.
  Added more error message if creation of flextgl, glfw or glfwwindow fails.
* Merge pull request `#328 <https://github.com/LCAS/libfreenect2/issues/328>`_ from xlz/macosx-docs
  Mac OS X docs update
* Update README.md
  Include build dependencies: wget, git, autotools
  Do not brew install libusb.
  Do not build turbojpeg from source.
  Do not cmake CMakeLists.txt in source directory.
* Update README.md
* Merge pull request `#326 <https://github.com/LCAS/libfreenect2/issues/326>`_ from floe/frame-align
  make sure data pointer in Frame object is 64-byte aligned
* amend pointer arithmetic (by @xlz), protect internals (by @christiankerl)
* remove useless include
* make sure data pointer in Frame object is 64-byte aligned
* Merge pull request `#324 <https://github.com/LCAS/libfreenect2/issues/324>`_ from floe/opencl-fix
  fix opencl rebuild after config change
* fix opencl rebuild after config change
* Update README.md
* Merge pull request `#317 <https://github.com/LCAS/libfreenect2/issues/317>`_ from floe/registration-hd
  allow supplying an external Frame for the depth buffer
* Merge pull request `#318 <https://github.com/LCAS/libfreenect2/issues/318>`_ from hanyazou/xcode-opencl-header
  Use newer OpenCL include path to save cl.hpp
* Use newer OpenCL include path to save cl.hpp
* allow supplying an external Frame for the depth buffer
* Merge pull request `#293 <https://github.com/LCAS/libfreenect2/issues/293>`_ from HenningJ/opencl-build
  Build OpenCL program as soon as the OpenCL device is initialized
* Merge pull request `#315 <https://github.com/LCAS/libfreenect2/issues/315>`_ from wiedemeyer/open_device_fix
  fixed memory leak in openDevice
* added note to header file.
* fixed memory leak due to unknown state of packet pipeline pointer.
* Update README.md
* Update README.md
* Merge pull request `#308 <https://github.com/LCAS/libfreenect2/issues/308>`_ from HenningJ/patch-1
  Raise required CMake version to 2.8.12
* Raise required CMake version to 2.8.12
* Build OpenCL program as soon as the OpenCL device is initialized.
  Before this, the program was built when the first frame arrives and the following frames were dropped, because building the program takes a while.
  Now, the program is built before the device is started. When the first frame arrives, it only needs to be initialized, which is quite fast.
* Merge pull request `#301 <https://github.com/LCAS/libfreenect2/issues/301>`_ from goldhoorn/comments
  Added comments for lib-names
* Merge pull request `#300 <https://github.com/LCAS/libfreenect2/issues/300>`_ from goldhoorn/fix_turbojpeg
  Extended name of libtubrojpeg for debian packaging
* Added comments for lib-names
* Merge pull request `#289 <https://github.com/LCAS/libfreenect2/issues/289>`_ from goldhoorn/pkg-config
  Added pkg-config file to support external library usages
* Merge pull request `#294 <https://github.com/LCAS/libfreenect2/issues/294>`_ from laborer2008/master
  Various small fixes
* Merge pull request `#299 <https://github.com/LCAS/libfreenect2/issues/299>`_ from xlz/ubuntu-deps
  Fix Ubuntu 14.04 installation issues
* Extended name of libtubrojpeg for debian packaging
* Fix Ubuntu 14.04 installation issues
  On Ubuntu 14.04, libturbojpeg.a and turbojpeg.h are provided by
  libjpeg-turbo8-dev, and libturbojpeg.so.0 is provided by
  libturbojpeg. Both packages are needed for building shared library.
  Also, libglfw3-doc requires unrelated dependency libjs-jquery.
  libglfw3-doc is not required for building and can be removed.
* Variable 'success' is reassigned a value before the old one has been used
* rethrow caught exception instead of creation a new one.
  See details: http://en.cppreference.com/w/cpp/language/throw
* throw operator is an exit point from the function. Next return is unnecessary
* More complete checking of Registration::apply() arguments:
  depth pointer is dereferenced afterwards and therefore should be controlled
* Merge pull request `#290 <https://github.com/LCAS/libfreenect2/issues/290>`_ from hanyazou/libfreenect2-h
  Fix compile error in libfreenect2.h
* Merge pull request `#253 <https://github.com/LCAS/libfreenect2/issues/253>`_ from wiedemeyer/improved_registration
  Added filtering of shadowed color regions to registration
* Fix compile error in libfreenect2.h
* Added pkg-config file to support external library usages
* Changed jpeg processor to always output BGRX format.
  Updated registration and removed handling of 3 byte color images.
  Updated protonect to display color image correct.
* updated protonect due to registration changes.
* small bug fix. always output 4 byte color image and alpha channel is set to zero.
* made filtering optional, but enabled by default.
* registration code can now handle 3 byte and 4 byte color images.
* implemented filtering of shadowed regions.
* added comments, moved an addition out of the loop, simplified color image boundary check.
* Apply will also undistort the depth image.
  Improved speed, there was still a double conversion in one if statement.
* fixed bug and simplified a formula.
* Improved speed of registration by factor 5.
  Changed type for registered image to libfreenect2::Frame, so that it is possible to check for correct size.
  Changed layout of maps to be similar to the image layout.
  Added a map for precomputed y color indices.
* Merge pull request `#276 <https://github.com/LCAS/libfreenect2/issues/276>`_ from floe/static_shared
  create static and shared library from same source build
* Merge pull request `#278 <https://github.com/LCAS/libfreenect2/issues/278>`_ from xlz/refactor-opencl
  Move loadBufferFromResources() to resource.h from OpenCL depth processor
* add special MSVC case for static library name
* Merge pull request `#279 <https://github.com/LCAS/libfreenect2/issues/279>`_ from xlz/docs
  Documentation update
* Docs: update Windows instructions
* Docs: OpenCL on Linux instructions
  Stolen from iai_kinect2.
* Docs: update Linux instructions
* Docs: update Mac OSX instructions
* Docs: update hardware compatibility notes
* Move loadBufferFromResources() to resource.h
  CUDA depth processor will also use this function.
* Merge pull request `#277 <https://github.com/LCAS/libfreenect2/issues/277>`_ from larshg/findlibusbfix
  Add libusb as a path_suffixes - as libusb doesn't have a include folder.
* Merge pull request `#275 <https://github.com/LCAS/libfreenect2/issues/275>`_ from xlz/transfer-pool
  Fix transfer pool thread safety
* Add libusb as a path_suffixes - as libusb doesn't have a include folder.
* create static and shared library from same source build
* move resources.inc to resources.inc.h so cmake knows how to handle it
* Merge pull request `#274 <https://github.com/LCAS/libfreenect2/issues/274>`_ from xlz/cmake
  CMake cleanup
* Fix a path typo in FindLibUSB.cmake
* Fix transfer pool thread safety
  Avoid unsafe access during transfer resubmission by refactoring
  TransferPool using std::vector.
  Wait for all transfers during cancellation.
* Use DEPENDS_DIR to simplify paths
* Clean up FindTurboJPEG.cmake on Linux/Mac/Win
* Fix coding style in FindTurboJPEG.cmake
* Simply FindLibUSB.cmake for Windows
  Also, do not maintain two libusb profiles (Release/Debug).
  The user can choose one to build libfreenect2 against.
* Clean up FindLibUSB.cmake on Linux and Mac OSX
* Move FindLibUsb-1.0.cmake to FindLibUSB.cmake
* Clean up FindGLFW3.cmake
* Merge pull request `#270 <https://github.com/LCAS/libfreenect2/issues/270>`_ from larshg/libusbFixs
  Added depends search path.
* Merge pull request `#269 <https://github.com/LCAS/libfreenect2/issues/269>`_ from larshg/findglfwfixes
  Added default install path to glfw on windows
* Merge pull request `#272 <https://github.com/LCAS/libfreenect2/issues/272>`_ from larshg/Dependsguidewindows
  Getting dependencies on windows.
* Added default install path to glfw on windows for include and lib search paths.
  added static name of glfw libraries.
* formatting
* Merge pull request `#271 <https://github.com/LCAS/libfreenect2/issues/271>`_ from floe/depends_v2
  More modular solution for dependency installation
* Merge pull request `#268 <https://github.com/LCAS/libfreenect2/issues/268>`_ from floe/rpath
  add libusb directory to RPATH
* fix for missing turbojpeg link
* fix pkgconfig path to include depends/ folder
* split dependency installation scripts, use official glfw3 .deb packages
* add libusb directory to RPATH
* Added depends search path.
  Removed old paths and text.
  Added condition if debug is not found to set debug as the release library.
* Merge pull request `#266 <https://github.com/LCAS/libfreenect2/issues/266>`_ from xlz/set-e
  Make install script abort on errors
* Merge pull request `#265 <https://github.com/LCAS/libfreenect2/issues/265>`_ from xlz/macosx-docs
  Quick documentation fix
* Make install script abort on errors
* Documentation fix
  - Fix a typo
  - How to verify USB 3 on Mac OS X
  - How to verify linked libusb
* Use external turbojpeg
  Issue `#184 <https://github.com/LCAS/libfreenect2/issues/184>`_ reported turbojpeg built from source produces corrupted
  output. Use pre-built binary from homebrew for now.
* Merge pull request `#264 <https://github.com/LCAS/libfreenect2/issues/264>`_ from OpenKinect/glfw3_fix
  fix GLFW3 conditional
* Merge pull request `#260 <https://github.com/LCAS/libfreenect2/issues/260>`_ from larshg/findturbojpegfixes
  Missing include and lib for default path on windows.
* Missing include and lib for default path on windows.
  Missing /include and /lib for depends folder.
* fix GLFW3 conditional
* Merge pull request `#259 <https://github.com/LCAS/libfreenect2/issues/259>`_ from OpenKinect/cmake_libusb_1.0
  search for libusb-1.0 instead of libusb
* search for libusb-1.0 instead of libusb
* Merge pull request `#257 <https://github.com/LCAS/libfreenect2/issues/257>`_ from larshg/FixFindLibJPEG
  Streamlined the JPEG and added environment to work on linux/mac too.
* Streamlined the JPEG and added environment to work on linux/mac too.
  Added depends/libjpeg_turbo as search path
* Merge pull request `#68 <https://github.com/LCAS/libfreenect2/issues/68>`_ from larshg/libfreenect2FindLibs
  Added FindLibrary files for various libraries
* Added two missing spaces.
* Changed to have a single enviroment variable.
  So you set it up for either 32 or 64 bits. Not both.
* Removed _DIR from the path variable to be consistent with other libraries.
* Corrected indention.
  Removed Lib found announcement.
  Removed lib was already known.
* Added intelSDK enviroment path.
* Added Findlibraries cmake files, to search for the respective libraries, instead of hardcoding in a sub depend folder.
  Added pkg-config support for linux to find libraries externally.
* Merge pull request `#70 <https://github.com/LCAS/libfreenect2/issues/70>`_ from larshg/libfreenect2headers
  Added header files so they are visible in VS solution tree.
* Merge pull request `#240 <https://github.com/LCAS/libfreenect2/issues/240>`_ from floe/faq
  add a brief (linux-centric) FAQ section
* Added header files, so they are visible in VS solution.
* Merge pull request `#241 <https://github.com/LCAS/libfreenect2/issues/241>`_ from larshg/DublicatedName
  Renamed contrib folder to rules
* Renamed folder
* add a brief (linux-centric) FAQ section
* Merge pull request `#239 <https://github.com/LCAS/libfreenect2/issues/239>`_ from floe/udev
  add udev rules file by @wiedemeyer
* add udev rules file by @wiedemeyer
* Merge pull request `#238 <https://github.com/LCAS/libfreenect2/issues/238>`_ from gaborpapp/cpu-depth-packet-unused-variable-remove
  commented out unused variable from cpu_depth_packet_processor.cpp
* commented out unused variable
* Revert "removed unused variable"
  This reverts commit 7161148b2488a3e0c48afc7dbf4a02c52c1efb60.
* Merge pull request `#236 <https://github.com/LCAS/libfreenect2/issues/236>`_ from wiedemeyer/extended_protonect
  Extension of Protonect to allow selection of pipeline and device
* added check for connected devices.
* fixed type, removed enum, shortened code, initialize serial with default.
* removed unused variable
* Merge pull request `#221 <https://github.com/LCAS/libfreenect2/issues/221>`_ from xlz/stream-parsers
  Improve RGB and depth stream parsers
* Extended Protonect to allow selection of the pipeline and the device via parameters.
* Pass timestamps and sequence numbers
  Pass timestamps and sequence numbers from {rgb,depth} stream
  processors to turbojpeg rgb processor and {cpu,opengl,opencl}
  depth processors, then to rgb and depth frames.
  This commit subsumes PR `#71 <https://github.com/LCAS/libfreenect2/issues/71>`_ by @hovren and `#148 <https://github.com/LCAS/libfreenect2/issues/148>`_ by @MasWag.
* Clean up depth stream parser
  Remove magic footer scanning: may appear in the middle.
  Assume fixed packet size.
* Add detailed RGB stream checking
  Inspect the magic markers at the end of a JPEG frame
  and match the sequence number and length.
  Find out the exact size of the JPEG image for decoders
  that can't handle garbage after JPEG EOI.
* Merge pull request `#227 <https://github.com/LCAS/libfreenect2/issues/227>`_ from laborer2008/master
  Updated error reporting messages
* Merge pull request `#226 <https://github.com/LCAS/libfreenect2/issues/226>`_ from floe/registration
  add convenience method & sample code for registration
* Merge pull request `#225 <https://github.com/LCAS/libfreenect2/issues/225>`_ from hanyazou/master
  Use cl_device_type for clGetDeviceInfo(CL_DEVICE_TYPE) instead of size_t...
* switch to portable unsigned char*
* allocate registration object on freestore
* allocate registered image on freestore
* remove noise by setting skipped pixels to zero
* Correct function name for more error messages
* Merge branch 'master' of https://github.com/laborer2008/libfreenect2
* Actualized error reporting messages in the rgb_packet_stream_parser.cpp .
  According to the history RgbPacketStreamParser::handleNewData() function
  was renamed to RgbPacketStreamParser::onDataReceived().
* use bytes_per_pixel instead of hardcoded value
* Merge pull request `#224 <https://github.com/LCAS/libfreenect2/issues/224>`_ from laborer2008/master
  Fixed shebang for all the depends scripts
* add all-in-one registration convenience function
* remove duplicate undistort_depth call
* Merge branch 'master' into registration
* Use cl_device_type for clGetDeviceInfo(CL_DEVICE_TYPE) instead of size_t.
* Fixed shebang for all the depends scripts
* Merge pull request `#207 <https://github.com/LCAS/libfreenect2/issues/207>`_ from xlz/msvcbug
  Fix FTBFS on ARM introduced in PR `#103 <https://github.com/LCAS/libfreenect2/issues/103>`_
* Fix FTBFS on ARM introduced in PR `#103 <https://github.com/LCAS/libfreenect2/issues/103>`_
  PR `#103 <https://github.com/LCAS/libfreenect2/issues/103>`_ tried to fix a linking issue in Visual Studio 2013 on
  Windows 7. It added multiple explicit template instantiations
  which violates the standard and results in failure to build
  from source on ARM.
  Further testing failed to reproduce the linking issue with
  Visual Studio 2013 on Windows 8.1. Thus this commit removes
  the explicit template instantiations.
* Merge pull request `#166 <https://github.com/LCAS/libfreenect2/issues/166>`_ from larshg/VSSolutionRemove
  Remove the VS solution as it is outdated.
* Merge branch 'master' into registration
* Merge pull request `#171 <https://github.com/LCAS/libfreenect2/issues/171>`_ from gaborpapp/texture-upload-fix
  fixed OpenGLDepthPacketProcessor texture upload
* Merge pull request `#167 <https://github.com/LCAS/libfreenect2/issues/167>`_ from goldhoorn/nvidiafix
  Make opencl processor compiling on newer linux nvidia CL version
* switch to pass-by-value for camera param blocks
* Merge pull request `#111 <https://github.com/LCAS/libfreenect2/issues/111>`_ from gaborpapp/test_opengl-osx-fix
  test_opengl OSX fix
* Merge pull request `#180 <https://github.com/LCAS/libfreenect2/issues/180>`_ from Lyptik/master
  Added <limit> header missing and preventing compiling on Ubuntu 14.04.2
* Merge pull request `#170 <https://github.com/LCAS/libfreenect2/issues/170>`_ from gaborpapp/char-comparison-fix
  fixed char comparison warning
* Merge pull request `#190 <https://github.com/LCAS/libfreenect2/issues/190>`_ from floe/registration
  add basic Registration class based on information by @sh0
* add missing transfer of fields from raw command response
* switch to external structures
* add missing color camera parameters
* store local copy of camera params
* add missing transformation to depth camera coordinates
* Merge pull request `#189 <https://github.com/LCAS/libfreenect2/issues/189>`_ from wiedemeyer/opencl_filter_fix
  fix for opencl implementation of the bilateral filter as discussed in `#183 <https://github.com/LCAS/libfreenect2/issues/183>`_
* fix for opencl implementation of the bilateral filter
* add apply method
* add first part of actual mapping (LUT generation)
* add registration class
* Merge pull request `#179 <https://github.com/LCAS/libfreenect2/issues/179>`_ from blen2r/master
  Added automake to list of dependencies for Ubuntu 14.04
* Added <limit> header missing and preventing compiling on Ubuntu 14.04.2
* Added automake to list of dependencies for Ubuntu 14.04
* fixed OpenGLDepthPacketProcessor texture upload
* fixed char comparison warning
* Make opengl processor compining on newer linux nvidia CL version
* Remove the VS solution as it is outdated.
  Updated README for now
* Merge pull request `#162 <https://github.com/LCAS/libfreenect2/issues/162>`_ from floe/fix_script
  fix typo in shell script variable
* fix typo in shell script variable
* Merge pull request `#149 <https://github.com/LCAS/libfreenect2/issues/149>`_ from christiankerl/update_libusb_dependency
  updated libusb dependency
* Merge pull request `#158 <https://github.com/LCAS/libfreenect2/issues/158>`_ from floe/registration
  add info about intrinsic structure by @sh0
* add info about intrinsic structure as provided by @sh0 in `#41 <https://github.com/LCAS/libfreenect2/issues/41>`_
* Merge pull request `#125 <https://github.com/LCAS/libfreenect2/issues/125>`_ from wuendsch/patch-1
  Update README.md - Ubuntu Dependencies
* updated libusb dependency, removed custom patch
* Merge pull request `#130 <https://github.com/LCAS/libfreenect2/issues/130>`_ from christiankerl/optional_opengl_dependencies
  optional OpenGL dependency
* removed glfw include
* added cmake option to disable OpenGL dependencies; choose DefaultPacketPipeline depending on available processors
* Merge pull request `#129 <https://github.com/LCAS/libfreenect2/issues/129>`_ from christiankerl/replace_glew_with_flextgl
  removed GLEW dependency and use OpenGL function loader generated with flextGL
* Merge pull request `#138 <https://github.com/LCAS/libfreenect2/issues/138>`_ from christiankerl/update_freenect2_cmake_in
  updated freenect2.cmake.in
* Merge pull request `#145 <https://github.com/LCAS/libfreenect2/issues/145>`_ from wiedemeyer/opencl_config_fix
  Fix for OpenCL depth packet processor ignoring min and max depth values from config.
* small fix.
* OpenCL depth packet processor now uses min and max depth from config.
  splitted device and program initialization to enable reconfiguration while processor is running.
* updated freenect2.cmake.in; fixes `#131 <https://github.com/LCAS/libfreenect2/issues/131>`_
* added parameter for parent GLFW window pointer to OpenGLPacketPipeline
* changed global OpenGLBindings object to per instance of OpenGLDepthPacketProcessor
* removed GLEW dependency and use OpenGL function loader generated with flextGL
* Merge pull request `#128 <https://github.com/LCAS/libfreenect2/issues/128>`_ from wiedemeyer/FIX_DEFINITION_OCL
  fix for wrong define name: WITH_OPENCL_SUPPORT -> LIBFREENECT2_WITH_OPENCL_SUPPORT
* fix for wrong define name.
* Merge pull request `#127 <https://github.com/LCAS/libfreenect2/issues/127>`_ from christiankerl/fix_shutdown_name_conflict
  renamed global variable shutdown to protonect_shutdown
* renamed global variable shutdown to protonect_shutdown; fixes `#120 <https://github.com/LCAS/libfreenect2/issues/120>`_
* Merge pull request `#103 <https://github.com/LCAS/libfreenect2/issues/103>`_ from christiankerl/generate_macro_header
  generate header file with platform and build configuration macros
* Merge pull request `#124 <https://github.com/LCAS/libfreenect2/issues/124>`_ from wiedemeyer/ocl_device_selection
  added posibility to select openCL device for depth processing and improved openCL device listing
* Merge pull request `#119 <https://github.com/LCAS/libfreenect2/issues/119>`_ from larshg/openclFix
  OpenCL fixes
* Merge pull request `#113 <https://github.com/LCAS/libfreenect2/issues/113>`_ from dorian3d/fix/openDevice-idx
  Fix openDevice idx
* Update README.md
* Update README.md
* added posibility to select openCL device for depth processing.
  if not specified priority is GPU, CPU, others.
  listing of multiple devices work now correctly.
* Added another Enviorment variable on windows.
  Mine is AMDAPPSDKROOT instead of ATISTREAMSDKROOT.
  And cleaned a bit how it was searching.
* Opencl uses M_PI, which is defined <math.h> with _USE_MATH_DEFINES defined.
* Fix openDevice idx
* test_opengl OSX fix
* Merge pull request `#104 <https://github.com/LCAS/libfreenect2/issues/104>`_ from dorian3d/fix/wget-cl
  Do not download cl.hpp if it exists
* updated exports of templated classes to fix visualc++ problems
* Do not download cl.hpp if it exists
* added libfreenect2/config.h defining all platform and build configuration dependend macros; fixes `#100 <https://github.com/LCAS/libfreenect2/issues/100>`_, includes `#69 <https://github.com/LCAS/libfreenect2/issues/69>`_
* Merge pull request `#99 <https://github.com/LCAS/libfreenect2/issues/99>`_ from floe/ignore_fix
  ignore generated resource file
* Merge pull request `#98 <https://github.com/LCAS/libfreenect2/issues/98>`_ from christiankerl/refactor_data_received_callback
  moved DataReceivedCallback from TransferPool to separate header
* Merge pull request `#95 <https://github.com/LCAS/libfreenect2/issues/95>`_ from christiankerl/configurable_opengl_debug_window
  added option to hide the debug window of OpenGLDepthPacketProcessor
* ignore generated resource file
* moved DataReceivedCallback from TransferPool to separate header to break dependencies
* added option to hide the debug window of OpenGLDepthPacketProcessor
* Merge pull request `#94 <https://github.com/LCAS/libfreenect2/issues/94>`_ from christiankerl/fix_packet_pipeline_without_opencl
  fix compilation without OpenCL support
* added #ifdef guard to hide OpenCLPacketPipeline if we build without OpenCL support; added cmake option to enable OpenCL support
* Merge pull request `#81 <https://github.com/LCAS/libfreenect2/issues/81>`_ from christiankerl/add_enable_cxx11_option
  added cmake option to enable c++11
* added option to cmake to enable c++11
* Merge pull request `#58 <https://github.com/LCAS/libfreenect2/issues/58>`_ from christiankerl/opencl_depth_packet_processor
  opencl depth packet processor
* fixing merge artifacts
* added packet pipeline implementations to choose the different built-in DepthPacketProcessors
* implemented a better device selection. Try to use the first GPU device or if not found try to use first CPU device.
  added class and method name to output.
* changed curl to wget to be consistent
* fixing compilation if opencl is not available
* using found opencl library.
* cleaned up CMakeLists.txt. Removed c++11 dependency.
* added opencl implementation of the depth processor.
* fixed opencl source file string
* fixed bug in loadResource
* fixing compilation if opencl is not available
* Allow apple platforms to find the cl.hpp file
* Get the missing cl.hpp from Khronos.org
* using found opencl library.
* Parameters are now read in from the Parameters struct.
  Config is read from the Config struct.
  Removed unused variables from opencl code.
* cleaned up CMakeLists.txt. Removed c++11 dependency.
* added opencl implementation of the depth processor.
* Merge pull request `#66 <https://github.com/LCAS/libfreenect2/issues/66>`_ from larshg/libfreenect2packeddata
  Made a ifdef for packing data to work on windows also.
* Merge pull request `#52 <https://github.com/LCAS/libfreenect2/issues/52>`_ from dorian3d/feature/install
  make install enabled
* Path of freenect2Config.cmake fixed
* Made a ifdef for packing data to work on windows also.
* Merge pull request `#80 <https://github.com/LCAS/libfreenect2/issues/80>`_ from christiankerl/fix_device_identification
  fix device identification
* replaced libusb_get_port_number with libusb_get_device_address to correctly identify devices, fixes `#65 <https://github.com/LCAS/libfreenect2/issues/65>`_
* Merge pull request `#67 <https://github.com/LCAS/libfreenect2/issues/67>`_ from larshg/libfreenect2mathfix
  Added include <algorithm> in ifdef WIN32 and VS2013
* Merge pull request `#77 <https://github.com/LCAS/libfreenect2/issues/77>`_ from davetcoleman/upmaster_readme_formatting
  Formatting README to Markdown format. Thanks @davetcoleman!
* Formatting README to Markdown format
* Added Ubuntu documentation
* Added include <algorithm>
  Added type in std::min/std::max
  Added include <math.h> and _USE_MATH_DEFINES if WIN32
* Merge pull request `#38 <https://github.com/LCAS/libfreenect2/issues/38>`_ from christiankerl/configurable_pipeline
  make packet processing pipeline configurable
* Merge pull request `#57 <https://github.com/LCAS/libfreenect2/issues/57>`_ from christiankerl/refactor_frame_listener
  SyncMultiFrameListener changes
* SyncMultiFrameListener changes:
  - implementation using pimpl - fixes `#48 <https://github.com/LCAS/libfreenect2/issues/48>`_
  - added non-blocking method to check if all frames are available - fixes `#56 <https://github.com/LCAS/libfreenect2/issues/56>`_
  - added timed wait function if compiled with c++0x or c++11
* Definitions and headers for threads added
* make install enabled
  The shared library, headers and a cmake file can be make installed
* Merge pull request `#30 <https://github.com/LCAS/libfreenect2/issues/30>`_ from christiankerl/fix_max_iso_packet_size
  reimplement custom version of libusb_get_max_iso_packet_size
* Merge pull request `#40 <https://github.com/LCAS/libfreenect2/issues/40>`_ from BillinghamJ/patch-1
  Updated readme
* Updated readme
  Added two extra brew dependencies - install will not work without them
* Merge branch 'master' into configurable_pipeline
  Conflicts:
  examples/protonect/include/libfreenect2/libfreenect2.hpp
  examples/protonect/src/libfreenect2.cpp
* updated README
* Merge pull request `#34 <https://github.com/LCAS/libfreenect2/issues/34>`_ from rjw57/reset-workaround
  Workaround for libusb_reset_device behaviour
* libfreenect2: coding style fixes (if braces)
  Make if0statement braces consistent with the rest of the file. (*Mea culpa*.)
* libfreenect2: reinstate tryGetDevice as an error
  If tryGetDevice fails, it *is* unrecoverable as far as initialisation is concerned.
* Merge pull request `#32 <https://github.com/LCAS/libfreenect2/issues/32>`_ from MrTatsch/patch-1
  libjpeg_turbo fails to configure
* openDevice: if tryGetDevice fails, it is a warning not an error
  Change the error message into a warning message and allow open to
  continue.
* change sleep() call to libfreenect2::this_thread::sleep_for()
* introduce a small delay after reset before reenumeration
  This is a rather nasty hack but is required to give certainty that the
  Kinnect has re-appeared on the bus after a reset failed. In the absence
  of a better solution this Gets The Job Done(TM).
* handle LIBUSB_ERROR_NOT_FOUND from libusb_reset_device
  It is possible (and indeed on my controller certain) that
  libusb_reset_device may return LIBUSB_ERROR_NOT_FOUND under certain
  circumstances outlined in the libusb documentation. In such cases we
  should re-start device enumeration and re-open the device without
  attempting reset.
* refactor Freenect2::openDevice to be less nested
  Freenect2::openDevice was in danger of becoming a twisty maze of if/else
  statements all alike.
* removed early exit from install_deps.sh
* changed libusbx dependency to libusb
* Merge branch 'master' into configurable_pipeline
  Conflicts:
  examples/protonect/src/libfreenect2.cpp
* reimplemented custom version of get_max_iso_packet_size, which works for usb 3 endpoints; this allows to switch from the forked libusb version of @JoshBlake to the latest official libusb version;
* Merge pull request `#29 <https://github.com/LCAS/libfreenect2/issues/29>`_ from christiankerl/fix_device_enumeration
  enhance device enumeration
  - implements the methods to get device serial numbers and to open a device identified by its serial number
  - resets device inside openDevice method
* renamed PacketProcessorFactory to PacketPipeline, moved all ownership handling of packet parser and packet processor objects to PacketPipeline
* replaced default argument with method overload
* fixed license header
* refactoring to make rgb and ir packet stream parsers and packet processors configurable, this will allow to easily swap different implementations and even use different implementations per device
* add --host x86_64-apple-darwin flag to configure
  As pointed out in the build recipe of libjpeg_turbo:
  64-bit Build on 64-bit OS X
  ---------------------------
  Add
  --host x86_64-apple-darwin NASM=/opt/local/bin/nasm
  to the configure command line.  NASM 2.07 or later from MacPorts must be
  installed.
  linking will fail on 64bit systems without this flag due to:
  "configure: error: configuration problem: maybe object file format mismatch". I guess there is hardly any macs  out there with USB3 but without 64bit OS. NASM should also be installed, in my case its installed from homebrew and found in the PATH.
* Merge branch 'master' into fix_device_enumeration
  Conflicts:
  examples/protonect/src/libfreenect2.cpp
* Merge pull request `#28 <https://github.com/LCAS/libfreenect2/issues/28>`_ from christiankerl/fix_shader_filter
  renamed filter functions in GLSL shader code, fixes `#27 <https://github.com/LCAS/libfreenect2/issues/27>`_
* Merge pull request `#26 <https://github.com/LCAS/libfreenect2/issues/26>`_ from RyanGordon/refactor_protocol_ryan
  fixes MacOSX compilation; adds usb product ids for release version of Kinect v2;
* moved usb device reset from enumerateDevices() to openDevice() otherwise there are problems if multiple processes use libfreenect2 to access different Kinects
* Removing comment that no longer applies
* Fixing permissions of install files
* Working on abstracting deps for *nix systems and having separate install scripts for mac versus ubuntu
* fixed problem during device enumeration, if device is already open
* fixed SIGINT shutdown problem
* improved device enumeration to open every device, reset it, and get serial number; implemented methods to get serial number and open device by serial number; fixes `#21 <https://github.com/LCAS/libfreenect2/issues/21>`_
* renamed filter functions in GLSL shader code, fixes `#27 <https://github.com/LCAS/libfreenect2/issues/27>`_
* Fixing URL for OpenKinect repo
* Slightly better error message
* Merge pull request `#23 <https://github.com/LCAS/libfreenect2/issues/23>`_ from christiankerl/refactor_protocol
  refactored Kinect v2 control command functions and implemented c++ api
* Fixes for working on Mac OSX
* Merge remote-tracking branch 'christiankerl/refactor_protocol' into refactor_protocol_ryan
  Conflicts:
  examples/protonect/Protonect.cpp
* Fixing up installation instructions
* Fixes for compiling and running libfreenect2 on Max OSX
* changed depth packet processor to opengl version
* removed obsolete protonect path parameter
* fixed memory leak in OpenGLDepthPacketProcessor, if listener doesn't take ownership of frame
* re-enabled p0table flipping in CpuDepthPacketProcessor and added functionality to OpenGLDepthPacketProcessor
* adapted OpenGLDepthPacketProcessor to refactorings
* removed Protonect.h; added cmake build rule for libfreenect2; Protonect is now a single main linking against libfreenect2;
* added methods to access color and ir camera params
* moved all command response parsing to libfreenct2/protocol/response.h; added more commands observed in usb logs; implemented method to get serial number and firmware version
* changed time spent waiting for usb transfer cancel; added more commands observed in usb logs, but still don't allow to restart camera
* finished first version of internal c++ api
* changed frame listener api
* started to implement internal c++ api
* fixed order of transferpool shutdown and device closing
* added more command definitions observed in usb logs; updated shutdown sequence
* moved CommandTransaction implementation to cpp file
* improved error reporting in UsbControl
* removed old usb control and command code from Protonect.cpp
* moved set configuration; claim/release interfaces to UsbControl class
* moved usb control transfers to separate class
* fixed bug in CommandTransaction
* renamed command, which gets the serial number string
* refactored command stuff
* started to refactor control protocol/command functions
* Merge pull request `#19 <https://github.com/LCAS/libfreenect2/issues/19>`_ from christiankerl/opengl_depth_processor
  implemented DepthPacketProcessor using OpenGL shaders
* implemented proper opengl/glew multithreaded context handling
* fixed small bug in first shader stage
* fixed MaxEdgeTest data type
* changed first shader stage such that output norm is equal to the later, in-place norm computation in cpu version
* extended CpuDepthPacketProcessor such that it can be used in the OpenGLDepthPacketProcessor test; fixed some bugs in OpenGLDepthPacketProcessor; there are still some minor differences between cpu and opengl version
* removed shader folder parameter from OpenGLDepthPacketProcessor
* CpuDepthPacketProcessor now uses embedded resources
* added conversion from min/max depth in meters to millimeters when setting configuration of DepthPacketProcessor
* moved common DepthPacketProcessor parameters to struct; replaced hard coded parameter values in shaders with uniform structure; not yet tested;
* explicitly link in pthread on Linux/MacOSX
* removed several functions, which complicate current implementation; added functions to support ir intensity frames
* initial draft for libfreenect2 api based on libfreenect api
* Merge pull request `#12 <https://github.com/LCAS/libfreenect2/issues/12>`_ from christiankerl/api
  initial draft for libfreenect2 api based on libfreenect api
* Merge pull request `#20 <https://github.com/LCAS/libfreenect2/issues/20>`_ from floe/cmake_fix
  explicitly link in pthread on Linux/MacOSX
* explicitly link in pthread on Linux/MacOSX
* embedded resource generation command in cmake now depends on the input files, so it gets recompiled once the input files change
* moved shader layout qualifiers to the correct position
* binary resource, like coefficient tables and shaders are now embedded into the Protonect binary
* OpenGLDepthPacketProcessor now uses its configuration
* removed opencv dependency from OpenGLDepthPacketProcessor
* shortened image format definitions
* implemented gpu depth processing using opengl shaders
* increased opengl version to 3.3
* added glfw and glew as dependencies; implemented basic opengl window display;
* Merge pull request `#17 <https://github.com/LCAS/libfreenect2/issues/17>`_ from christiankerl/second_depth_filter_stage
  - implemented edge aware filter stage in CpuDepthPacketProcessor
  - added configuration options to DepthPacketProcessor interface
* added options to enable/disable the filters to DephPacketProcessor::Config
* added configuration to DepthPacketProcessor, right now just allows to set min and max depth
* fixed small bug
* implemented second filter stage in depth packet processor
* Merge pull request `#14 <https://github.com/LCAS/libfreenect2/issues/14>`_ from christiankerl/remove_boost_threading
  Removed boost thread dependency
* Merge pull request `#16 <https://github.com/LCAS/libfreenect2/issues/16>`_ from christiankerl/fix_depth_stream_parser_segfault
  fixed segfault in DepthPacketStreamParser
* fixed segfault in DepthPacketStreamParser; fixes `OpenKinect/libfreenect2#15 <https://github.com/OpenKinect/libfreenect2/issues/15>`_
* removed boost threads dependency from depends/README.depends.txt
* replaced boost threading dependency with stdlib or tinythread implementation
* removed several functions, which complicate current implementation; added functions to support ir intensity frames
* initial draft for libfreenect2 api based on libfreenect api
* Merge pull request `#9 <https://github.com/LCAS/libfreenect2/issues/9>`_ from christiankerl/ir_iso_transfer
  Merge: Initial working version of libfreenect2
* Merge pull request `#5 <https://github.com/LCAS/libfreenect2/issues/5>`_ from floe/trig_tables
  replace per-pixel trig calculations with table lookups
* use simple n*6 float arrays instead of multiple cv::Mat instances
* Merge branch 'ir_iso_transfer' into trig_tables
  Conflicts:
  examples/protonect/src/cpu_depth_packet_processor.cpp
* added FrameListener to synchronize rgb/ir/depth images and display them in the main thread
* replace per-pixel trig calculations with table lookups
* merged remote
* updated depends README to include change of MAX_ISO_BUFFER_LENGTH and min kernel version
* updated depends README to include change of MAX_ISO_BUFFER_LENGTH and min kernel version
* added patch to increase libusbx MAX_ISO_BUFFER_LENGTH
* removed boost signal dependency
* added shell script to download/build correct libusbx version on ubuntu; updated dependecy list
* added comments with interpretation of two unknown data blocks
* Merge pull request `#3 <https://github.com/LCAS/libfreenect2/issues/3>`_ from floe/intrinsics
  hex dump of unknown response data; parsing color and depth camera intrinsics;
* add guesstimated structure for second block of camera params
* move struct definition for depth cam intrinsics to proper header file
* parse depth camera parameters (partly still guesses)
* remove unnecessary hex prefix
* dump hex data from unknown commands
* Merge pull request `#2 <https://github.com/LCAS/libfreenect2/issues/2>`_ from floe/p0table
  access p0table via struct definition instead of hard-coded offsets
* move p0 table definitions to header dir
* Merge branch 'ir_iso_transfer' into p0table
  Conflicts:
  examples/protonect/src/cpu_depth_packet_processor.cpp
* implemented bilateral filter on differential a/b images; really slow
* simplified rgb/depth packet processor api; async processing is now implemented in a templated decorator class
* change p0table struct as suggested by ck
* replace hard-coded offsets with struct references
* add p0tables definition header
* Merge pull request `#1 <https://github.com/LCAS/libfreenect2/issues/1>`_ from floe/api_cleanup
  API cleanup
* rename command wrapper functions
* refactored the control commands
* Merge branch 'ir_iso_transfer' into api_cleanup
* implemented depth disambiguation
* refactor init/status commands (part `#1 <https://github.com/LCAS/libfreenect2/issues/1>`_)
* removed old Makefile and decode utility programs
* added display of RGB images using libjpeg-turbo decoding and opencv gui
* added some time profiling to depth decoding
* increased number of iso packets
* first working version of ir/depth decoding; several post processing steps like depth disambiguation, bilateral filtering, edge-aware filtering, implemented in the official SDK are missing; the implemented CPU decoding is slow and only runs at 10Hz or less;
* update to capture all iso data and write it to one large binary file; decode extracts individual images from this binary file;
* more depth decoding experiments
* fixed copy&paste bug
* reformated Protonect.cpp; removed opencv dependency; added little program to analyze captured ir data;
* disabled bulk transfer; use correct packet size constant
* implemented simple iso packet capture
* implemented proof of concept for rgb transfer
* Reduced chattiness and increased iso loop count
* don't overwrite buffer size on return
* fix standard includes
* add Makefile
* Initial commit of prototype driver.
  Signed-off-by: Joshua Blake <joshblake@gmail.com>
* Contributors: Albert Gil, Albert Hofkamp, Alistair, Anton Onishchenko, Bill, Brendan Burns, Christian Kerl, DD, Dave Coleman, David Poirier-Quinot, Dorian Galvez-Lopez, Eric Schleicher, Federico Spinelli, Felix, Florian Echtler, Francisco Facioni, Gabor Papp, Giacomo Dabisias, Henning Jungkurth, HenningJ, James Billingham, Jesse Kaukonen, Jonathan Doig, Joshua Blake, Kay, Lars Glud, Lingzhu Xiang, Ludique, Manuel Fernandez-Carmona, Mariano, Mario Wündsch, Matthias Goldhoorn, Matthieu FT, Maxime Tournier, MrTatsch, P.E. Viau, Patrick Stotko, Paul Reynolds, Rich Wareham, Ryan Gordon, Sergey Gusarov, Serguei Mokhov, Steffen Fuchs, Stephen McDowell, Thiemo Wiedemeyer, Zou Hanya, augmenta, christiankerl, hanyazou, larshg, rahulraw, sjdrc, sven, vinouz, yuanmingze
