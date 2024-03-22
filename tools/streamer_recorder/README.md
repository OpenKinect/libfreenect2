# libfreenect2 Streamer/Recorder toolbox

## Table of Contents

* [Description](README.md#description)
* [Maintainers](README.md#maintainers)
* [Installation](README.md#installation)
  * [Windows / Visual Studio](README.md#windows--visual-studio)
  * [MacOS X](README.md#mac-osx)
  * [Linux](README.md#linux)


## Description

Additional toolbox based off `Protonect` featuring:
- UDP streaming of Kinect captured images (``-streamer`` option)
- Recording of Kinect captured images to disk (``-recorder`` option)
- Replay of Kinect captured images from disk (``-replay`` option)

## Maintainers

* David Poirier-Quinot
* Serguei A. Mokhov

## Installation

### Windows / Visual Studio

TODO

### Mac OS X

* Install OpenCV

```
brew install opencv3
```

* Install Numpy for Blender viewer
```
pip3 install numpy
```

and link numpy and cv2 to blender python3 site-package (rename / remove old numpy if needed)

(tested with 1.10.4, previous versions happened to raise ``ImportError: numpy.core.multiarray failed to import`` when typing ``import cv2`` in python)

* Build (start from the libfreenect2 root directory)
```
mkdir build && cd build
cmake .. -DBUILD_STREAMER_RECORDER=ON
make
make install
```
* Run the test program (accepts all the same options as Protonect with 3 extra):
    - `./bin/ProtonectSR -record` -- to start recording frames
    - `./bin/ProtonectSR -stream` -- to start streaming frames to a receiver application
    - `./bin/ProtonectSR -replay` -- to start replaying recorded frames
    - `./bin/ProtonectSR -replay -stream` -- to relay and stream recorded frames
    - `./bin/ProtonectSR -record -stream` -- to record and stream frames

### Linux

* Install build tools
```
sudo apt-get install opencv3
```

* Build (start from the libfreenect2 root directory)
```
mkdir build && cd build
cmake .. -DBUILD_STREAMER_RECORDER=ON
make
make install
```

* Run the test program (accepts all the same options as Protonect with 3 extra):
    - `./bin/ProtonectSR -record` -- to start recording enable frames (`freenect2-record` presupposes this option)
    - `./bin/ProtonectSR -stream` -- to start streaming frames to a receiver application (`freenect2-stream` presupposes this option)
    - `./bin/ProtonectSR -replay` -- to start replaying recorded frames (`freenect2-replay` presupposes this option)
    - `./bin/ProtonectSR -replay -stream` -- to relay and stream recorded frames
    - `./bin/ProtonectSR -record -stream` -- to record and stream frames
