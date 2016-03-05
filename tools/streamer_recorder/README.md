# libfreenect2 Streamer/Recorder toolbox

## Table of Contents

* [Description](README.md#description)
* [Maintainers](README.md#maintainers)
* [Installation](README.md#installation)
  * [Windows / Visual Studio](README.md#windows--visual-studio)
  * [MacOS X](README.md#mac-osx)
  * [Linux](README.md#linux)


## Description

Additional toolbox featuring:
- UDP streaming of kinect captured images (``-streamer`` option)
- Recording of kinect captured images to disk (``-recorder`` option)

## Maintainers

* David Poirier-Quinot

## Installation

### Windows / Visual Studio

### Mac OSX

* Install OPENCV

    ```
brew install opencv3
```

* Install Numpy for Blender viewer

    ```
pip3 install numpy
```

and link numpy and cv2 to blender python3 site-package (rename / remove old numpy if needed)

(tested with 1.10.4, previous versions happened to raise ``ImportError: numpy.core.multiarray failed to import`` when typing ``import cv2`` in python)

* Build

    ```
mkdir build && cd build
cmake ..
make
make install
```
* Run the test program: `.ProtonectSR`


### Linux

* Install build tools

    ```
sudo apt-get install opencv3
```

* Build

    ```

mkdir build && cd build
cmake ..
make
make install

* Run the test program: `./ProtonectSR`

