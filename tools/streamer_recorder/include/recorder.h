/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2017 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

#ifndef RECORDER_H
#define RECORDER_H

#include <libfreenect2/frame_listener.hpp>
#include <opencv2/opencv.hpp>
#include "config.h"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>      // file input/output functions
#include <cstdlib>      // time stamp
#include <sys/timeb.h>  // time stamp

class Recorder
{
public:
  // methods
  Recorder();
  void initialize();
  void record(libfreenect2::Frame* frame, const std::string& frame_type);

  void stream(libfreenect2::Frame* frame);

  void saveTimeStamp();
  void registTimeStamp();

private:
  /////// RECORD VIDEO, NOT READY YET (RECORD IMAGE FOR NOW) ///////
  // cv::VideoWriter out_capture;
  /////////////////////////////////////////////////////////////////

  cv::Mat cvMat_frame;

  // SAVE IMAGE
  //cv::vector<int> img_comp_param; //vector that stores the compression parameters of the image
  std::vector<int> img_comp_param; //vector that stores the compression parameters of the image
  int frameID;
  // int maxFrameID; //  16.6min max at 30 FPS (max frame ID sort of hardcoded in image naming too, see below)

  // static int timeStamps [MAX_FRAME_ID];
  std::vector<int> timeStamps;

  int t_start;
  int t_now;

  std::ostringstream oss_recordPath;
  std::string recordPath;
  // -----------------

  // Timer
  timeb tb;
  int nSpan;
  int getMilliSpan(int nTimeStart);
  int getMilliCount();
};

#endif
