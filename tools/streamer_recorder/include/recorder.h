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
