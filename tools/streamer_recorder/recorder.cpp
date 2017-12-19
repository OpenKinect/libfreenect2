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

#include "recorder.h"
#include <cstdlib>
#include <iomanip>

Recorder::Recorder() : timeStamps(MAX_FRAME_ID)
{
}

// get initial time in ms
int Recorder::getMilliCount()
{
  ftime(&tb);
  int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
  return nCount;
}

// get time diff from nTimeStart to now
int Recorder::getMilliSpan(int nTimeStart)
{
  nSpan = Recorder::getMilliCount() - nTimeStart;
  if(nSpan < 0)
    nSpan += 0x100000 * 1000;
  return nSpan;
}

// get time diff from nTimeStart to now
void Recorder::registTimeStamp()
{
  // record time stamp for FPS syncing
  timeStamps[frameID] = Recorder::getMilliSpan(t_start);
  // printf("Elapsed time = %u ms \n", timeStamps[frameID]);

  frameID++;
}

void Recorder::initialize()
{
  std::cout << "Initialize Recorder." << std::endl;

  /////// RECORD VIDEO, NOT READY YET (RECORD IMAGE FOR NOW) ///////

  // // out_capture.open("TestVideo.avi", CV_FOURCC('M','J','P','G'), 30, cv::Size(depth->height, depth->width)); // JPEG
  // out_capture.open("TestVideo.avi", CV_FOURCC('P','I','M','1'), 30, cv::Size(depth->height, depth->width),1); // MPEG, last argument defines image color yes o (channel 3 or 1)
  // // out_capture.open("TestVideo.avi", CV_FOURCC('D','I','V','X'), 30, cv::Size(depth->height, depth->width));

  // if( !out_capture.isOpened() )
  // {
  // std::cout << "AVI file can not open."  << std::endl;
  // return 1;
  // }

  /////////////////////////////////////////////////////////////////

  // record image: define compression parameters and frame counter
  img_comp_param.push_back(CV_IMWRITE_JPEG_QUALITY); //specify the compression technique
  img_comp_param.push_back(100); //specify the compression quality
  frameID = 0;

  // record timeStamp
  t_start = getMilliCount();
}

void Recorder::record(libfreenect2::Frame* frame, const std::string& frame_type)
{
  if(frame_type == "depth")
  {
    // std::cout << "Run Recorder." << std::endl;
    cvMat_frame = cv::Mat(frame->height, frame->width, CV_32FC1, frame->data) / 10;
    // TODO: handle relative path + check Windows / UNIX compat.
    oss_recordPath << "../recordings/depth/" << std::setw( 5 ) << std::setfill( '0' ) << frameID << ".depth";
  }
  else if (frame_type == "registered" || frame_type == "rgb")
  {
    cvMat_frame = cv::Mat(frame->height, frame->width, CV_8UC4, frame->data);
    // TODO: handle relative path + check Windows / UNIX compat.
    oss_recordPath << "../recordings/regist/" << std::setw( 5 ) << std::setfill( '0' ) << frameID << ".jpg";
    // std::cout << frame->height << ":" << frame->width << ":" << frame->bytes_per_pixel << std::endl;
  }

  recordPath = oss_recordPath.str();

  // SAVE IMAGE
  cv::imwrite(recordPath, cvMat_frame, img_comp_param); //write the image to file
  // std::cout << recordPath << std::endl;

  // show image
  // cv::namedWindow( "recorded frame", CV_WINDOW_AUTOSIZE);
  // cv::imshow("recorded frame", cvMat_frame);
  // cv::waitKey(0);

  // reset ostr
  oss_recordPath.str("");
  oss_recordPath.clear();

  // feedback on current recording state
  if(frameID % 100 == 0)
    std::cout << "-> " << frameID << "/" << MAX_FRAME_ID << " recorded frames/maxFrameID (" << frame_type << ")" << std::endl;

  /////// RECORD VIDEO, NOT READY YET (RECORD IMAGE FOR NOW) ///////

  // cv::Mat frame_depth = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data) / 4500.0f;
  // cv::convertScaleAbs(frame_depth, frame_depth);

  // DISPLAY INFOS
  // std::cout << "kinect (h,w): " << depth->height <<  "," << depth->width << std::endl;

  // cv::Size s = frame_depth.size();
  // double rows = s.height;
  // double cols = s.width;
  // std::cout << "cvMat (h,w):  " << rows << "," << cols << std::endl;

  // RECORD VIDEO
  // std::cout << "11111" << std::endl;
  // std::cout << "Img channels: " << frame_depth.channels() << std::endl;
  // std::cout << "Img depth: " << frame_depth.depth() << std::endl;
  // std::cout << "Img type: " << frame_depth.type() << std::endl;

  // frame_depth.convertTo(frame_depth, CV_8UC1); // IPL_DEPTH_8U
  // cv::cvtColor(frame_depth, frame_depth, CV_GRAY2BGR); // convert image to RGB

  // CONVERT form CV_32FC1 to CV_8UC1
  // double minVal, maxVal;
  // minMaxLoc(frame_depth, &minVal, &maxVal); //find minimum and maximum intensities
  // cv::Mat draw;
  // frame_depth.convertTo(frame_depth, CV_8UC1, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

  // out_capture.open("TestVideo.avi", CV_FOURCC('P','I','M','1'), 30, dest_rgb.size());
  // cv::Mat imgY = cv::Mat(dest_rgb.size(), CV_32FC1);
  // cv::cvtColor(frame_depth, frame_depth, CV_GRAY2BGR); // convert image to RGB
  // std::cout << "111111" << std::endl;
  // if( !frame_depth.empty() && frame_depth.data)
  // {
  //   std::cout << "Img channels: " << frame_depth.channels() << std::endl;
  //   std::cout << "Img depth: " << frame_depth.depth() << std::endl;
  //   std::cout << "Img type: " << frame_depth.type() << std::endl;

  //   out_capture.write(frame_depth);
  //   // out_capture << frame_depth; // same same
  //   std::cout << "22222" << std::endl;
  // }

  /////////////////////////////////////////////////////////////////
}

// save timeStamp file for FPS syncing
void Recorder::saveTimeStamp()
{
  // TODO: handle relative path + check Windows / UNIX compat.
  std::ofstream fout("../recordings/timeStamp.txt");
  if(fout.is_open())
  {
      std::cout << "recording lasted " << ((timeStamps[frameID-1]-timeStamps[0])/1000.0) << " sec(s), writing timeStamp data..." << std::endl;
      fout << "# Elapsed time in ms # \n";
      for(int i = 0; i<frameID; i++)
      {
        fout << (float)timeStamps[i];
        fout << '\n';
      }
  }
  else
  {
    std::cout << "File could not be opened." << std::endl;
  }
}
