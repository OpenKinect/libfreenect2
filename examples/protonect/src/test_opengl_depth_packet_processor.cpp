/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 individual OpenKinect contributors. See the CONTRIB file
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

/** @file test_opengl_depth_packet_processor.cpp Test program for the OpenGL depth packet processor. */

#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

#include <libfreenect2/async_packet_processor.h>
#include <libfreenect2/depth_packet_processor.h>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/resource.h>

void loadBufferFromFile(const std::string& filename, unsigned char *buffer, size_t n)
{
  std::ifstream in(filename.c_str());

  in.read(reinterpret_cast<char*>(buffer), n);
  if(in.gcount() != n) throw std::exception();

  in.close();
}

int main(int argc, char **argv) {
  std::string program_path(argv[0]);
  size_t executable_name_idx = program_path.rfind("test_opengl");
  std::string binpath = "./";

  if(executable_name_idx != std::string::npos)
  {
    binpath = program_path.substr(0, executable_name_idx);
  }

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  GLFWwindow* window = glfwCreateWindow(1200, 600, "OpenGL", 0, 0); // Windowed

  libfreenect2::SyncMultiFrameListener fl(libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
  libfreenect2::FrameMap frames;

  libfreenect2::DepthPacketProcessor::Config cfg;
  cfg.EnableBilateralFilter = false;
  cfg.EnableEdgeAwareFilter = false;

  libfreenect2::OpenGLDepthPacketProcessor processor(window, true);
  processor.setConfiguration(cfg);
  processor.setFrameListener(&fl);
  processor.loadP0TablesFromFiles((binpath + "../p00.bin").c_str(), (binpath + "../p01.bin").c_str(), (binpath + "../p02.bin").c_str());
  processor.load11To16LutFromFile("");
  processor.loadXTableFromFile("");
  processor.loadZTableFromFile("");

  libfreenect2::CpuDepthPacketProcessor ref_processor;
  ref_processor.setConfiguration(cfg);
  ref_processor.setFrameListener(&fl);
  ref_processor.loadP0TablesFromFiles((binpath + "../p00.bin").c_str(), (binpath + "../p01.bin").c_str(), (binpath + "../p02.bin").c_str());
  ref_processor.load11To16LutFromFile("");
  ref_processor.loadXTableFromFile("");
  ref_processor.loadZTableFromFile("");

  libfreenect2::AsyncPacketProcessor<libfreenect2::DepthPacket> async(&processor);

  libfreenect2::DepthPacket p;
  p.buffer_length = 352*424*10*2;
  p.buffer = new unsigned char[p.buffer_length];

  loadBufferFromFile(binpath + "../rawir/rawir_4599.bin", p.buffer, p.buffer_length);

  libfreenect2::Frame *ir, *depth;
  cv::Mat cpu_ir, cpu_depth, ogl_ir, ogl_depth;

  ref_processor.process(p);
  fl.waitForNewFrame(frames);

  ir = frames[libfreenect2::Frame::Ir];
  depth = frames[libfreenect2::Frame::Depth];
  cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(cpu_ir);
  cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(cpu_depth);

  fl.release(frames);

  processor.process(p);
  fl.waitForNewFrame(frames);

  ir = frames[libfreenect2::Frame::Ir];
  depth = frames[libfreenect2::Frame::Depth];
  cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(ogl_ir);
  cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(ogl_depth);

  fl.release(frames);

  cv::Mat diff_ir = cv::abs(cpu_ir - ogl_ir);
  cv::Mat diff_depth = cv::abs(cpu_depth - ogl_depth);

  cv::imshow("cpu_ir", cpu_ir / 65535.0f);
  cv::imshow("cpu_depth", cpu_depth / 4500.0f);

  cv::imshow("diff_ir", diff_ir);
  cv::imshow("diff_depth", diff_depth);
  cv::waitKey(0);

  double mi, ma;
  cv::minMaxIdx(diff_depth, &mi, &ma);
  std::cout << "depth difference min: " << mi << " max: " << ma << std::endl;

  while(!glfwWindowShouldClose(window))
  {
    if(async.ready())
      async.process(p);
    //processor.process(p);

    glfwMakeContextCurrent(window);
    glfwSwapBuffers(window);
    //glfwSwapBuffers(window_background);
    glfwPollEvents();
  }

  return 0;
}
