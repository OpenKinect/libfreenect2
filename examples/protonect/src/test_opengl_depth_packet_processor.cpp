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

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

#include <libfreenect2/async_packet_processor.h>
#include <libfreenect2/depth_packet_processor.h>
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
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  GLFWwindow* window = glfwCreateWindow(1200, 600, "OpenGL", 0, 0); // Windowed
  glfwMakeContextCurrent(window);

  glewExperimental = GL_TRUE;
  glewInit();
  glfwMakeContextCurrent(0);

  libfreenect2::OpenGLDepthPacketProcessor processor(window, (binpath + "../src/shader/").c_str());
  processor.loadP0TablesFromFiles((binpath + "../p00.bin").c_str(), (binpath + "../p01.bin").c_str(), (binpath + "../p02.bin").c_str());
  processor.load11To16LutFromFile((binpath + "../11to16.bin").c_str());
  processor.loadXTableFromFile((binpath + "../xTable.bin").c_str());
  processor.loadZTableFromFile((binpath + "../zTable.bin").c_str());
  glfwMakeContextCurrent(0);

  libfreenect2::AsyncPacketProcessor<libfreenect2::DepthPacket, libfreenect2::DepthPacketProcessor> async(&processor);

  libfreenect2::DepthPacket p;
  p.buffer_length = 352*424*10*2;
  p.buffer = new unsigned char[p.buffer_length];

  loadBufferFromFile(binpath + "../rawir/rawir_4599.bin", p.buffer, p.buffer_length);


  while(!glfwWindowShouldClose(window))
  {
    glfwMakeContextCurrent(0);

    if(async.ready())
      async.process(p);
    //processor.process(p);

    glfwMakeContextCurrent(window);
    glfwSwapBuffers(window);
    //glfwSwapBuffers(window_background);
    glfwPollEvents();
  }

  glfwDestroyWindow(window);

  return 0;
}
