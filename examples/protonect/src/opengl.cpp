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

#include <libfreenect2/opengl.h>
#include <libfreenect2/threading.h>

#include <iostream>

namespace libfreenect2
{

OpenGLContext::OpenGLContext(GLFWwindow *ctx)
{
  //std::cerr << "[OpenGLContext] ctor " << ctx << "!" << std::endl;

  glfw_ctx = ctx;
  glew_ctx = new GLEWContext();

  ChangeCurrentOpenGLContext change_ctx(*this);

  glewExperimental = GL_TRUE;
  GLenum r = glewInit();

  if(r != GLEW_OK)
  {
    std::cerr << "[OpenGLContext] failed to initialize glew for the current context!" << std::endl;
  }
}

OpenGLContext::~OpenGLContext()
{
  delete glew_ctx;
  glfwDestroyWindow(glfw_ctx);
}

static thread_local const OpenGLContext* current_ctx = 0;

void OpenGLContext::makeCurrent() const
{
  //std::cerr << "[OpenGLContext::makeCurrent] making " << glfw_ctx << " current!" << std::endl;
  glfwMakeContextCurrent(glfw_ctx);
  current_ctx = this;
}


void OpenGLContext::detachCurrent()
{
  //std::cerr << "[OpenGLContext::makeCurrent] detaching current!" << std::endl;
  glfwMakeContextCurrent(0);
  current_ctx = 0;
}

const OpenGLContext* OpenGLContext::current()
{
  return current_ctx;
}


ChangeCurrentOpenGLContext::ChangeCurrentOpenGLContext(const OpenGLContext &new_context)
{
  last_ctx = OpenGLContext::current();
  new_context.makeCurrent();
}

ChangeCurrentOpenGLContext::~ChangeCurrentOpenGLContext()
{
  //std::cerr << "[ChangeCurrentOpenGLContext] restoring context!" << std::endl;
  if(last_ctx != 0)
  {
    last_ctx->makeCurrent();
  }
  else
  {
    OpenGLContext::detachCurrent();
  }
}

} /* namespace libfreenect2 */

GLEWContext *glewGetContext()
{
  const libfreenect2::OpenGLContext *ctx = libfreenect2::OpenGLContext::current();

  return ctx != 0 ? ctx->glew_ctx : 0;
}
