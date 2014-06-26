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

#ifndef OPENGL_H_
#define OPENGL_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace libfreenect2
{

struct OpenGLContext
{
  GLFWwindow *glfw_ctx;
  GLEWContext *glew_ctx;

  OpenGLContext(GLFWwindow *ctx);
  ~OpenGLContext();

  void makeCurrent() const;

  static void detachCurrent();
  static const OpenGLContext* current();
};


struct ChangeCurrentOpenGLContext
{
  const OpenGLContext *last_ctx;

  ChangeCurrentOpenGLContext(const OpenGLContext &new_context);
  ~ChangeCurrentOpenGLContext();
};

} /* namespace libfreenect2 */

GLEWContext *glewGetContext();

#endif /* OPENGL_H_ */
