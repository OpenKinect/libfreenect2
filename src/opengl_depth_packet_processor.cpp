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

/** @file opengl_depth_packet_processor.cpp Depth packet processor implementation using OpenGL. */

#include <libfreenect2/depth_packet_processor.h>
#include <libfreenect2/resource.h>
#include <libfreenect2/protocol/response.h>
#include <libfreenect2/logging.h>
#include "flextGL.h"
#include <GLFW/glfw3.h>

#include <fstream>
#include <string>
#include <map>
#include <cstdlib>

#include <stdint.h>

#define CHECKGL() do { \
for (GLenum glerror = glGetError(); glerror != GL_NO_ERROR; glerror = glGetError()) \
  LOG_ERROR << "line " << __LINE__ << ": GL error " << glerror; \
} while(0)

namespace libfreenect2
{

struct ChangeCurrentOpenGLContext
{
  GLFWwindow *last_ctx;

  ChangeCurrentOpenGLContext(GLFWwindow *new_context);
  ~ChangeCurrentOpenGLContext();
};

ChangeCurrentOpenGLContext::ChangeCurrentOpenGLContext(GLFWwindow *new_context)
{
  last_ctx = glfwGetCurrentContext();
  glfwMakeContextCurrent(new_context);
}

ChangeCurrentOpenGLContext::~ChangeCurrentOpenGLContext()
{
  //LOG_INFO << "restoring context!";
  if(last_ctx != 0)
  {
    glfwMakeContextCurrent(last_ctx);
  }
  else
  {
    glfwMakeContextCurrent(0);
  }
}

class WithOpenGLBindings
{
private:
  OpenGLBindings *bindings;
protected:
  WithOpenGLBindings() : bindings(0) {}
  virtual ~WithOpenGLBindings() {}
  
  virtual void onOpenGLBindingsChanged(OpenGLBindings *b) { }
public:
  void gl(OpenGLBindings *bindings)
  {
    this->bindings = bindings;
    onOpenGLBindingsChanged(this->bindings);
  }
  
  OpenGLBindings *gl()
  {
    return bindings;
  }
};

std::string loadShaderSource(const std::string& filename)
{
  const unsigned char* data;
  size_t length = 0;

  if(!loadResource(filename, &data, &length))
  {
    LOG_ERROR << "failed to load shader source!";
    return "";
  }

  return std::string(reinterpret_cast<const char*>(data), length);
}

struct ShaderProgram : public WithOpenGLBindings
{
  typedef std::map<std::string, int> FragDataMap;
  FragDataMap frag_data_map_;
  GLuint program, vertex_shader, fragment_shader;

  char error_buffer[2048];

  std::string defines;
  bool is_mesa_checked;

  ShaderProgram() :
    program(0),
    vertex_shader(0),
    fragment_shader(0),
    is_mesa_checked(false)
  {
  }

  void checkMesaBug()
  {
    if (is_mesa_checked)
      return;
    is_mesa_checked = true;
    std::string ren((const char*)glGetString(GL_RENDERER));
    std::string ver((const char*)glGetString(GL_VERSION));
    if (ren.find("Mesa DRI Intel") == 0)
    {
      size_t mesa_pos = ver.rfind("Mesa ");
      if (mesa_pos != std::string::npos)
      {
        double mesa_ver = atof(ver.substr(mesa_pos + 5).c_str());
        if (mesa_ver < 10.3)
        {
          defines += "#define MESA_BUGGY_BOOL_CMP\n";
          LOG_WARNING << "Working around buggy boolean instructions in your Mesa driver. Mesa DRI 10.3+ is recommended.";
        }
      }
    }
  }

  void setVertexShader(const std::string& src)
  {
    checkMesaBug();
    const GLchar *sources[] = {"#version 140\n", defines.c_str(), src.c_str()};
    vertex_shader = gl()->glCreateShader(GL_VERTEX_SHADER);
    gl()->glShaderSource(vertex_shader, 3, sources, NULL);
    CHECKGL();
  }

  void setFragmentShader(const std::string& src)
  {
    checkMesaBug();
    const GLchar *sources[] = {"#version 140\n", defines.c_str(), src.c_str()};
    fragment_shader = gl()->glCreateShader(GL_FRAGMENT_SHADER);
    gl()->glShaderSource(fragment_shader, 3, sources, NULL);
    CHECKGL();
  }

  void bindFragDataLocation(const std::string &name, int output)
  {
    frag_data_map_[name] = output;
  }

  void build()
  {
    GLint status;

    gl()->glCompileShader(vertex_shader);
    gl()->glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &status);

    if(status != GL_TRUE)
    {
      gl()->glGetShaderInfoLog(vertex_shader, sizeof(error_buffer), NULL, error_buffer);

      LOG_ERROR << "failed to compile vertex shader!" << std::endl << error_buffer;
    }

    gl()->glCompileShader(fragment_shader);

    gl()->glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &status);
    if(status != GL_TRUE)
    {
      gl()->glGetShaderInfoLog(fragment_shader, sizeof(error_buffer), NULL, error_buffer);

      LOG_ERROR << "failed to compile fragment shader!" << std::endl << error_buffer;
    }

    program = gl()->glCreateProgram();
    gl()->glAttachShader(program, vertex_shader);
    gl()->glAttachShader(program, fragment_shader);

    for(FragDataMap::iterator it = frag_data_map_.begin(); it != frag_data_map_.end(); ++it)
    {
      gl()->glBindFragDataLocation(program, it->second, it->first.c_str());
    }

    gl()->glLinkProgram(program);

    gl()->glGetProgramiv(program, GL_LINK_STATUS, &status);

    if(status != GL_TRUE)
    {
      gl()->glGetProgramInfoLog(program, sizeof(error_buffer), NULL, error_buffer);
      LOG_ERROR << "failed to link shader program!" << std::endl << error_buffer;
    }
    CHECKGL();
  }

  GLint getAttributeLocation(const std::string& name)
  {
    return gl()->glGetAttribLocation(program, name.c_str());
  }

  void setUniform(const std::string& name, GLint value)
  {
    GLint idx = gl()->glGetUniformLocation(program, name.c_str());
    if(idx == -1) return;

    gl()->glUniform1i(idx, value);
    CHECKGL();
  }

  void setUniform(const std::string& name, GLfloat value)
  {
    GLint idx = gl()->glGetUniformLocation(program, name.c_str());
    if(idx == -1) return;

    gl()->glUniform1f(idx, value);
    CHECKGL();
  }

  void setUniformVector3(const std::string& name, GLfloat value[3])
  {
    GLint idx = gl()->glGetUniformLocation(program, name.c_str());
    if(idx == -1) return;

    gl()->glUniform3fv(idx, 1, value);
    CHECKGL();
  }

  void setUniformMatrix3(const std::string& name, GLfloat value[9])
  {
    GLint idx = gl()->glGetUniformLocation(program, name.c_str());
    if(idx == -1) return;

    gl()->glUniformMatrix3fv(idx, 1, false, value);
    CHECKGL();
  }

  void use()
  {
    gl()->glUseProgram(program);
    CHECKGL();
  }
};

template<size_t TBytesPerPixel, GLenum TInternalFormat, GLenum TFormat, GLenum TType>
struct ImageFormat
{
  static const size_t BytesPerPixel = TBytesPerPixel;
  static const GLenum InternalFormat = TInternalFormat;
  static const GLenum Format = TFormat;
  static const GLenum Type = TType;
};

typedef ImageFormat<1, GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE> U8C1;
typedef ImageFormat<2, GL_R16I, GL_RED_INTEGER, GL_SHORT> S16C1;
typedef ImageFormat<2, GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT> U16C1;
typedef ImageFormat<4, GL_R32F, GL_RED, GL_FLOAT> F32C1;
typedef ImageFormat<8, GL_RG32F, GL_RG, GL_FLOAT> F32C2;
typedef ImageFormat<12, GL_RGB32F, GL_RGB, GL_FLOAT> F32C3;
typedef ImageFormat<16, GL_RGBA32F, GL_RGBA, GL_FLOAT> F32C4;

template<typename FormatT>
struct Texture : public WithOpenGLBindings
{
protected:
  size_t bytes_per_pixel, height, width;

public:
  GLuint texture;
  unsigned char *data;
  size_t size;

  Texture() : bytes_per_pixel(FormatT::BytesPerPixel), height(0), width(0), texture(0), data(0), size(0)
  {
  }

  ~Texture()
  {
    delete[] data;
  }

  void bindToUnit(GLenum unit)
  {
    gl()->glActiveTexture(unit);
    glBindTexture(GL_TEXTURE_RECTANGLE, texture);
    CHECKGL();
  }

  void allocate(size_t new_width, size_t new_height)
  {
    if (size)
      return;

    GLint max_size;
    glGetIntegerv(GL_MAX_RECTANGLE_TEXTURE_SIZE, &max_size);
    if (new_width > (size_t)max_size || new_height > (size_t)max_size)
    {
      LOG_ERROR << "GL_MAX_RECTANGLE_TEXTURE_SIZE is too small: " << max_size;
      exit(-1);
    }

    width = new_width;
    height = new_height;
    size = height * width * bytes_per_pixel;
    data = new unsigned char[size];

    glGenTextures(1, &texture);
    bindToUnit(GL_TEXTURE0);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, FormatT::InternalFormat, width, height, 0, FormatT::Format, FormatT::Type, 0);
    CHECKGL();
  }

  void upload()
  {
    bindToUnit(GL_TEXTURE0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE, /*level*/0, /*xoffset*/0, /*yoffset*/0, width, height, FormatT::Format, FormatT::Type, data);
    CHECKGL();
  }

  void download()
  {
    downloadToBuffer(data);
  }

  void downloadToBuffer(unsigned char *data)
  {
    glReadPixels(0, 0, width, height, FormatT::Format, FormatT::Type, data);
    CHECKGL();
  }

  void flipY()
  {
    flipYBuffer(data);
  }

  void flipYBuffer(unsigned char *data)
  {
    typedef unsigned char type;

    size_t linestep = width * bytes_per_pixel / sizeof(type);

    type *first_line = reinterpret_cast<type *>(data), *last_line = reinterpret_cast<type *>(data) + (height - 1) * linestep;

    for(size_t y = 0; y < height / 2; ++y)
    {
      for(size_t x = 0; x < linestep; ++x, ++first_line, ++last_line)
      {
        std::swap(*first_line, *last_line);
      }
      last_line -= 2 * linestep;
    }
  }

  Frame *downloadToNewFrame()
  {
    Frame *f = new Frame(width, height, bytes_per_pixel);
    f->format = Frame::Float;
    downloadToBuffer(f->data);
    flipYBuffer(f->data);

    return f;
  }
};

class OpenGLDepthPacketProcessorImpl : public WithOpenGLBindings, public WithPerfLogging
{
public:
  GLFWwindow *opengl_context_ptr;
  libfreenect2::DepthPacketProcessor::Config config;

  GLuint square_vbo, square_vao, stage1_framebuffer, filter1_framebuffer, stage2_framebuffer, filter2_framebuffer;
  Texture<S16C1> lut11to16;
  Texture<U16C1> p0table[3];
  Texture<F32C1> x_table, z_table;

  Texture<U16C1> input_data;

  Texture<F32C4> stage1_debug;
  Texture<F32C4> stage1_data[3];
  Texture<F32C1> stage1_infrared;

  Texture<F32C4> filter1_data[2];
  Texture<U8C1> filter1_max_edge_test;
  Texture<F32C4> filter1_debug;

  Texture<F32C4> stage2_debug;

  Texture<F32C1> stage2_depth;
  Texture<F32C2> stage2_depth_and_ir_sum;

  Texture<F32C4> filter2_debug;
  Texture<F32C1> filter2_depth;

  ShaderProgram stage1, filter1, stage2, filter2, debug;

  DepthPacketProcessor::Parameters params;
  bool params_need_update;

  bool do_debug;

  struct Vertex
  {
    float x, y;
    float u, v;
  };

  OpenGLDepthPacketProcessorImpl(GLFWwindow *new_opengl_context_ptr, bool debug) :
    opengl_context_ptr(new_opengl_context_ptr),
    square_vbo(0),
    square_vao(0),
    stage1_framebuffer(0),
    filter1_framebuffer(0),
    stage2_framebuffer(0),
    filter2_framebuffer(0),
    params_need_update(true),
    do_debug(debug)
  {
  }

  virtual ~OpenGLDepthPacketProcessorImpl()
  {
    if(gl() != 0)
    {
      delete gl();
      gl(0);
    }
    glfwDestroyWindow(opengl_context_ptr);
    opengl_context_ptr = 0;
  }
  
  virtual void onOpenGLBindingsChanged(OpenGLBindings *b) 
  {
    lut11to16.gl(b);
    p0table[0].gl(b);
    p0table[1].gl(b);
    p0table[2].gl(b);
    x_table.gl(b);
    z_table.gl(b);

    input_data.gl(b);

    stage1_debug.gl(b);
    stage1_data[0].gl(b);
    stage1_data[1].gl(b);
    stage1_data[2].gl(b);
    stage1_infrared.gl(b);

    filter1_data[0].gl(b);
    filter1_data[1].gl(b);
    filter1_max_edge_test.gl(b);
    filter1_debug.gl(b);

    stage2_debug.gl(b);

    stage2_depth.gl(b);
    stage2_depth_and_ir_sum.gl(b);

    filter2_debug.gl(b);
    filter2_depth.gl(b);
 
    stage1.gl(b);
    filter1.gl(b);
    stage2.gl(b);
    filter2.gl(b);
    debug.gl(b);
  }

  static void glfwErrorCallback(int error, const char* description)
  {
    LOG_ERROR << "GLFW error " << error << " " << description;
  }

  void checkFBO(GLenum target)
  {
    GLenum status = gl()->glCheckFramebufferStatus(target);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
      LOG_ERROR << "incomplete FBO " << status;
      exit(-1);
    }
    CHECKGL();
  }

  void initialize()
  {
    ChangeCurrentOpenGLContext ctx(opengl_context_ptr);
    
    int major = glfwGetWindowAttrib(opengl_context_ptr, GLFW_CONTEXT_VERSION_MAJOR);
    int minor = glfwGetWindowAttrib(opengl_context_ptr, GLFW_CONTEXT_VERSION_MINOR);

    if (major * 10 + minor < 31) {
        LOG_ERROR << "OpenGL version 3.1 not supported.";
        LOG_ERROR << "Your version is " << major << "." << minor;
        LOG_ERROR << "Try updating your graphics driver.";
        exit(-1);
    }

    OpenGLBindings *b = new OpenGLBindings();
    flextInit(b);
    gl(b);

    input_data.allocate(352, 424 * 9);

    for(int i = 0; i < 3; ++i)
      stage1_data[i].allocate(512, 424);

    if(do_debug) stage1_debug.allocate(512, 424);
    stage1_infrared.allocate(512, 424);

    for(int i = 0; i < 2; ++i)
      filter1_data[i].allocate(512, 424);

    filter1_max_edge_test.allocate(512, 424);
    if(do_debug) filter1_debug.allocate(512, 424);

    if(do_debug) stage2_debug.allocate(512, 424);
    stage2_depth.allocate(512, 424);
    stage2_depth_and_ir_sum.allocate(512, 424);

    if(do_debug) filter2_debug.allocate(512, 424);
    filter2_depth.allocate(512, 424);

    stage1.setVertexShader(loadShaderSource("default.vs"));
    stage1.setFragmentShader(loadShaderSource("stage1.fs"));
    stage1.bindFragDataLocation("Debug", 0);
    stage1.bindFragDataLocation("A", 1);
    stage1.bindFragDataLocation("B", 2);
    stage1.bindFragDataLocation("Norm", 3);
    stage1.bindFragDataLocation("Infrared", 4);
    stage1.build();

    filter1.setVertexShader(loadShaderSource("default.vs"));
    filter1.setFragmentShader(loadShaderSource("filter1.fs"));
    filter1.bindFragDataLocation("Debug", 0);
    filter1.bindFragDataLocation("FilterA", 1);
    filter1.bindFragDataLocation("FilterB", 2);
    filter1.bindFragDataLocation("MaxEdgeTest", 3);
    filter1.build();

    stage2.setVertexShader(loadShaderSource("default.vs"));
    stage2.setFragmentShader(loadShaderSource("stage2.fs"));
    stage2.bindFragDataLocation("Debug", 0);
    stage2.bindFragDataLocation("Depth", 1);
    stage2.bindFragDataLocation("DepthAndIrSum", 2);
    stage2.build();

    filter2.setVertexShader(loadShaderSource("default.vs"));
    filter2.setFragmentShader(loadShaderSource("filter2.fs"));
    filter2.bindFragDataLocation("Debug", 0);
    filter2.bindFragDataLocation("FilterDepth", 1);
    filter2.build();

    if(do_debug)
    {
      debug.setVertexShader(loadShaderSource("default.vs"));
      debug.setFragmentShader(loadShaderSource("debug.fs"));
      debug.bindFragDataLocation("Debug", 0);
      debug.build();
    }

    GLenum debug_attachment = do_debug ? GL_COLOR_ATTACHMENT0 : GL_NONE;

    gl()->glGenFramebuffers(1, &stage1_framebuffer);
    gl()->glBindFramebuffer(GL_FRAMEBUFFER, stage1_framebuffer);

    const GLenum stage1_buffers[] = { debug_attachment, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4 };
    gl()->glDrawBuffers(5, stage1_buffers);
    glReadBuffer(GL_COLOR_ATTACHMENT4);

    if(do_debug) gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, stage1_debug.texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, stage1_data[0].texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_RECTANGLE, stage1_data[1].texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_RECTANGLE, stage1_data[2].texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_RECTANGLE, stage1_infrared.texture, 0);
    checkFBO(GL_FRAMEBUFFER);

    gl()->glGenFramebuffers(1, &filter1_framebuffer);
    gl()->glBindFramebuffer(GL_FRAMEBUFFER, filter1_framebuffer);

    const GLenum filter1_buffers[] = { debug_attachment, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
    gl()->glDrawBuffers(4, filter1_buffers);
    glReadBuffer(GL_NONE);

    if(do_debug) gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, filter1_debug.texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, filter1_data[0].texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_RECTANGLE, filter1_data[1].texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_RECTANGLE, filter1_max_edge_test.texture, 0);
    checkFBO(GL_FRAMEBUFFER);

    gl()->glGenFramebuffers(1, &stage2_framebuffer);
    gl()->glBindFramebuffer(GL_FRAMEBUFFER, stage2_framebuffer);

    const GLenum stage2_buffers[] = { debug_attachment, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    gl()->glDrawBuffers(3, stage2_buffers);
    glReadBuffer(GL_COLOR_ATTACHMENT1);

    if(do_debug) gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, stage2_debug.texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, stage2_depth.texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_RECTANGLE, stage2_depth_and_ir_sum.texture, 0);
    checkFBO(GL_FRAMEBUFFER);

    gl()->glGenFramebuffers(1, &filter2_framebuffer);
    gl()->glBindFramebuffer(GL_FRAMEBUFFER, filter2_framebuffer);

    const GLenum filter2_buffers[] = { debug_attachment, GL_COLOR_ATTACHMENT1 };
    gl()->glDrawBuffers(2, filter2_buffers);
    glReadBuffer(GL_COLOR_ATTACHMENT1);

    if(do_debug) gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, filter2_debug.texture, 0);
    gl()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, filter2_depth.texture, 0);
    checkFBO(GL_FRAMEBUFFER);

    Vertex bl = {-1.0f, -1.0f, 0.0f, 0.0f }, br = { 1.0f, -1.0f, 512.0f, 0.0f }, tl = {-1.0f, 1.0f, 0.0f, 424.0f }, tr = { 1.0f, 1.0f, 512.0f, 424.0f };
    Vertex vertices[] = {
        bl, tl, tr, tr, br, bl
    };
    gl()->glGenBuffers(1, &square_vbo);
    gl()->glGenVertexArrays(1, &square_vao);

    gl()->glBindVertexArray(square_vao);
    gl()->glBindBuffer(GL_ARRAY_BUFFER, square_vbo);
    gl()->glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    GLint position_attr = stage1.getAttributeLocation("InputPosition");
    gl()->glVertexAttribPointer(position_attr, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)0);
    gl()->glEnableVertexAttribArray(position_attr);

    GLint texcoord_attr = stage1.getAttributeLocation("InputTexCoord");
    gl()->glVertexAttribPointer(texcoord_attr, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)(2 * sizeof(float)));
    gl()->glEnableVertexAttribArray(texcoord_attr);
    CHECKGL();
  }

  void deinitialize()
  {
  }

  void updateShaderParametersForProgram(ShaderProgram &program)
  {
    if(!params_need_update) return;

    program.setUniform("Params.ab_multiplier", params.ab_multiplier);
    program.setUniformVector3("Params.ab_multiplier_per_frq", params.ab_multiplier_per_frq);
    program.setUniform("Params.ab_output_multiplier", params.ab_output_multiplier);

    program.setUniformVector3("Params.phase_in_rad", params.phase_in_rad);

    program.setUniform("Params.joint_bilateral_ab_threshold", params.joint_bilateral_ab_threshold);
    program.setUniform("Params.joint_bilateral_max_edge", params.joint_bilateral_max_edge);
    program.setUniform("Params.joint_bilateral_exp", params.joint_bilateral_exp);
    program.setUniformMatrix3("Params.gaussian_kernel", params.gaussian_kernel);

    program.setUniform("Params.phase_offset", params.phase_offset);
    program.setUniform("Params.unambigious_dist", params.unambigious_dist);
    program.setUniform("Params.individual_ab_threshold", params.individual_ab_threshold);
    program.setUniform("Params.ab_threshold", params.ab_threshold);
    program.setUniform("Params.ab_confidence_slope", params.ab_confidence_slope);
    program.setUniform("Params.ab_confidence_offset", params.ab_confidence_offset);
    program.setUniform("Params.min_dealias_confidence", params.min_dealias_confidence);
    program.setUniform("Params.max_dealias_confidence", params.max_dealias_confidence);

    program.setUniform("Params.edge_ab_avg_min_value", params.edge_ab_avg_min_value);
    program.setUniform("Params.edge_ab_std_dev_threshold", params.edge_ab_std_dev_threshold);
    program.setUniform("Params.edge_close_delta_threshold", params.edge_close_delta_threshold);
    program.setUniform("Params.edge_far_delta_threshold", params.edge_far_delta_threshold);
    program.setUniform("Params.edge_max_delta_threshold", params.edge_max_delta_threshold);
    program.setUniform("Params.edge_avg_delta_threshold", params.edge_avg_delta_threshold);
    program.setUniform("Params.max_edge_count", params.max_edge_count);

    program.setUniform("Params.min_depth", params.min_depth);
    program.setUniform("Params.max_depth", params.max_depth);
  }

  void run(Frame **ir, Frame **depth)
  {
    // data processing 1
    glViewport(0, 0, 512, 424);
    stage1.use();
    updateShaderParametersForProgram(stage1);

    p0table[0].bindToUnit(GL_TEXTURE0);
    stage1.setUniform("P0Table0", 0);
    p0table[1].bindToUnit(GL_TEXTURE1);
    stage1.setUniform("P0Table1", 1);
    p0table[2].bindToUnit(GL_TEXTURE2);
    stage1.setUniform("P0Table2", 2);
    lut11to16.bindToUnit(GL_TEXTURE3);
    stage1.setUniform("Lut11to16", 3);
    input_data.bindToUnit(GL_TEXTURE4);
    stage1.setUniform("Data", 4);
    z_table.bindToUnit(GL_TEXTURE5);
    stage1.setUniform("ZTable", 5);

    gl()->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, stage1_framebuffer);
    glClear(GL_COLOR_BUFFER_BIT);

    gl()->glBindVertexArray(square_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    CHECKGL();

    if(ir != 0)
    {
      gl()->glBindFramebuffer(GL_READ_FRAMEBUFFER, stage1_framebuffer);
      glReadBuffer(GL_COLOR_ATTACHMENT4);
      *ir = stage1_infrared.downloadToNewFrame();
    }

    if(config.EnableBilateralFilter)
    {
      // bilateral filter
      gl()->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, filter1_framebuffer);
      glClear(GL_COLOR_BUFFER_BIT);

      filter1.use();
      updateShaderParametersForProgram(filter1);

      stage1_data[0].bindToUnit(GL_TEXTURE0);
      filter1.setUniform("A", 0);
      stage1_data[1].bindToUnit(GL_TEXTURE1);
      filter1.setUniform("B", 1);
      stage1_data[2].bindToUnit(GL_TEXTURE2);
      filter1.setUniform("Norm", 2);

      gl()->glBindVertexArray(square_vao);
      glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    // data processing 2
    gl()->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, stage2_framebuffer);
    glClear(GL_COLOR_BUFFER_BIT);

    stage2.use();
    updateShaderParametersForProgram(stage2);
    CHECKGL();

    if(config.EnableBilateralFilter)
    {
      filter1_data[0].bindToUnit(GL_TEXTURE0);
      filter1_data[1].bindToUnit(GL_TEXTURE1);
    }
    else
    {
      stage1_data[0].bindToUnit(GL_TEXTURE0);
      stage1_data[1].bindToUnit(GL_TEXTURE1);
    }
    stage2.setUniform("A", 0);
    stage2.setUniform("B", 1);
    x_table.bindToUnit(GL_TEXTURE2);
    stage2.setUniform("XTable", 2);
    z_table.bindToUnit(GL_TEXTURE3);
    stage2.setUniform("ZTable", 3);

    gl()->glBindVertexArray(square_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    CHECKGL();

    if(config.EnableEdgeAwareFilter)
    {
      // edge aware filter
      gl()->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, filter2_framebuffer);
      glClear(GL_COLOR_BUFFER_BIT);

      filter2.use();
      updateShaderParametersForProgram(filter2);

      stage2_depth_and_ir_sum.bindToUnit(GL_TEXTURE0);
      filter2.setUniform("DepthAndIrSum", 0);
      filter1_max_edge_test.bindToUnit(GL_TEXTURE1);
      filter2.setUniform("MaxEdgeTest", 1);

      gl()->glBindVertexArray(square_vao);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      if(depth != 0)
      {
        gl()->glBindFramebuffer(GL_READ_FRAMEBUFFER, filter2_framebuffer);
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        *depth = filter2_depth.downloadToNewFrame();
      }
    }
    else
    {
      if(depth != 0)
      {
        gl()->glBindFramebuffer(GL_READ_FRAMEBUFFER, stage2_framebuffer);
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        *depth = stage2_depth.downloadToNewFrame();
      }
    }
    CHECKGL();

    if(do_debug)
    {
      // debug drawing
      gl()->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
      glClear(GL_COLOR_BUFFER_BIT);

      gl()->glBindVertexArray(square_vao);

      debug.use();
      stage2_debug.bindToUnit(GL_TEXTURE0);
      debug.setUniform("Debug", 0);

      glDrawArrays(GL_TRIANGLES, 0, 6);

      glViewport(512, 0, 512, 424);
      filter2_debug.bindToUnit(GL_TEXTURE0);
      debug.setUniform("Debug", 0);

      glDrawArrays(GL_TRIANGLES, 0, 6);

      glViewport(0, 424, 512, 424);
      stage1_debug.bindToUnit(GL_TEXTURE0);
      debug.setUniform("Debug", 0);

      glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    CHECKGL();

    params_need_update = false;
  }
};

OpenGLDepthPacketProcessor::OpenGLDepthPacketProcessor(void *parent_opengl_context_ptr, bool debug)
{
  GLFWwindow* parent_window = (GLFWwindow *)parent_opengl_context_ptr;

  GLFWerrorfun prev_func = glfwSetErrorCallback(&OpenGLDepthPacketProcessorImpl::glfwErrorCallback);
  if (prev_func)
    glfwSetErrorCallback(prev_func);

  // init glfw - if already initialized nothing happens
  if (glfwInit() == GL_FALSE)
  {
      LOG_ERROR << "Failed to initialize GLFW.";
      exit(-1);
  }
  
  // setup context
  glfwDefaultWindowHints();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
#ifdef __APPLE__
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#else
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
#endif
  glfwWindowHint(GLFW_VISIBLE, debug ? GL_TRUE : GL_FALSE);

  GLFWwindow* window = glfwCreateWindow(1024, 848, "OpenGLDepthPacketProcessor", 0, parent_window);

  if (window == NULL)
  {
      LOG_ERROR << "Failed to create opengl window.";
      exit(-1);
  }

  impl_ = new OpenGLDepthPacketProcessorImpl(window, debug);
  impl_->initialize();
}

OpenGLDepthPacketProcessor::~OpenGLDepthPacketProcessor()
{
  delete impl_;
}


void OpenGLDepthPacketProcessor::setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config)
{
  DepthPacketProcessor::setConfiguration(config);
  impl_->config = config;

  impl_->params.min_depth = impl_->config.MinDepth * 1000.0f;
  impl_->params.max_depth = impl_->config.MaxDepth * 1000.0f;

  impl_->params_need_update = true;
}

void OpenGLDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length)
{
  ChangeCurrentOpenGLContext ctx(impl_->opengl_context_ptr);

  size_t n = 512 * 424;
  libfreenect2::protocol::P0TablesResponse* p0table = (libfreenect2::protocol::P0TablesResponse*)buffer;

  impl_->p0table[0].allocate(512, 424);
  std::copy(reinterpret_cast<unsigned char*>(p0table->p0table0), reinterpret_cast<unsigned char*>(p0table->p0table0 + n), impl_->p0table[0].data);
  impl_->p0table[0].flipY();
  impl_->p0table[0].upload();

  impl_->p0table[1].allocate(512, 424);
  std::copy(reinterpret_cast<unsigned char*>(p0table->p0table1), reinterpret_cast<unsigned char*>(p0table->p0table1 + n), impl_->p0table[1].data);
  impl_->p0table[1].flipY();
  impl_->p0table[1].upload();

  impl_->p0table[2].allocate(512, 424);
  std::copy(reinterpret_cast<unsigned char*>(p0table->p0table2), reinterpret_cast<unsigned char*>(p0table->p0table2 + n), impl_->p0table[2].data);
  impl_->p0table[2].flipY();
  impl_->p0table[2].upload();

}

void OpenGLDepthPacketProcessor::loadXZTables(const float *xtable, const float *ztable)
{
  ChangeCurrentOpenGLContext ctx(impl_->opengl_context_ptr);

  impl_->x_table.allocate(512, 424);
  std::copy(xtable, xtable + TABLE_SIZE, (float *)impl_->x_table.data);
  impl_->x_table.upload();

  impl_->z_table.allocate(512, 424);
  std::copy(ztable, ztable + TABLE_SIZE, (float *)impl_->z_table.data);
  impl_->z_table.upload();
}

void OpenGLDepthPacketProcessor::loadLookupTable(const short *lut)
{
  ChangeCurrentOpenGLContext ctx(impl_->opengl_context_ptr);

  impl_->lut11to16.allocate(2048, 1);
  std::copy(lut, lut + LUT_SIZE, (short *)impl_->lut11to16.data);
  impl_->lut11to16.upload();
}

void OpenGLDepthPacketProcessor::process(const DepthPacket &packet)
{
  if (!listener_)
    return;
  Frame *ir = 0, *depth = 0;

  impl_->startTiming();

  glfwMakeContextCurrent(impl_->opengl_context_ptr);

  std::copy(packet.buffer, packet.buffer + packet.buffer_length/10*9, impl_->input_data.data);
  impl_->input_data.upload();
  impl_->run(&ir, &depth);

  if(impl_->do_debug) glfwSwapBuffers(impl_->opengl_context_ptr);

  impl_->stopTiming(LOG_INFO);

  ir->timestamp = packet.timestamp;
  depth->timestamp = packet.timestamp;
  ir->sequence = packet.sequence;
  depth->sequence = packet.sequence;

  if(!listener_->onNewFrame(Frame::Ir, ir))
    delete ir;

  if(!listener_->onNewFrame(Frame::Depth, depth))
    delete depth;
}

} /* namespace libfreenect2 */
