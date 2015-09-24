#include "flextGL.h"
#include "GLFW/glfw3.h"

#ifdef __cplusplus
extern "C" {
#endif

void flextInit(OpenGLBindings *bindings)
{
    /* --- Function pointer loading --- */


    /* GL_VERSION_1_2 */

    bindings->glDrawRangeElements = (PFNGLDRAWRANGEELEMENTS_PROC*)glfwGetProcAddress("glDrawRangeElements");
    bindings->glTexImage3D = (PFNGLTEXIMAGE3D_PROC*)glfwGetProcAddress("glTexImage3D");
    bindings->glTexSubImage3D = (PFNGLTEXSUBIMAGE3D_PROC*)glfwGetProcAddress("glTexSubImage3D");
    bindings->glCopyTexSubImage3D = (PFNGLCOPYTEXSUBIMAGE3D_PROC*)glfwGetProcAddress("glCopyTexSubImage3D");

    /* GL_VERSION_1_3 */

    bindings->glActiveTexture = (PFNGLACTIVETEXTURE_PROC*)glfwGetProcAddress("glActiveTexture");
    bindings->glSampleCoverage = (PFNGLSAMPLECOVERAGE_PROC*)glfwGetProcAddress("glSampleCoverage");
    bindings->glCompressedTexImage3D = (PFNGLCOMPRESSEDTEXIMAGE3D_PROC*)glfwGetProcAddress("glCompressedTexImage3D");
    bindings->glCompressedTexImage2D = (PFNGLCOMPRESSEDTEXIMAGE2D_PROC*)glfwGetProcAddress("glCompressedTexImage2D");
    bindings->glCompressedTexImage1D = (PFNGLCOMPRESSEDTEXIMAGE1D_PROC*)glfwGetProcAddress("glCompressedTexImage1D");
    bindings->glCompressedTexSubImage3D = (PFNGLCOMPRESSEDTEXSUBIMAGE3D_PROC*)glfwGetProcAddress("glCompressedTexSubImage3D");
    bindings->glCompressedTexSubImage2D = (PFNGLCOMPRESSEDTEXSUBIMAGE2D_PROC*)glfwGetProcAddress("glCompressedTexSubImage2D");
    bindings->glCompressedTexSubImage1D = (PFNGLCOMPRESSEDTEXSUBIMAGE1D_PROC*)glfwGetProcAddress("glCompressedTexSubImage1D");
    bindings->glGetCompressedTexImage = (PFNGLGETCOMPRESSEDTEXIMAGE_PROC*)glfwGetProcAddress("glGetCompressedTexImage");

    /* GL_VERSION_1_4 */

    bindings->glBlendFuncSeparate = (PFNGLBLENDFUNCSEPARATE_PROC*)glfwGetProcAddress("glBlendFuncSeparate");
    bindings->glMultiDrawArrays = (PFNGLMULTIDRAWARRAYS_PROC*)glfwGetProcAddress("glMultiDrawArrays");
    bindings->glMultiDrawElements = (PFNGLMULTIDRAWELEMENTS_PROC*)glfwGetProcAddress("glMultiDrawElements");
    bindings->glPointParameterf = (PFNGLPOINTPARAMETERF_PROC*)glfwGetProcAddress("glPointParameterf");
    bindings->glPointParameterfv = (PFNGLPOINTPARAMETERFV_PROC*)glfwGetProcAddress("glPointParameterfv");
    bindings->glPointParameteri = (PFNGLPOINTPARAMETERI_PROC*)glfwGetProcAddress("glPointParameteri");
    bindings->glPointParameteriv = (PFNGLPOINTPARAMETERIV_PROC*)glfwGetProcAddress("glPointParameteriv");
    bindings->glBlendColor = (PFNGLBLENDCOLOR_PROC*)glfwGetProcAddress("glBlendColor");
    bindings->glBlendEquation = (PFNGLBLENDEQUATION_PROC*)glfwGetProcAddress("glBlendEquation");

    /* GL_VERSION_1_5 */

    bindings->glGenQueries = (PFNGLGENQUERIES_PROC*)glfwGetProcAddress("glGenQueries");
    bindings->glDeleteQueries = (PFNGLDELETEQUERIES_PROC*)glfwGetProcAddress("glDeleteQueries");
    bindings->glIsQuery = (PFNGLISQUERY_PROC*)glfwGetProcAddress("glIsQuery");
    bindings->glBeginQuery = (PFNGLBEGINQUERY_PROC*)glfwGetProcAddress("glBeginQuery");
    bindings->glEndQuery = (PFNGLENDQUERY_PROC*)glfwGetProcAddress("glEndQuery");
    bindings->glGetQueryiv = (PFNGLGETQUERYIV_PROC*)glfwGetProcAddress("glGetQueryiv");
    bindings->glGetQueryObjectiv = (PFNGLGETQUERYOBJECTIV_PROC*)glfwGetProcAddress("glGetQueryObjectiv");
    bindings->glGetQueryObjectuiv = (PFNGLGETQUERYOBJECTUIV_PROC*)glfwGetProcAddress("glGetQueryObjectuiv");
    bindings->glBindBuffer = (PFNGLBINDBUFFER_PROC*)glfwGetProcAddress("glBindBuffer");
    bindings->glDeleteBuffers = (PFNGLDELETEBUFFERS_PROC*)glfwGetProcAddress("glDeleteBuffers");
    bindings->glGenBuffers = (PFNGLGENBUFFERS_PROC*)glfwGetProcAddress("glGenBuffers");
    bindings->glIsBuffer = (PFNGLISBUFFER_PROC*)glfwGetProcAddress("glIsBuffer");
    bindings->glBufferData = (PFNGLBUFFERDATA_PROC*)glfwGetProcAddress("glBufferData");
    bindings->glBufferSubData = (PFNGLBUFFERSUBDATA_PROC*)glfwGetProcAddress("glBufferSubData");
    bindings->glGetBufferSubData = (PFNGLGETBUFFERSUBDATA_PROC*)glfwGetProcAddress("glGetBufferSubData");
    bindings->glMapBuffer = (PFNGLMAPBUFFER_PROC*)glfwGetProcAddress("glMapBuffer");
    bindings->glUnmapBuffer = (PFNGLUNMAPBUFFER_PROC*)glfwGetProcAddress("glUnmapBuffer");
    bindings->glGetBufferParameteriv = (PFNGLGETBUFFERPARAMETERIV_PROC*)glfwGetProcAddress("glGetBufferParameteriv");
    bindings->glGetBufferPointerv = (PFNGLGETBUFFERPOINTERV_PROC*)glfwGetProcAddress("glGetBufferPointerv");

    /* GL_VERSION_2_0 */

    bindings->glBlendEquationSeparate = (PFNGLBLENDEQUATIONSEPARATE_PROC*)glfwGetProcAddress("glBlendEquationSeparate");
    bindings->glDrawBuffers = (PFNGLDRAWBUFFERS_PROC*)glfwGetProcAddress("glDrawBuffers");
    bindings->glStencilOpSeparate = (PFNGLSTENCILOPSEPARATE_PROC*)glfwGetProcAddress("glStencilOpSeparate");
    bindings->glStencilFuncSeparate = (PFNGLSTENCILFUNCSEPARATE_PROC*)glfwGetProcAddress("glStencilFuncSeparate");
    bindings->glStencilMaskSeparate = (PFNGLSTENCILMASKSEPARATE_PROC*)glfwGetProcAddress("glStencilMaskSeparate");
    bindings->glAttachShader = (PFNGLATTACHSHADER_PROC*)glfwGetProcAddress("glAttachShader");
    bindings->glBindAttribLocation = (PFNGLBINDATTRIBLOCATION_PROC*)glfwGetProcAddress("glBindAttribLocation");
    bindings->glCompileShader = (PFNGLCOMPILESHADER_PROC*)glfwGetProcAddress("glCompileShader");
    bindings->glCreateProgram = (PFNGLCREATEPROGRAM_PROC*)glfwGetProcAddress("glCreateProgram");
    bindings->glCreateShader = (PFNGLCREATESHADER_PROC*)glfwGetProcAddress("glCreateShader");
    bindings->glDeleteProgram = (PFNGLDELETEPROGRAM_PROC*)glfwGetProcAddress("glDeleteProgram");
    bindings->glDeleteShader = (PFNGLDELETESHADER_PROC*)glfwGetProcAddress("glDeleteShader");
    bindings->glDetachShader = (PFNGLDETACHSHADER_PROC*)glfwGetProcAddress("glDetachShader");
    bindings->glDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAY_PROC*)glfwGetProcAddress("glDisableVertexAttribArray");
    bindings->glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAY_PROC*)glfwGetProcAddress("glEnableVertexAttribArray");
    bindings->glGetActiveAttrib = (PFNGLGETACTIVEATTRIB_PROC*)glfwGetProcAddress("glGetActiveAttrib");
    bindings->glGetActiveUniform = (PFNGLGETACTIVEUNIFORM_PROC*)glfwGetProcAddress("glGetActiveUniform");
    bindings->glGetAttachedShaders = (PFNGLGETATTACHEDSHADERS_PROC*)glfwGetProcAddress("glGetAttachedShaders");
    bindings->glGetAttribLocation = (PFNGLGETATTRIBLOCATION_PROC*)glfwGetProcAddress("glGetAttribLocation");
    bindings->glGetProgramiv = (PFNGLGETPROGRAMIV_PROC*)glfwGetProcAddress("glGetProgramiv");
    bindings->glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOG_PROC*)glfwGetProcAddress("glGetProgramInfoLog");
    bindings->glGetShaderiv = (PFNGLGETSHADERIV_PROC*)glfwGetProcAddress("glGetShaderiv");
    bindings->glGetShaderInfoLog = (PFNGLGETSHADERINFOLOG_PROC*)glfwGetProcAddress("glGetShaderInfoLog");
    bindings->glGetShaderSource = (PFNGLGETSHADERSOURCE_PROC*)glfwGetProcAddress("glGetShaderSource");
    bindings->glGetUniformLocation = (PFNGLGETUNIFORMLOCATION_PROC*)glfwGetProcAddress("glGetUniformLocation");
    bindings->glGetUniformfv = (PFNGLGETUNIFORMFV_PROC*)glfwGetProcAddress("glGetUniformfv");
    bindings->glGetUniformiv = (PFNGLGETUNIFORMIV_PROC*)glfwGetProcAddress("glGetUniformiv");
    bindings->glGetVertexAttribdv = (PFNGLGETVERTEXATTRIBDV_PROC*)glfwGetProcAddress("glGetVertexAttribdv");
    bindings->glGetVertexAttribfv = (PFNGLGETVERTEXATTRIBFV_PROC*)glfwGetProcAddress("glGetVertexAttribfv");
    bindings->glGetVertexAttribiv = (PFNGLGETVERTEXATTRIBIV_PROC*)glfwGetProcAddress("glGetVertexAttribiv");
    bindings->glGetVertexAttribPointerv = (PFNGLGETVERTEXATTRIBPOINTERV_PROC*)glfwGetProcAddress("glGetVertexAttribPointerv");
    bindings->glIsProgram = (PFNGLISPROGRAM_PROC*)glfwGetProcAddress("glIsProgram");
    bindings->glIsShader = (PFNGLISSHADER_PROC*)glfwGetProcAddress("glIsShader");
    bindings->glLinkProgram = (PFNGLLINKPROGRAM_PROC*)glfwGetProcAddress("glLinkProgram");
    bindings->glShaderSource = (PFNGLSHADERSOURCE_PROC*)glfwGetProcAddress("glShaderSource");
    bindings->glUseProgram = (PFNGLUSEPROGRAM_PROC*)glfwGetProcAddress("glUseProgram");
    bindings->glUniform1f = (PFNGLUNIFORM1F_PROC*)glfwGetProcAddress("glUniform1f");
    bindings->glUniform2f = (PFNGLUNIFORM2F_PROC*)glfwGetProcAddress("glUniform2f");
    bindings->glUniform3f = (PFNGLUNIFORM3F_PROC*)glfwGetProcAddress("glUniform3f");
    bindings->glUniform4f = (PFNGLUNIFORM4F_PROC*)glfwGetProcAddress("glUniform4f");
    bindings->glUniform1i = (PFNGLUNIFORM1I_PROC*)glfwGetProcAddress("glUniform1i");
    bindings->glUniform2i = (PFNGLUNIFORM2I_PROC*)glfwGetProcAddress("glUniform2i");
    bindings->glUniform3i = (PFNGLUNIFORM3I_PROC*)glfwGetProcAddress("glUniform3i");
    bindings->glUniform4i = (PFNGLUNIFORM4I_PROC*)glfwGetProcAddress("glUniform4i");
    bindings->glUniform1fv = (PFNGLUNIFORM1FV_PROC*)glfwGetProcAddress("glUniform1fv");
    bindings->glUniform2fv = (PFNGLUNIFORM2FV_PROC*)glfwGetProcAddress("glUniform2fv");
    bindings->glUniform3fv = (PFNGLUNIFORM3FV_PROC*)glfwGetProcAddress("glUniform3fv");
    bindings->glUniform4fv = (PFNGLUNIFORM4FV_PROC*)glfwGetProcAddress("glUniform4fv");
    bindings->glUniform1iv = (PFNGLUNIFORM1IV_PROC*)glfwGetProcAddress("glUniform1iv");
    bindings->glUniform2iv = (PFNGLUNIFORM2IV_PROC*)glfwGetProcAddress("glUniform2iv");
    bindings->glUniform3iv = (PFNGLUNIFORM3IV_PROC*)glfwGetProcAddress("glUniform3iv");
    bindings->glUniform4iv = (PFNGLUNIFORM4IV_PROC*)glfwGetProcAddress("glUniform4iv");
    bindings->glUniformMatrix2fv = (PFNGLUNIFORMMATRIX2FV_PROC*)glfwGetProcAddress("glUniformMatrix2fv");
    bindings->glUniformMatrix3fv = (PFNGLUNIFORMMATRIX3FV_PROC*)glfwGetProcAddress("glUniformMatrix3fv");
    bindings->glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FV_PROC*)glfwGetProcAddress("glUniformMatrix4fv");
    bindings->glValidateProgram = (PFNGLVALIDATEPROGRAM_PROC*)glfwGetProcAddress("glValidateProgram");
    bindings->glVertexAttrib1d = (PFNGLVERTEXATTRIB1D_PROC*)glfwGetProcAddress("glVertexAttrib1d");
    bindings->glVertexAttrib1dv = (PFNGLVERTEXATTRIB1DV_PROC*)glfwGetProcAddress("glVertexAttrib1dv");
    bindings->glVertexAttrib1f = (PFNGLVERTEXATTRIB1F_PROC*)glfwGetProcAddress("glVertexAttrib1f");
    bindings->glVertexAttrib1fv = (PFNGLVERTEXATTRIB1FV_PROC*)glfwGetProcAddress("glVertexAttrib1fv");
    bindings->glVertexAttrib1s = (PFNGLVERTEXATTRIB1S_PROC*)glfwGetProcAddress("glVertexAttrib1s");
    bindings->glVertexAttrib1sv = (PFNGLVERTEXATTRIB1SV_PROC*)glfwGetProcAddress("glVertexAttrib1sv");
    bindings->glVertexAttrib2d = (PFNGLVERTEXATTRIB2D_PROC*)glfwGetProcAddress("glVertexAttrib2d");
    bindings->glVertexAttrib2dv = (PFNGLVERTEXATTRIB2DV_PROC*)glfwGetProcAddress("glVertexAttrib2dv");
    bindings->glVertexAttrib2f = (PFNGLVERTEXATTRIB2F_PROC*)glfwGetProcAddress("glVertexAttrib2f");
    bindings->glVertexAttrib2fv = (PFNGLVERTEXATTRIB2FV_PROC*)glfwGetProcAddress("glVertexAttrib2fv");
    bindings->glVertexAttrib2s = (PFNGLVERTEXATTRIB2S_PROC*)glfwGetProcAddress("glVertexAttrib2s");
    bindings->glVertexAttrib2sv = (PFNGLVERTEXATTRIB2SV_PROC*)glfwGetProcAddress("glVertexAttrib2sv");
    bindings->glVertexAttrib3d = (PFNGLVERTEXATTRIB3D_PROC*)glfwGetProcAddress("glVertexAttrib3d");
    bindings->glVertexAttrib3dv = (PFNGLVERTEXATTRIB3DV_PROC*)glfwGetProcAddress("glVertexAttrib3dv");
    bindings->glVertexAttrib3f = (PFNGLVERTEXATTRIB3F_PROC*)glfwGetProcAddress("glVertexAttrib3f");
    bindings->glVertexAttrib3fv = (PFNGLVERTEXATTRIB3FV_PROC*)glfwGetProcAddress("glVertexAttrib3fv");
    bindings->glVertexAttrib3s = (PFNGLVERTEXATTRIB3S_PROC*)glfwGetProcAddress("glVertexAttrib3s");
    bindings->glVertexAttrib3sv = (PFNGLVERTEXATTRIB3SV_PROC*)glfwGetProcAddress("glVertexAttrib3sv");
    bindings->glVertexAttrib4Nbv = (PFNGLVERTEXATTRIB4NBV_PROC*)glfwGetProcAddress("glVertexAttrib4Nbv");
    bindings->glVertexAttrib4Niv = (PFNGLVERTEXATTRIB4NIV_PROC*)glfwGetProcAddress("glVertexAttrib4Niv");
    bindings->glVertexAttrib4Nsv = (PFNGLVERTEXATTRIB4NSV_PROC*)glfwGetProcAddress("glVertexAttrib4Nsv");
    bindings->glVertexAttrib4Nub = (PFNGLVERTEXATTRIB4NUB_PROC*)glfwGetProcAddress("glVertexAttrib4Nub");
    bindings->glVertexAttrib4Nubv = (PFNGLVERTEXATTRIB4NUBV_PROC*)glfwGetProcAddress("glVertexAttrib4Nubv");
    bindings->glVertexAttrib4Nuiv = (PFNGLVERTEXATTRIB4NUIV_PROC*)glfwGetProcAddress("glVertexAttrib4Nuiv");
    bindings->glVertexAttrib4Nusv = (PFNGLVERTEXATTRIB4NUSV_PROC*)glfwGetProcAddress("glVertexAttrib4Nusv");
    bindings->glVertexAttrib4bv = (PFNGLVERTEXATTRIB4BV_PROC*)glfwGetProcAddress("glVertexAttrib4bv");
    bindings->glVertexAttrib4d = (PFNGLVERTEXATTRIB4D_PROC*)glfwGetProcAddress("glVertexAttrib4d");
    bindings->glVertexAttrib4dv = (PFNGLVERTEXATTRIB4DV_PROC*)glfwGetProcAddress("glVertexAttrib4dv");
    bindings->glVertexAttrib4f = (PFNGLVERTEXATTRIB4F_PROC*)glfwGetProcAddress("glVertexAttrib4f");
    bindings->glVertexAttrib4fv = (PFNGLVERTEXATTRIB4FV_PROC*)glfwGetProcAddress("glVertexAttrib4fv");
    bindings->glVertexAttrib4iv = (PFNGLVERTEXATTRIB4IV_PROC*)glfwGetProcAddress("glVertexAttrib4iv");
    bindings->glVertexAttrib4s = (PFNGLVERTEXATTRIB4S_PROC*)glfwGetProcAddress("glVertexAttrib4s");
    bindings->glVertexAttrib4sv = (PFNGLVERTEXATTRIB4SV_PROC*)glfwGetProcAddress("glVertexAttrib4sv");
    bindings->glVertexAttrib4ubv = (PFNGLVERTEXATTRIB4UBV_PROC*)glfwGetProcAddress("glVertexAttrib4ubv");
    bindings->glVertexAttrib4uiv = (PFNGLVERTEXATTRIB4UIV_PROC*)glfwGetProcAddress("glVertexAttrib4uiv");
    bindings->glVertexAttrib4usv = (PFNGLVERTEXATTRIB4USV_PROC*)glfwGetProcAddress("glVertexAttrib4usv");
    bindings->glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTER_PROC*)glfwGetProcAddress("glVertexAttribPointer");

    /* GL_VERSION_2_1 */

    bindings->glUniformMatrix2x3fv = (PFNGLUNIFORMMATRIX2X3FV_PROC*)glfwGetProcAddress("glUniformMatrix2x3fv");
    bindings->glUniformMatrix3x2fv = (PFNGLUNIFORMMATRIX3X2FV_PROC*)glfwGetProcAddress("glUniformMatrix3x2fv");
    bindings->glUniformMatrix2x4fv = (PFNGLUNIFORMMATRIX2X4FV_PROC*)glfwGetProcAddress("glUniformMatrix2x4fv");
    bindings->glUniformMatrix4x2fv = (PFNGLUNIFORMMATRIX4X2FV_PROC*)glfwGetProcAddress("glUniformMatrix4x2fv");
    bindings->glUniformMatrix3x4fv = (PFNGLUNIFORMMATRIX3X4FV_PROC*)glfwGetProcAddress("glUniformMatrix3x4fv");
    bindings->glUniformMatrix4x3fv = (PFNGLUNIFORMMATRIX4X3FV_PROC*)glfwGetProcAddress("glUniformMatrix4x3fv");

    /* GL_VERSION_3_0 */

    bindings->glColorMaski = (PFNGLCOLORMASKI_PROC*)glfwGetProcAddress("glColorMaski");
    bindings->glGetBooleani_v = (PFNGLGETBOOLEANI_V_PROC*)glfwGetProcAddress("glGetBooleani_v");
    bindings->glGetIntegeri_v = (PFNGLGETINTEGERI_V_PROC*)glfwGetProcAddress("glGetIntegeri_v");
    bindings->glEnablei = (PFNGLENABLEI_PROC*)glfwGetProcAddress("glEnablei");
    bindings->glDisablei = (PFNGLDISABLEI_PROC*)glfwGetProcAddress("glDisablei");
    bindings->glIsEnabledi = (PFNGLISENABLEDI_PROC*)glfwGetProcAddress("glIsEnabledi");
    bindings->glBeginTransformFeedback = (PFNGLBEGINTRANSFORMFEEDBACK_PROC*)glfwGetProcAddress("glBeginTransformFeedback");
    bindings->glEndTransformFeedback = (PFNGLENDTRANSFORMFEEDBACK_PROC*)glfwGetProcAddress("glEndTransformFeedback");
    bindings->glBindBufferRange = (PFNGLBINDBUFFERRANGE_PROC*)glfwGetProcAddress("glBindBufferRange");
    bindings->glBindBufferBase = (PFNGLBINDBUFFERBASE_PROC*)glfwGetProcAddress("glBindBufferBase");
    bindings->glTransformFeedbackVaryings = (PFNGLTRANSFORMFEEDBACKVARYINGS_PROC*)glfwGetProcAddress("glTransformFeedbackVaryings");
    bindings->glGetTransformFeedbackVarying = (PFNGLGETTRANSFORMFEEDBACKVARYING_PROC*)glfwGetProcAddress("glGetTransformFeedbackVarying");
    bindings->glClampColor = (PFNGLCLAMPCOLOR_PROC*)glfwGetProcAddress("glClampColor");
    bindings->glBeginConditionalRender = (PFNGLBEGINCONDITIONALRENDER_PROC*)glfwGetProcAddress("glBeginConditionalRender");
    bindings->glEndConditionalRender = (PFNGLENDCONDITIONALRENDER_PROC*)glfwGetProcAddress("glEndConditionalRender");
    bindings->glVertexAttribIPointer = (PFNGLVERTEXATTRIBIPOINTER_PROC*)glfwGetProcAddress("glVertexAttribIPointer");
    bindings->glGetVertexAttribIiv = (PFNGLGETVERTEXATTRIBIIV_PROC*)glfwGetProcAddress("glGetVertexAttribIiv");
    bindings->glGetVertexAttribIuiv = (PFNGLGETVERTEXATTRIBIUIV_PROC*)glfwGetProcAddress("glGetVertexAttribIuiv");
    bindings->glVertexAttribI1i = (PFNGLVERTEXATTRIBI1I_PROC*)glfwGetProcAddress("glVertexAttribI1i");
    bindings->glVertexAttribI2i = (PFNGLVERTEXATTRIBI2I_PROC*)glfwGetProcAddress("glVertexAttribI2i");
    bindings->glVertexAttribI3i = (PFNGLVERTEXATTRIBI3I_PROC*)glfwGetProcAddress("glVertexAttribI3i");
    bindings->glVertexAttribI4i = (PFNGLVERTEXATTRIBI4I_PROC*)glfwGetProcAddress("glVertexAttribI4i");
    bindings->glVertexAttribI1ui = (PFNGLVERTEXATTRIBI1UI_PROC*)glfwGetProcAddress("glVertexAttribI1ui");
    bindings->glVertexAttribI2ui = (PFNGLVERTEXATTRIBI2UI_PROC*)glfwGetProcAddress("glVertexAttribI2ui");
    bindings->glVertexAttribI3ui = (PFNGLVERTEXATTRIBI3UI_PROC*)glfwGetProcAddress("glVertexAttribI3ui");
    bindings->glVertexAttribI4ui = (PFNGLVERTEXATTRIBI4UI_PROC*)glfwGetProcAddress("glVertexAttribI4ui");
    bindings->glVertexAttribI1iv = (PFNGLVERTEXATTRIBI1IV_PROC*)glfwGetProcAddress("glVertexAttribI1iv");
    bindings->glVertexAttribI2iv = (PFNGLVERTEXATTRIBI2IV_PROC*)glfwGetProcAddress("glVertexAttribI2iv");
    bindings->glVertexAttribI3iv = (PFNGLVERTEXATTRIBI3IV_PROC*)glfwGetProcAddress("glVertexAttribI3iv");
    bindings->glVertexAttribI4iv = (PFNGLVERTEXATTRIBI4IV_PROC*)glfwGetProcAddress("glVertexAttribI4iv");
    bindings->glVertexAttribI1uiv = (PFNGLVERTEXATTRIBI1UIV_PROC*)glfwGetProcAddress("glVertexAttribI1uiv");
    bindings->glVertexAttribI2uiv = (PFNGLVERTEXATTRIBI2UIV_PROC*)glfwGetProcAddress("glVertexAttribI2uiv");
    bindings->glVertexAttribI3uiv = (PFNGLVERTEXATTRIBI3UIV_PROC*)glfwGetProcAddress("glVertexAttribI3uiv");
    bindings->glVertexAttribI4uiv = (PFNGLVERTEXATTRIBI4UIV_PROC*)glfwGetProcAddress("glVertexAttribI4uiv");
    bindings->glVertexAttribI4bv = (PFNGLVERTEXATTRIBI4BV_PROC*)glfwGetProcAddress("glVertexAttribI4bv");
    bindings->glVertexAttribI4sv = (PFNGLVERTEXATTRIBI4SV_PROC*)glfwGetProcAddress("glVertexAttribI4sv");
    bindings->glVertexAttribI4ubv = (PFNGLVERTEXATTRIBI4UBV_PROC*)glfwGetProcAddress("glVertexAttribI4ubv");
    bindings->glVertexAttribI4usv = (PFNGLVERTEXATTRIBI4USV_PROC*)glfwGetProcAddress("glVertexAttribI4usv");
    bindings->glGetUniformuiv = (PFNGLGETUNIFORMUIV_PROC*)glfwGetProcAddress("glGetUniformuiv");
    bindings->glBindFragDataLocation = (PFNGLBINDFRAGDATALOCATION_PROC*)glfwGetProcAddress("glBindFragDataLocation");
    bindings->glGetFragDataLocation = (PFNGLGETFRAGDATALOCATION_PROC*)glfwGetProcAddress("glGetFragDataLocation");
    bindings->glUniform1ui = (PFNGLUNIFORM1UI_PROC*)glfwGetProcAddress("glUniform1ui");
    bindings->glUniform2ui = (PFNGLUNIFORM2UI_PROC*)glfwGetProcAddress("glUniform2ui");
    bindings->glUniform3ui = (PFNGLUNIFORM3UI_PROC*)glfwGetProcAddress("glUniform3ui");
    bindings->glUniform4ui = (PFNGLUNIFORM4UI_PROC*)glfwGetProcAddress("glUniform4ui");
    bindings->glUniform1uiv = (PFNGLUNIFORM1UIV_PROC*)glfwGetProcAddress("glUniform1uiv");
    bindings->glUniform2uiv = (PFNGLUNIFORM2UIV_PROC*)glfwGetProcAddress("glUniform2uiv");
    bindings->glUniform3uiv = (PFNGLUNIFORM3UIV_PROC*)glfwGetProcAddress("glUniform3uiv");
    bindings->glUniform4uiv = (PFNGLUNIFORM4UIV_PROC*)glfwGetProcAddress("glUniform4uiv");
    bindings->glTexParameterIiv = (PFNGLTEXPARAMETERIIV_PROC*)glfwGetProcAddress("glTexParameterIiv");
    bindings->glTexParameterIuiv = (PFNGLTEXPARAMETERIUIV_PROC*)glfwGetProcAddress("glTexParameterIuiv");
    bindings->glGetTexParameterIiv = (PFNGLGETTEXPARAMETERIIV_PROC*)glfwGetProcAddress("glGetTexParameterIiv");
    bindings->glGetTexParameterIuiv = (PFNGLGETTEXPARAMETERIUIV_PROC*)glfwGetProcAddress("glGetTexParameterIuiv");
    bindings->glClearBufferiv = (PFNGLCLEARBUFFERIV_PROC*)glfwGetProcAddress("glClearBufferiv");
    bindings->glClearBufferuiv = (PFNGLCLEARBUFFERUIV_PROC*)glfwGetProcAddress("glClearBufferuiv");
    bindings->glClearBufferfv = (PFNGLCLEARBUFFERFV_PROC*)glfwGetProcAddress("glClearBufferfv");
    bindings->glClearBufferfi = (PFNGLCLEARBUFFERFI_PROC*)glfwGetProcAddress("glClearBufferfi");
    bindings->glGetStringi = (PFNGLGETSTRINGI_PROC*)glfwGetProcAddress("glGetStringi");
    bindings->glIsRenderbuffer = (PFNGLISRENDERBUFFER_PROC*)glfwGetProcAddress("glIsRenderbuffer");
    bindings->glBindRenderbuffer = (PFNGLBINDRENDERBUFFER_PROC*)glfwGetProcAddress("glBindRenderbuffer");
    bindings->glDeleteRenderbuffers = (PFNGLDELETERENDERBUFFERS_PROC*)glfwGetProcAddress("glDeleteRenderbuffers");
    bindings->glGenRenderbuffers = (PFNGLGENRENDERBUFFERS_PROC*)glfwGetProcAddress("glGenRenderbuffers");
    bindings->glRenderbufferStorage = (PFNGLRENDERBUFFERSTORAGE_PROC*)glfwGetProcAddress("glRenderbufferStorage");
    bindings->glGetRenderbufferParameteriv = (PFNGLGETRENDERBUFFERPARAMETERIV_PROC*)glfwGetProcAddress("glGetRenderbufferParameteriv");
    bindings->glIsFramebuffer = (PFNGLISFRAMEBUFFER_PROC*)glfwGetProcAddress("glIsFramebuffer");
    bindings->glBindFramebuffer = (PFNGLBINDFRAMEBUFFER_PROC*)glfwGetProcAddress("glBindFramebuffer");
    bindings->glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERS_PROC*)glfwGetProcAddress("glDeleteFramebuffers");
    bindings->glGenFramebuffers = (PFNGLGENFRAMEBUFFERS_PROC*)glfwGetProcAddress("glGenFramebuffers");
    bindings->glCheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUS_PROC*)glfwGetProcAddress("glCheckFramebufferStatus");
    bindings->glFramebufferTexture1D = (PFNGLFRAMEBUFFERTEXTURE1D_PROC*)glfwGetProcAddress("glFramebufferTexture1D");
    bindings->glFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2D_PROC*)glfwGetProcAddress("glFramebufferTexture2D");
    bindings->glFramebufferTexture3D = (PFNGLFRAMEBUFFERTEXTURE3D_PROC*)glfwGetProcAddress("glFramebufferTexture3D");
    bindings->glFramebufferRenderbuffer = (PFNGLFRAMEBUFFERRENDERBUFFER_PROC*)glfwGetProcAddress("glFramebufferRenderbuffer");
    bindings->glGetFramebufferAttachmentParameteriv = (PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIV_PROC*)glfwGetProcAddress("glGetFramebufferAttachmentParameteriv");
    bindings->glGenerateMipmap = (PFNGLGENERATEMIPMAP_PROC*)glfwGetProcAddress("glGenerateMipmap");
    bindings->glBlitFramebuffer = (PFNGLBLITFRAMEBUFFER_PROC*)glfwGetProcAddress("glBlitFramebuffer");
    bindings->glRenderbufferStorageMultisample = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLE_PROC*)glfwGetProcAddress("glRenderbufferStorageMultisample");
    bindings->glFramebufferTextureLayer = (PFNGLFRAMEBUFFERTEXTURELAYER_PROC*)glfwGetProcAddress("glFramebufferTextureLayer");
    bindings->glMapBufferRange = (PFNGLMAPBUFFERRANGE_PROC*)glfwGetProcAddress("glMapBufferRange");
    bindings->glFlushMappedBufferRange = (PFNGLFLUSHMAPPEDBUFFERRANGE_PROC*)glfwGetProcAddress("glFlushMappedBufferRange");
    bindings->glBindVertexArray = (PFNGLBINDVERTEXARRAY_PROC*)glfwGetProcAddress("glBindVertexArray");
    bindings->glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYS_PROC*)glfwGetProcAddress("glDeleteVertexArrays");
    bindings->glGenVertexArrays = (PFNGLGENVERTEXARRAYS_PROC*)glfwGetProcAddress("glGenVertexArrays");
    bindings->glIsVertexArray = (PFNGLISVERTEXARRAY_PROC*)glfwGetProcAddress("glIsVertexArray");

    /* GL_VERSION_3_1 */

    bindings->glDrawArraysInstanced = (PFNGLDRAWARRAYSINSTANCED_PROC*)glfwGetProcAddress("glDrawArraysInstanced");
    bindings->glDrawElementsInstanced = (PFNGLDRAWELEMENTSINSTANCED_PROC*)glfwGetProcAddress("glDrawElementsInstanced");
    bindings->glTexBuffer = (PFNGLTEXBUFFER_PROC*)glfwGetProcAddress("glTexBuffer");
    bindings->glPrimitiveRestartIndex = (PFNGLPRIMITIVERESTARTINDEX_PROC*)glfwGetProcAddress("glPrimitiveRestartIndex");
    bindings->glCopyBufferSubData = (PFNGLCOPYBUFFERSUBDATA_PROC*)glfwGetProcAddress("glCopyBufferSubData");
    bindings->glGetUniformIndices = (PFNGLGETUNIFORMINDICES_PROC*)glfwGetProcAddress("glGetUniformIndices");
    bindings->glGetActiveUniformsiv = (PFNGLGETACTIVEUNIFORMSIV_PROC*)glfwGetProcAddress("glGetActiveUniformsiv");
    bindings->glGetActiveUniformName = (PFNGLGETACTIVEUNIFORMNAME_PROC*)glfwGetProcAddress("glGetActiveUniformName");
    bindings->glGetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEX_PROC*)glfwGetProcAddress("glGetUniformBlockIndex");
    bindings->glGetActiveUniformBlockiv = (PFNGLGETACTIVEUNIFORMBLOCKIV_PROC*)glfwGetProcAddress("glGetActiveUniformBlockiv");
    bindings->glGetActiveUniformBlockName = (PFNGLGETACTIVEUNIFORMBLOCKNAME_PROC*)glfwGetProcAddress("glGetActiveUniformBlockName");
    bindings->glUniformBlockBinding = (PFNGLUNIFORMBLOCKBINDING_PROC*)glfwGetProcAddress("glUniformBlockBinding");

}

/* ----------------------- Extension flag definitions ---------------------- */

#ifdef __cplusplus
}
#endif
