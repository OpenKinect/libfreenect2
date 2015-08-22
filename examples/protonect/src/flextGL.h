#ifndef __gl_h_
#define __gl_h_

#ifdef __cplusplus
extern "C" {
#endif

/* Function declaration macros - to move into glplatform.h */

#if defined(_WIN32) && !defined(APIENTRY) && !defined(__CYGWIN__) && !defined(__SCITECH_SNAP__)
#define WIN32_LEAN_AND_MEAN 1
#ifndef WINAPI
#define WINAPI __stdcall
#endif
#define APIENTRY WINAPI
#endif

#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef APIENTRYP
#define APIENTRYP APIENTRY *
#endif
#ifndef GLAPI
#define GLAPI extern
#endif

/* -------------------------------- DATA TYPES ------------------------------- */

#include <stddef.h>
#ifndef GLEXT_64_TYPES_DEFINED
/* This code block is duplicated in glxext.h, so must be protected */
#define GLEXT_64_TYPES_DEFINED
/* Define int32_t, int64_t, and uint64_t types for UST/MSC */
/* (as used in the GL_EXT_timer_query extension). */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#include <inttypes.h>
#elif defined(__sun__) || defined(__digital__)
#include <inttypes.h>
#if defined(__STDC__)
#if defined(__arch64__) || defined(_LP64)
typedef long int int64_t;
typedef unsigned long int uint64_t;
#else
typedef long long int int64_t;
typedef unsigned long long int uint64_t;
#endif /* __arch64__ */
#endif /* __STDC__ */
#elif defined( __VMS ) || defined(__sgi)
#include <inttypes.h>
#elif defined(__SCO__) || defined(__USLC__)
#include <stdint.h>
#elif defined(__UNIXOS2__) || defined(__SOL64__)
typedef long int int32_t;
typedef long long int int64_t;
typedef unsigned long long int uint64_t;
#elif defined(_WIN32) && defined(__GNUC__)
#include <stdint.h>
#elif defined(_WIN32)
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
/* Fallback if nothing above works */
#include <inttypes.h>
#endif
#endif
typedef unsigned int GLenum;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef void GLvoid;
typedef signed char GLbyte;
typedef short GLshort;
typedef int GLint;
typedef unsigned char GLubyte;
typedef unsigned short GLushort;
typedef unsigned int GLuint;
typedef int GLsizei;
typedef float GLfloat;
typedef float GLclampf;
typedef double GLdouble;
typedef double GLclampd;
typedef char GLchar;
typedef unsigned short GLhalf;
typedef ptrdiff_t GLintptr;
typedef ptrdiff_t GLsizeiptr;
typedef int64_t GLint64;
typedef uint64_t GLuint64;
typedef struct __GLsync *GLsync;

/* ----------------------------------- ENUMS --------------------------------- */

/* GL_VERSION_1_1 */

#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_STENCIL_BUFFER_BIT 0x00000400
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_FALSE 0
#define GL_TRUE 1
#define GL_POINTS 0x0000
#define GL_LINES 0x0001
#define GL_LINE_LOOP 0x0002
#define GL_LINE_STRIP 0x0003
#define GL_TRIANGLES 0x0004
#define GL_TRIANGLE_STRIP 0x0005
#define GL_TRIANGLE_FAN 0x0006
#define GL_NEVER 0x0200
#define GL_LESS 0x0201
#define GL_EQUAL 0x0202
#define GL_LEQUAL 0x0203
#define GL_GREATER 0x0204
#define GL_NOTEQUAL 0x0205
#define GL_GEQUAL 0x0206
#define GL_ALWAYS 0x0207
#define GL_ZERO 0
#define GL_ONE 1
#define GL_SRC_COLOR 0x0300
#define GL_ONE_MINUS_SRC_COLOR 0x0301
#define GL_SRC_ALPHA 0x0302
#define GL_ONE_MINUS_SRC_ALPHA 0x0303
#define GL_DST_ALPHA 0x0304
#define GL_ONE_MINUS_DST_ALPHA 0x0305
#define GL_DST_COLOR 0x0306
#define GL_ONE_MINUS_DST_COLOR 0x0307
#define GL_SRC_ALPHA_SATURATE 0x0308
#define GL_NONE 0
#define GL_FRONT_LEFT 0x0400
#define GL_FRONT_RIGHT 0x0401
#define GL_BACK_LEFT 0x0402
#define GL_BACK_RIGHT 0x0403
#define GL_FRONT 0x0404
#define GL_BACK 0x0405
#define GL_LEFT 0x0406
#define GL_RIGHT 0x0407
#define GL_FRONT_AND_BACK 0x0408
#define GL_NO_ERROR 0
#define GL_INVALID_ENUM 0x0500
#define GL_INVALID_VALUE 0x0501
#define GL_INVALID_OPERATION 0x0502
#define GL_OUT_OF_MEMORY 0x0505
#define GL_CW 0x0900
#define GL_CCW 0x0901
#define GL_POINT_SIZE 0x0B11
#define GL_POINT_SIZE_RANGE 0x0B12
#define GL_POINT_SIZE_GRANULARITY 0x0B13
#define GL_LINE_SMOOTH 0x0B20
#define GL_LINE_WIDTH 0x0B21
#define GL_LINE_WIDTH_RANGE 0x0B22
#define GL_LINE_WIDTH_GRANULARITY 0x0B23
#define GL_POLYGON_MODE 0x0B40
#define GL_POLYGON_SMOOTH 0x0B41
#define GL_CULL_FACE 0x0B44
#define GL_CULL_FACE_MODE 0x0B45
#define GL_FRONT_FACE 0x0B46
#define GL_DEPTH_RANGE 0x0B70
#define GL_DEPTH_TEST 0x0B71
#define GL_DEPTH_WRITEMASK 0x0B72
#define GL_DEPTH_CLEAR_VALUE 0x0B73
#define GL_DEPTH_FUNC 0x0B74
#define GL_STENCIL_TEST 0x0B90
#define GL_STENCIL_CLEAR_VALUE 0x0B91
#define GL_STENCIL_FUNC 0x0B92
#define GL_STENCIL_VALUE_MASK 0x0B93
#define GL_STENCIL_FAIL 0x0B94
#define GL_STENCIL_PASS_DEPTH_FAIL 0x0B95
#define GL_STENCIL_PASS_DEPTH_PASS 0x0B96
#define GL_STENCIL_REF 0x0B97
#define GL_STENCIL_WRITEMASK 0x0B98
#define GL_VIEWPORT 0x0BA2
#define GL_DITHER 0x0BD0
#define GL_BLEND_DST 0x0BE0
#define GL_BLEND_SRC 0x0BE1
#define GL_BLEND 0x0BE2
#define GL_LOGIC_OP_MODE 0x0BF0
#define GL_COLOR_LOGIC_OP 0x0BF2
#define GL_DRAW_BUFFER 0x0C01
#define GL_READ_BUFFER 0x0C02
#define GL_SCISSOR_BOX 0x0C10
#define GL_SCISSOR_TEST 0x0C11
#define GL_COLOR_CLEAR_VALUE 0x0C22
#define GL_COLOR_WRITEMASK 0x0C23
#define GL_DOUBLEBUFFER 0x0C32
#define GL_STEREO 0x0C33
#define GL_LINE_SMOOTH_HINT 0x0C52
#define GL_POLYGON_SMOOTH_HINT 0x0C53
#define GL_UNPACK_SWAP_BYTES 0x0CF0
#define GL_UNPACK_LSB_FIRST 0x0CF1
#define GL_UNPACK_ROW_LENGTH 0x0CF2
#define GL_UNPACK_SKIP_ROWS 0x0CF3
#define GL_UNPACK_SKIP_PIXELS 0x0CF4
#define GL_UNPACK_ALIGNMENT 0x0CF5
#define GL_PACK_SWAP_BYTES 0x0D00
#define GL_PACK_LSB_FIRST 0x0D01
#define GL_PACK_ROW_LENGTH 0x0D02
#define GL_PACK_SKIP_ROWS 0x0D03
#define GL_PACK_SKIP_PIXELS 0x0D04
#define GL_PACK_ALIGNMENT 0x0D05
#define GL_MAX_TEXTURE_SIZE 0x0D33
#define GL_MAX_VIEWPORT_DIMS 0x0D3A
#define GL_SUBPIXEL_BITS 0x0D50
#define GL_TEXTURE_1D 0x0DE0
#define GL_TEXTURE_2D 0x0DE1
#define GL_POLYGON_OFFSET_UNITS 0x2A00
#define GL_POLYGON_OFFSET_POINT 0x2A01
#define GL_POLYGON_OFFSET_LINE 0x2A02
#define GL_POLYGON_OFFSET_FILL 0x8037
#define GL_POLYGON_OFFSET_FACTOR 0x8038
#define GL_TEXTURE_BINDING_1D 0x8068
#define GL_TEXTURE_BINDING_2D 0x8069
#define GL_TEXTURE_WIDTH 0x1000
#define GL_TEXTURE_HEIGHT 0x1001
#define GL_TEXTURE_INTERNAL_FORMAT 0x1003
#define GL_TEXTURE_BORDER_COLOR 0x1004
#define GL_TEXTURE_RED_SIZE 0x805C
#define GL_TEXTURE_GREEN_SIZE 0x805D
#define GL_TEXTURE_BLUE_SIZE 0x805E
#define GL_TEXTURE_ALPHA_SIZE 0x805F
#define GL_DONT_CARE 0x1100
#define GL_FASTEST 0x1101
#define GL_NICEST 0x1102
#define GL_BYTE 0x1400
#define GL_UNSIGNED_BYTE 0x1401
#define GL_SHORT 0x1402
#define GL_UNSIGNED_SHORT 0x1403
#define GL_INT 0x1404
#define GL_UNSIGNED_INT 0x1405
#define GL_FLOAT 0x1406
#define GL_DOUBLE 0x140A
#define GL_CLEAR 0x1500
#define GL_AND 0x1501
#define GL_AND_REVERSE 0x1502
#define GL_COPY 0x1503
#define GL_AND_INVERTED 0x1504
#define GL_NOOP 0x1505
#define GL_XOR 0x1506
#define GL_OR 0x1507
#define GL_NOR 0x1508
#define GL_EQUIV 0x1509
#define GL_INVERT 0x150A
#define GL_OR_REVERSE 0x150B
#define GL_COPY_INVERTED 0x150C
#define GL_OR_INVERTED 0x150D
#define GL_NAND 0x150E
#define GL_SET 0x150F
#define GL_TEXTURE 0x1702
#define GL_COLOR 0x1800
#define GL_DEPTH 0x1801
#define GL_STENCIL 0x1802
#define GL_STENCIL_INDEX 0x1901
#define GL_DEPTH_COMPONENT 0x1902
#define GL_RED 0x1903
#define GL_GREEN 0x1904
#define GL_BLUE 0x1905
#define GL_ALPHA 0x1906
#define GL_RGB 0x1907
#define GL_RGBA 0x1908
#define GL_POINT 0x1B00
#define GL_LINE 0x1B01
#define GL_FILL 0x1B02
#define GL_KEEP 0x1E00
#define GL_REPLACE 0x1E01
#define GL_INCR 0x1E02
#define GL_DECR 0x1E03
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_EXTENSIONS 0x1F03
#define GL_NEAREST 0x2600
#define GL_LINEAR 0x2601
#define GL_NEAREST_MIPMAP_NEAREST 0x2700
#define GL_LINEAR_MIPMAP_NEAREST 0x2701
#define GL_NEAREST_MIPMAP_LINEAR 0x2702
#define GL_LINEAR_MIPMAP_LINEAR 0x2703
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_PROXY_TEXTURE_1D 0x8063
#define GL_PROXY_TEXTURE_2D 0x8064
#define GL_REPEAT 0x2901
#define GL_R3_G3_B2 0x2A10
#define GL_RGB4 0x804F
#define GL_RGB5 0x8050
#define GL_RGB8 0x8051
#define GL_RGB10 0x8052
#define GL_RGB12 0x8053
#define GL_RGB16 0x8054
#define GL_RGBA2 0x8055
#define GL_RGBA4 0x8056
#define GL_RGB5_A1 0x8057
#define GL_RGBA8 0x8058
#define GL_RGB10_A2 0x8059
#define GL_RGBA12 0x805A
#define GL_RGBA16 0x805B

/* GL_VERSION_1_2 */

#define GL_UNSIGNED_BYTE_3_3_2 0x8032
#define GL_UNSIGNED_SHORT_4_4_4_4 0x8033
#define GL_UNSIGNED_SHORT_5_5_5_1 0x8034
#define GL_UNSIGNED_INT_8_8_8_8 0x8035
#define GL_UNSIGNED_INT_10_10_10_2 0x8036
#define GL_TEXTURE_BINDING_3D 0x806A
#define GL_PACK_SKIP_IMAGES 0x806B
#define GL_PACK_IMAGE_HEIGHT 0x806C
#define GL_UNPACK_SKIP_IMAGES 0x806D
#define GL_UNPACK_IMAGE_HEIGHT 0x806E
#define GL_TEXTURE_3D 0x806F
#define GL_PROXY_TEXTURE_3D 0x8070
#define GL_TEXTURE_DEPTH 0x8071
#define GL_TEXTURE_WRAP_R 0x8072
#define GL_MAX_3D_TEXTURE_SIZE 0x8073
#define GL_UNSIGNED_BYTE_2_3_3_REV 0x8362
#define GL_UNSIGNED_SHORT_5_6_5 0x8363
#define GL_UNSIGNED_SHORT_5_6_5_REV 0x8364
#define GL_UNSIGNED_SHORT_4_4_4_4_REV 0x8365
#define GL_UNSIGNED_SHORT_1_5_5_5_REV 0x8366
#define GL_UNSIGNED_INT_8_8_8_8_REV 0x8367
#define GL_UNSIGNED_INT_2_10_10_10_REV 0x8368
#define GL_BGR 0x80E0
#define GL_BGRA 0x80E1
#define GL_MAX_ELEMENTS_VERTICES 0x80E8
#define GL_MAX_ELEMENTS_INDICES 0x80E9
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_TEXTURE_MIN_LOD 0x813A
#define GL_TEXTURE_MAX_LOD 0x813B
#define GL_TEXTURE_BASE_LEVEL 0x813C
#define GL_TEXTURE_MAX_LEVEL 0x813D
#define GL_SMOOTH_POINT_SIZE_RANGE 0x0B12
#define GL_SMOOTH_POINT_SIZE_GRANULARITY 0x0B13
#define GL_SMOOTH_LINE_WIDTH_RANGE 0x0B22
#define GL_SMOOTH_LINE_WIDTH_GRANULARITY 0x0B23
#define GL_ALIASED_LINE_WIDTH_RANGE 0x846E

/* GL_VERSION_1_3 */

#define GL_TEXTURE0 0x84C0
#define GL_TEXTURE1 0x84C1
#define GL_TEXTURE2 0x84C2
#define GL_TEXTURE3 0x84C3
#define GL_TEXTURE4 0x84C4
#define GL_TEXTURE5 0x84C5
#define GL_TEXTURE6 0x84C6
#define GL_TEXTURE7 0x84C7
#define GL_TEXTURE8 0x84C8
#define GL_TEXTURE9 0x84C9
#define GL_TEXTURE10 0x84CA
#define GL_TEXTURE11 0x84CB
#define GL_TEXTURE12 0x84CC
#define GL_TEXTURE13 0x84CD
#define GL_TEXTURE14 0x84CE
#define GL_TEXTURE15 0x84CF
#define GL_TEXTURE16 0x84D0
#define GL_TEXTURE17 0x84D1
#define GL_TEXTURE18 0x84D2
#define GL_TEXTURE19 0x84D3
#define GL_TEXTURE20 0x84D4
#define GL_TEXTURE21 0x84D5
#define GL_TEXTURE22 0x84D6
#define GL_TEXTURE23 0x84D7
#define GL_TEXTURE24 0x84D8
#define GL_TEXTURE25 0x84D9
#define GL_TEXTURE26 0x84DA
#define GL_TEXTURE27 0x84DB
#define GL_TEXTURE28 0x84DC
#define GL_TEXTURE29 0x84DD
#define GL_TEXTURE30 0x84DE
#define GL_TEXTURE31 0x84DF
#define GL_ACTIVE_TEXTURE 0x84E0
#define GL_MULTISAMPLE 0x809D
#define GL_SAMPLE_ALPHA_TO_COVERAGE 0x809E
#define GL_SAMPLE_ALPHA_TO_ONE 0x809F
#define GL_SAMPLE_COVERAGE 0x80A0
#define GL_SAMPLE_BUFFERS 0x80A8
#define GL_SAMPLES 0x80A9
#define GL_SAMPLE_COVERAGE_VALUE 0x80AA
#define GL_SAMPLE_COVERAGE_INVERT 0x80AB
#define GL_TEXTURE_CUBE_MAP 0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP 0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X 0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y 0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y 0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z 0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z 0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP 0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE 0x851C
#define GL_COMPRESSED_RGB 0x84ED
#define GL_COMPRESSED_RGBA 0x84EE
#define GL_TEXTURE_COMPRESSION_HINT 0x84EF
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE 0x86A0
#define GL_TEXTURE_COMPRESSED 0x86A1
#define GL_NUM_COMPRESSED_TEXTURE_FORMATS 0x86A2
#define GL_COMPRESSED_TEXTURE_FORMATS 0x86A3
#define GL_CLAMP_TO_BORDER 0x812D

/* GL_VERSION_1_4 */

#define GL_BLEND_DST_RGB 0x80C8
#define GL_BLEND_SRC_RGB 0x80C9
#define GL_BLEND_DST_ALPHA 0x80CA
#define GL_BLEND_SRC_ALPHA 0x80CB
#define GL_POINT_FADE_THRESHOLD_SIZE 0x8128
#define GL_DEPTH_COMPONENT16 0x81A5
#define GL_DEPTH_COMPONENT24 0x81A6
#define GL_DEPTH_COMPONENT32 0x81A7
#define GL_MIRRORED_REPEAT 0x8370
#define GL_MAX_TEXTURE_LOD_BIAS 0x84FD
#define GL_TEXTURE_LOD_BIAS 0x8501
#define GL_INCR_WRAP 0x8507
#define GL_DECR_WRAP 0x8508
#define GL_TEXTURE_DEPTH_SIZE 0x884A
#define GL_TEXTURE_COMPARE_MODE 0x884C
#define GL_TEXTURE_COMPARE_FUNC 0x884D
#define GL_FUNC_ADD 0x8006
#define GL_FUNC_SUBTRACT 0x800A
#define GL_FUNC_REVERSE_SUBTRACT 0x800B
#define GL_MIN 0x8007
#define GL_MAX 0x8008
#define GL_CONSTANT_COLOR 0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR 0x8002
#define GL_CONSTANT_ALPHA 0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA 0x8004

/* GL_VERSION_1_5 */

#define GL_BUFFER_SIZE 0x8764
#define GL_BUFFER_USAGE 0x8765
#define GL_QUERY_COUNTER_BITS 0x8864
#define GL_CURRENT_QUERY 0x8865
#define GL_QUERY_RESULT 0x8866
#define GL_QUERY_RESULT_AVAILABLE 0x8867
#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_ARRAY_BUFFER_BINDING 0x8894
#define GL_ELEMENT_ARRAY_BUFFER_BINDING 0x8895
#define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING 0x889F
#define GL_READ_ONLY 0x88B8
#define GL_WRITE_ONLY 0x88B9
#define GL_READ_WRITE 0x88BA
#define GL_BUFFER_ACCESS 0x88BB
#define GL_BUFFER_MAPPED 0x88BC
#define GL_BUFFER_MAP_POINTER 0x88BD
#define GL_STREAM_DRAW 0x88E0
#define GL_STREAM_READ 0x88E1
#define GL_STREAM_COPY 0x88E2
#define GL_STATIC_DRAW 0x88E4
#define GL_STATIC_READ 0x88E5
#define GL_STATIC_COPY 0x88E6
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_DYNAMIC_READ 0x88E9
#define GL_DYNAMIC_COPY 0x88EA
#define GL_SAMPLES_PASSED 0x8914
#define GL_SRC1_ALPHA 0x8589

/* GL_VERSION_2_0 */

#define GL_BLEND_EQUATION_RGB 0x8009
#define GL_VERTEX_ATTRIB_ARRAY_ENABLED 0x8622
#define GL_VERTEX_ATTRIB_ARRAY_SIZE 0x8623
#define GL_VERTEX_ATTRIB_ARRAY_STRIDE 0x8624
#define GL_VERTEX_ATTRIB_ARRAY_TYPE 0x8625
#define GL_CURRENT_VERTEX_ATTRIB 0x8626
#define GL_VERTEX_PROGRAM_POINT_SIZE 0x8642
#define GL_VERTEX_ATTRIB_ARRAY_POINTER 0x8645
#define GL_STENCIL_BACK_FUNC 0x8800
#define GL_STENCIL_BACK_FAIL 0x8801
#define GL_STENCIL_BACK_PASS_DEPTH_FAIL 0x8802
#define GL_STENCIL_BACK_PASS_DEPTH_PASS 0x8803
#define GL_MAX_DRAW_BUFFERS 0x8824
#define GL_DRAW_BUFFER0 0x8825
#define GL_DRAW_BUFFER1 0x8826
#define GL_DRAW_BUFFER2 0x8827
#define GL_DRAW_BUFFER3 0x8828
#define GL_DRAW_BUFFER4 0x8829
#define GL_DRAW_BUFFER5 0x882A
#define GL_DRAW_BUFFER6 0x882B
#define GL_DRAW_BUFFER7 0x882C
#define GL_DRAW_BUFFER8 0x882D
#define GL_DRAW_BUFFER9 0x882E
#define GL_DRAW_BUFFER10 0x882F
#define GL_DRAW_BUFFER11 0x8830
#define GL_DRAW_BUFFER12 0x8831
#define GL_DRAW_BUFFER13 0x8832
#define GL_DRAW_BUFFER14 0x8833
#define GL_DRAW_BUFFER15 0x8834
#define GL_BLEND_EQUATION_ALPHA 0x883D
#define GL_MAX_VERTEX_ATTRIBS 0x8869
#define GL_VERTEX_ATTRIB_ARRAY_NORMALIZED 0x886A
#define GL_MAX_TEXTURE_IMAGE_UNITS 0x8872
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_MAX_FRAGMENT_UNIFORM_COMPONENTS 0x8B49
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS 0x8B4A
#define GL_MAX_VARYING_FLOATS 0x8B4B
#define GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS 0x8B4C
#define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS 0x8B4D
#define GL_SHADER_TYPE 0x8B4F
#define GL_FLOAT_VEC2 0x8B50
#define GL_FLOAT_VEC3 0x8B51
#define GL_FLOAT_VEC4 0x8B52
#define GL_INT_VEC2 0x8B53
#define GL_INT_VEC3 0x8B54
#define GL_INT_VEC4 0x8B55
#define GL_BOOL 0x8B56
#define GL_BOOL_VEC2 0x8B57
#define GL_BOOL_VEC3 0x8B58
#define GL_BOOL_VEC4 0x8B59
#define GL_FLOAT_MAT2 0x8B5A
#define GL_FLOAT_MAT3 0x8B5B
#define GL_FLOAT_MAT4 0x8B5C
#define GL_SAMPLER_1D 0x8B5D
#define GL_SAMPLER_2D 0x8B5E
#define GL_SAMPLER_3D 0x8B5F
#define GL_SAMPLER_CUBE 0x8B60
#define GL_SAMPLER_1D_SHADOW 0x8B61
#define GL_SAMPLER_2D_SHADOW 0x8B62
#define GL_DELETE_STATUS 0x8B80
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_VALIDATE_STATUS 0x8B83
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_ATTACHED_SHADERS 0x8B85
#define GL_ACTIVE_UNIFORMS 0x8B86
#define GL_ACTIVE_UNIFORM_MAX_LENGTH 0x8B87
#define GL_SHADER_SOURCE_LENGTH 0x8B88
#define GL_ACTIVE_ATTRIBUTES 0x8B89
#define GL_ACTIVE_ATTRIBUTE_MAX_LENGTH 0x8B8A
#define GL_FRAGMENT_SHADER_DERIVATIVE_HINT 0x8B8B
#define GL_SHADING_LANGUAGE_VERSION 0x8B8C
#define GL_CURRENT_PROGRAM 0x8B8D
#define GL_POINT_SPRITE_COORD_ORIGIN 0x8CA0
#define GL_LOWER_LEFT 0x8CA1
#define GL_UPPER_LEFT 0x8CA2
#define GL_STENCIL_BACK_REF 0x8CA3
#define GL_STENCIL_BACK_VALUE_MASK 0x8CA4
#define GL_STENCIL_BACK_WRITEMASK 0x8CA5

/* GL_VERSION_2_1 */

#define GL_PIXEL_PACK_BUFFER 0x88EB
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#define GL_PIXEL_PACK_BUFFER_BINDING 0x88ED
#define GL_PIXEL_UNPACK_BUFFER_BINDING 0x88EF
#define GL_FLOAT_MAT2x3 0x8B65
#define GL_FLOAT_MAT2x4 0x8B66
#define GL_FLOAT_MAT3x2 0x8B67
#define GL_FLOAT_MAT3x4 0x8B68
#define GL_FLOAT_MAT4x2 0x8B69
#define GL_FLOAT_MAT4x3 0x8B6A
#define GL_SRGB 0x8C40
#define GL_SRGB8 0x8C41
#define GL_SRGB_ALPHA 0x8C42
#define GL_SRGB8_ALPHA8 0x8C43
#define GL_COMPRESSED_SRGB 0x8C48
#define GL_COMPRESSED_SRGB_ALPHA 0x8C49

/* GL_VERSION_3_0 */

#define GL_COMPARE_REF_TO_TEXTURE 0x884E
#define GL_CLIP_DISTANCE0 0x3000
#define GL_CLIP_DISTANCE1 0x3001
#define GL_CLIP_DISTANCE2 0x3002
#define GL_CLIP_DISTANCE3 0x3003
#define GL_CLIP_DISTANCE4 0x3004
#define GL_CLIP_DISTANCE5 0x3005
#define GL_CLIP_DISTANCE6 0x3006
#define GL_CLIP_DISTANCE7 0x3007
#define GL_MAX_CLIP_DISTANCES 0x0D32
#define GL_MAJOR_VERSION 0x821B
#define GL_MINOR_VERSION 0x821C
#define GL_NUM_EXTENSIONS 0x821D
#define GL_CONTEXT_FLAGS 0x821E
#define GL_COMPRESSED_RED 0x8225
#define GL_COMPRESSED_RG 0x8226
#define GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT 0x00000001
#define GL_RGBA32F 0x8814
#define GL_RGB32F 0x8815
#define GL_RGBA16F 0x881A
#define GL_RGB16F 0x881B
#define GL_VERTEX_ATTRIB_ARRAY_INTEGER 0x88FD
#define GL_MAX_ARRAY_TEXTURE_LAYERS 0x88FF
#define GL_MIN_PROGRAM_TEXEL_OFFSET 0x8904
#define GL_MAX_PROGRAM_TEXEL_OFFSET 0x8905
#define GL_CLAMP_READ_COLOR 0x891C
#define GL_FIXED_ONLY 0x891D
#define GL_MAX_VARYING_COMPONENTS 0x8B4B
#define GL_TEXTURE_1D_ARRAY 0x8C18
#define GL_PROXY_TEXTURE_1D_ARRAY 0x8C19
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#define GL_PROXY_TEXTURE_2D_ARRAY 0x8C1B
#define GL_TEXTURE_BINDING_1D_ARRAY 0x8C1C
#define GL_TEXTURE_BINDING_2D_ARRAY 0x8C1D
#define GL_R11F_G11F_B10F 0x8C3A
#define GL_UNSIGNED_INT_10F_11F_11F_REV 0x8C3B
#define GL_RGB9_E5 0x8C3D
#define GL_UNSIGNED_INT_5_9_9_9_REV 0x8C3E
#define GL_TEXTURE_SHARED_SIZE 0x8C3F
#define GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH 0x8C76
#define GL_TRANSFORM_FEEDBACK_BUFFER_MODE 0x8C7F
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS 0x8C80
#define GL_TRANSFORM_FEEDBACK_VARYINGS 0x8C83
#define GL_TRANSFORM_FEEDBACK_BUFFER_START 0x8C84
#define GL_TRANSFORM_FEEDBACK_BUFFER_SIZE 0x8C85
#define GL_PRIMITIVES_GENERATED 0x8C87
#define GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN 0x8C88
#define GL_RASTERIZER_DISCARD 0x8C89
#define GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS 0x8C8A
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS 0x8C8B
#define GL_INTERLEAVED_ATTRIBS 0x8C8C
#define GL_SEPARATE_ATTRIBS 0x8C8D
#define GL_TRANSFORM_FEEDBACK_BUFFER 0x8C8E
#define GL_TRANSFORM_FEEDBACK_BUFFER_BINDING 0x8C8F
#define GL_RGBA32UI 0x8D70
#define GL_RGB32UI 0x8D71
#define GL_RGBA16UI 0x8D76
#define GL_RGB16UI 0x8D77
#define GL_RGBA8UI 0x8D7C
#define GL_RGB8UI 0x8D7D
#define GL_RGBA32I 0x8D82
#define GL_RGB32I 0x8D83
#define GL_RGBA16I 0x8D88
#define GL_RGB16I 0x8D89
#define GL_RGBA8I 0x8D8E
#define GL_RGB8I 0x8D8F
#define GL_RED_INTEGER 0x8D94
#define GL_GREEN_INTEGER 0x8D95
#define GL_BLUE_INTEGER 0x8D96
#define GL_RGB_INTEGER 0x8D98
#define GL_RGBA_INTEGER 0x8D99
#define GL_BGR_INTEGER 0x8D9A
#define GL_BGRA_INTEGER 0x8D9B
#define GL_SAMPLER_1D_ARRAY 0x8DC0
#define GL_SAMPLER_2D_ARRAY 0x8DC1
#define GL_SAMPLER_1D_ARRAY_SHADOW 0x8DC3
#define GL_SAMPLER_2D_ARRAY_SHADOW 0x8DC4
#define GL_SAMPLER_CUBE_SHADOW 0x8DC5
#define GL_UNSIGNED_INT_VEC2 0x8DC6
#define GL_UNSIGNED_INT_VEC3 0x8DC7
#define GL_UNSIGNED_INT_VEC4 0x8DC8
#define GL_INT_SAMPLER_1D 0x8DC9
#define GL_INT_SAMPLER_2D 0x8DCA
#define GL_INT_SAMPLER_3D 0x8DCB
#define GL_INT_SAMPLER_CUBE 0x8DCC
#define GL_INT_SAMPLER_1D_ARRAY 0x8DCE
#define GL_INT_SAMPLER_2D_ARRAY 0x8DCF
#define GL_UNSIGNED_INT_SAMPLER_1D 0x8DD1
#define GL_UNSIGNED_INT_SAMPLER_2D 0x8DD2
#define GL_UNSIGNED_INT_SAMPLER_3D 0x8DD3
#define GL_UNSIGNED_INT_SAMPLER_CUBE 0x8DD4
#define GL_UNSIGNED_INT_SAMPLER_1D_ARRAY 0x8DD6
#define GL_UNSIGNED_INT_SAMPLER_2D_ARRAY 0x8DD7
#define GL_QUERY_WAIT 0x8E13
#define GL_QUERY_NO_WAIT 0x8E14
#define GL_QUERY_BY_REGION_WAIT 0x8E15
#define GL_QUERY_BY_REGION_NO_WAIT 0x8E16
#define GL_BUFFER_ACCESS_FLAGS 0x911F
#define GL_BUFFER_MAP_LENGTH 0x9120
#define GL_BUFFER_MAP_OFFSET 0x9121
#define GL_DEPTH_COMPONENT32F 0x8CAC
#define GL_DEPTH32F_STENCIL8 0x8CAD
#define GL_FLOAT_32_UNSIGNED_INT_24_8_REV 0x8DAD
#define GL_INVALID_FRAMEBUFFER_OPERATION 0x0506
#define GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING 0x8210
#define GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE 0x8211
#define GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE 0x8212
#define GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE 0x8213
#define GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE 0x8214
#define GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE 0x8215
#define GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE 0x8216
#define GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE 0x8217
#define GL_FRAMEBUFFER_DEFAULT 0x8218
#define GL_FRAMEBUFFER_UNDEFINED 0x8219
#define GL_DEPTH_STENCIL_ATTACHMENT 0x821A
#define GL_MAX_RENDERBUFFER_SIZE 0x84E8
#define GL_DEPTH_STENCIL 0x84F9
#define GL_UNSIGNED_INT_24_8 0x84FA
#define GL_DEPTH24_STENCIL8 0x88F0
#define GL_TEXTURE_STENCIL_SIZE 0x88F1
#define GL_TEXTURE_RED_TYPE 0x8C10
#define GL_TEXTURE_GREEN_TYPE 0x8C11
#define GL_TEXTURE_BLUE_TYPE 0x8C12
#define GL_TEXTURE_ALPHA_TYPE 0x8C13
#define GL_TEXTURE_DEPTH_TYPE 0x8C16
#define GL_UNSIGNED_NORMALIZED 0x8C17
#define GL_FRAMEBUFFER_BINDING 0x8CA6
#define GL_DRAW_FRAMEBUFFER_BINDING 0x8CA6
#define GL_RENDERBUFFER_BINDING 0x8CA7
#define GL_READ_FRAMEBUFFER 0x8CA8
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#define GL_READ_FRAMEBUFFER_BINDING 0x8CAA
#define GL_RENDERBUFFER_SAMPLES 0x8CAB
#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE 0x8CD0
#define GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME 0x8CD1
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL 0x8CD2
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE 0x8CD3
#define GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER 0x8CD4
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT 0x8CD6
#define GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT 0x8CD7
#define GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER 0x8CDB
#define GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER 0x8CDC
#define GL_FRAMEBUFFER_UNSUPPORTED 0x8CDD
#define GL_MAX_COLOR_ATTACHMENTS 0x8CDF
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_COLOR_ATTACHMENT1 0x8CE1
#define GL_COLOR_ATTACHMENT2 0x8CE2
#define GL_COLOR_ATTACHMENT3 0x8CE3
#define GL_COLOR_ATTACHMENT4 0x8CE4
#define GL_COLOR_ATTACHMENT5 0x8CE5
#define GL_COLOR_ATTACHMENT6 0x8CE6
#define GL_COLOR_ATTACHMENT7 0x8CE7
#define GL_COLOR_ATTACHMENT8 0x8CE8
#define GL_COLOR_ATTACHMENT9 0x8CE9
#define GL_COLOR_ATTACHMENT10 0x8CEA
#define GL_COLOR_ATTACHMENT11 0x8CEB
#define GL_COLOR_ATTACHMENT12 0x8CEC
#define GL_COLOR_ATTACHMENT13 0x8CED
#define GL_COLOR_ATTACHMENT14 0x8CEE
#define GL_COLOR_ATTACHMENT15 0x8CEF
#define GL_DEPTH_ATTACHMENT 0x8D00
#define GL_STENCIL_ATTACHMENT 0x8D20
#define GL_FRAMEBUFFER 0x8D40
#define GL_RENDERBUFFER 0x8D41
#define GL_RENDERBUFFER_WIDTH 0x8D42
#define GL_RENDERBUFFER_HEIGHT 0x8D43
#define GL_RENDERBUFFER_INTERNAL_FORMAT 0x8D44
#define GL_STENCIL_INDEX1 0x8D46
#define GL_STENCIL_INDEX4 0x8D47
#define GL_STENCIL_INDEX8 0x8D48
#define GL_STENCIL_INDEX16 0x8D49
#define GL_RENDERBUFFER_RED_SIZE 0x8D50
#define GL_RENDERBUFFER_GREEN_SIZE 0x8D51
#define GL_RENDERBUFFER_BLUE_SIZE 0x8D52
#define GL_RENDERBUFFER_ALPHA_SIZE 0x8D53
#define GL_RENDERBUFFER_DEPTH_SIZE 0x8D54
#define GL_RENDERBUFFER_STENCIL_SIZE 0x8D55
#define GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE 0x8D56
#define GL_MAX_SAMPLES 0x8D57
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#define GL_HALF_FLOAT 0x140B
#define GL_MAP_READ_BIT 0x0001
#define GL_MAP_WRITE_BIT 0x0002
#define GL_MAP_INVALIDATE_RANGE_BIT 0x0004
#define GL_MAP_INVALIDATE_BUFFER_BIT 0x0008
#define GL_MAP_FLUSH_EXPLICIT_BIT 0x0010
#define GL_MAP_UNSYNCHRONIZED_BIT 0x0020
#define GL_COMPRESSED_RED_RGTC1 0x8DBB
#define GL_COMPRESSED_SIGNED_RED_RGTC1 0x8DBC
#define GL_COMPRESSED_RG_RGTC2 0x8DBD
#define GL_COMPRESSED_SIGNED_RG_RGTC2 0x8DBE
#define GL_RG 0x8227
#define GL_RG_INTEGER 0x8228
#define GL_R8 0x8229
#define GL_R16 0x822A
#define GL_RG8 0x822B
#define GL_RG16 0x822C
#define GL_R16F 0x822D
#define GL_R32F 0x822E
#define GL_RG16F 0x822F
#define GL_RG32F 0x8230
#define GL_R8I 0x8231
#define GL_R8UI 0x8232
#define GL_R16I 0x8233
#define GL_R16UI 0x8234
#define GL_R32I 0x8235
#define GL_R32UI 0x8236
#define GL_RG8I 0x8237
#define GL_RG8UI 0x8238
#define GL_RG16I 0x8239
#define GL_RG16UI 0x823A
#define GL_RG32I 0x823B
#define GL_RG32UI 0x823C
#define GL_VERTEX_ARRAY_BINDING 0x85B5

/* GL_VERSION_3_1 */

#define GL_SAMPLER_2D_RECT 0x8B63
#define GL_SAMPLER_2D_RECT_SHADOW 0x8B64
#define GL_SAMPLER_BUFFER 0x8DC2
#define GL_INT_SAMPLER_2D_RECT 0x8DCD
#define GL_INT_SAMPLER_BUFFER 0x8DD0
#define GL_UNSIGNED_INT_SAMPLER_2D_RECT 0x8DD5
#define GL_UNSIGNED_INT_SAMPLER_BUFFER 0x8DD8
#define GL_TEXTURE_BUFFER 0x8C2A
#define GL_MAX_TEXTURE_BUFFER_SIZE 0x8C2B
#define GL_TEXTURE_BINDING_BUFFER 0x8C2C
#define GL_TEXTURE_BUFFER_DATA_STORE_BINDING 0x8C2D
#define GL_TEXTURE_RECTANGLE 0x84F5
#define GL_TEXTURE_BINDING_RECTANGLE 0x84F6
#define GL_PROXY_TEXTURE_RECTANGLE 0x84F7
#define GL_MAX_RECTANGLE_TEXTURE_SIZE 0x84F8
#define GL_R8_SNORM 0x8F94
#define GL_RG8_SNORM 0x8F95
#define GL_RGB8_SNORM 0x8F96
#define GL_RGBA8_SNORM 0x8F97
#define GL_R16_SNORM 0x8F98
#define GL_RG16_SNORM 0x8F99
#define GL_RGB16_SNORM 0x8F9A
#define GL_RGBA16_SNORM 0x8F9B
#define GL_SIGNED_NORMALIZED 0x8F9C
#define GL_PRIMITIVE_RESTART 0x8F9D
#define GL_PRIMITIVE_RESTART_INDEX 0x8F9E
#define GL_COPY_READ_BUFFER 0x8F36
#define GL_COPY_WRITE_BUFFER 0x8F37
#define GL_UNIFORM_BUFFER 0x8A11
#define GL_UNIFORM_BUFFER_BINDING 0x8A28
#define GL_UNIFORM_BUFFER_START 0x8A29
#define GL_UNIFORM_BUFFER_SIZE 0x8A2A
#define GL_MAX_VERTEX_UNIFORM_BLOCKS 0x8A2B
#define GL_MAX_GEOMETRY_UNIFORM_BLOCKS 0x8A2C
#define GL_MAX_FRAGMENT_UNIFORM_BLOCKS 0x8A2D
#define GL_MAX_COMBINED_UNIFORM_BLOCKS 0x8A2E
#define GL_MAX_UNIFORM_BUFFER_BINDINGS 0x8A2F
#define GL_MAX_UNIFORM_BLOCK_SIZE 0x8A30
#define GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS 0x8A31
#define GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS 0x8A32
#define GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS 0x8A33
#define GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT 0x8A34
#define GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH 0x8A35
#define GL_ACTIVE_UNIFORM_BLOCKS 0x8A36
#define GL_UNIFORM_TYPE 0x8A37
#define GL_UNIFORM_SIZE 0x8A38
#define GL_UNIFORM_NAME_LENGTH 0x8A39
#define GL_UNIFORM_BLOCK_INDEX 0x8A3A
#define GL_UNIFORM_OFFSET 0x8A3B
#define GL_UNIFORM_ARRAY_STRIDE 0x8A3C
#define GL_UNIFORM_MATRIX_STRIDE 0x8A3D
#define GL_UNIFORM_IS_ROW_MAJOR 0x8A3E
#define GL_UNIFORM_BLOCK_BINDING 0x8A3F
#define GL_UNIFORM_BLOCK_DATA_SIZE 0x8A40
#define GL_UNIFORM_BLOCK_NAME_LENGTH 0x8A41
#define GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS 0x8A42
#define GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES 0x8A43
#define GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER 0x8A44
#define GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER 0x8A45
#define GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER 0x8A46
#define GL_INVALID_INDEX 0xFFFFFFFFu

/* --------------------------- FUNCTION PROTOTYPES --------------------------- */

    
/* GL_VERSION_1_0 */

GLAPI void APIENTRY glCullFace (GLenum mode);
GLAPI void APIENTRY glFrontFace (GLenum mode);
GLAPI void APIENTRY glHint (GLenum target, GLenum mode);
GLAPI void APIENTRY glLineWidth (GLfloat width);
GLAPI void APIENTRY glPointSize (GLfloat size);
GLAPI void APIENTRY glPolygonMode (GLenum face, GLenum mode);
GLAPI void APIENTRY glScissor (GLint x, GLint y, GLsizei width, GLsizei height);
GLAPI void APIENTRY glTexParameterf (GLenum target, GLenum pname, GLfloat param);
GLAPI void APIENTRY glTexParameterfv (GLenum target, GLenum pname, const GLfloat * params);
GLAPI void APIENTRY glTexParameteri (GLenum target, GLenum pname, GLint param);
GLAPI void APIENTRY glTexParameteriv (GLenum target, GLenum pname, const GLint * params);
GLAPI void APIENTRY glTexImage1D (GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void * pixels);
GLAPI void APIENTRY glTexImage2D (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void * pixels);
GLAPI void APIENTRY glDrawBuffer (GLenum buf);
GLAPI void APIENTRY glClear (GLbitfield mask);
GLAPI void APIENTRY glClearColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
GLAPI void APIENTRY glClearStencil (GLint s);
GLAPI void APIENTRY glClearDepth (GLdouble depth);
GLAPI void APIENTRY glStencilMask (GLuint mask);
GLAPI void APIENTRY glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
GLAPI void APIENTRY glDepthMask (GLboolean flag);
GLAPI void APIENTRY glDisable (GLenum cap);
GLAPI void APIENTRY glEnable (GLenum cap);
GLAPI void APIENTRY glFinish (void);
GLAPI void APIENTRY glFlush (void);
GLAPI void APIENTRY glBlendFunc (GLenum sfactor, GLenum dfactor);
GLAPI void APIENTRY glLogicOp (GLenum opcode);
GLAPI void APIENTRY glStencilFunc (GLenum func, GLint ref, GLuint mask);
GLAPI void APIENTRY glStencilOp (GLenum fail, GLenum zfail, GLenum zpass);
GLAPI void APIENTRY glDepthFunc (GLenum func);
GLAPI void APIENTRY glPixelStoref (GLenum pname, GLfloat param);
GLAPI void APIENTRY glPixelStorei (GLenum pname, GLint param);
GLAPI void APIENTRY glReadBuffer (GLenum src);
GLAPI void APIENTRY glReadPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void * pixels);
GLAPI void APIENTRY glGetBooleanv (GLenum pname, GLboolean * data);
GLAPI void APIENTRY glGetDoublev (GLenum pname, GLdouble * data);
GLAPI GLenum APIENTRY glGetError (void);
GLAPI void APIENTRY glGetFloatv (GLenum pname, GLfloat * data);
GLAPI void APIENTRY glGetIntegerv (GLenum pname, GLint * data);
GLAPI const GLubyte * APIENTRY glGetString (GLenum name);
GLAPI void APIENTRY glGetTexImage (GLenum target, GLint level, GLenum format, GLenum type, void * pixels);
GLAPI void APIENTRY glGetTexParameterfv (GLenum target, GLenum pname, GLfloat * params);
GLAPI void APIENTRY glGetTexParameteriv (GLenum target, GLenum pname, GLint * params);
GLAPI void APIENTRY glGetTexLevelParameterfv (GLenum target, GLint level, GLenum pname, GLfloat * params);
GLAPI void APIENTRY glGetTexLevelParameteriv (GLenum target, GLint level, GLenum pname, GLint * params);
GLAPI GLboolean APIENTRY glIsEnabled (GLenum cap);
GLAPI void APIENTRY glDepthRange (GLdouble near, GLdouble far);
GLAPI void APIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height);
    
/* GL_VERSION_1_1 */

GLAPI void APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count);
GLAPI void APIENTRY glDrawElements (GLenum mode, GLsizei count, GLenum type, const void * indices);
GLAPI void APIENTRY glPolygonOffset (GLfloat factor, GLfloat units);
GLAPI void APIENTRY glCopyTexImage1D (GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border);
GLAPI void APIENTRY glCopyTexImage2D (GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border);
GLAPI void APIENTRY glCopyTexSubImage1D (GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
GLAPI void APIENTRY glCopyTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
GLAPI void APIENTRY glTexSubImage1D (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void * pixels);
GLAPI void APIENTRY glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void * pixels);
GLAPI void APIENTRY glBindTexture (GLenum target, GLuint texture);
GLAPI void APIENTRY glDeleteTextures (GLsizei n, const GLuint * textures);
GLAPI void APIENTRY glGenTextures (GLsizei n, GLuint * textures);
GLAPI GLboolean APIENTRY glIsTexture (GLuint texture);

    
/* GL_VERSION_1_2 */
  
typedef void (APIENTRY PFNGLDRAWRANGEELEMENTS_PROC (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void * indices));
typedef void (APIENTRY PFNGLTEXIMAGE3D_PROC (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void * pixels));
typedef void (APIENTRY PFNGLTEXSUBIMAGE3D_PROC (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void * pixels));
typedef void (APIENTRY PFNGLCOPYTEXSUBIMAGE3D_PROC (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height));
    
/* GL_VERSION_1_3 */
  
typedef void (APIENTRY PFNGLACTIVETEXTURE_PROC (GLenum texture));
typedef void (APIENTRY PFNGLSAMPLECOVERAGE_PROC (GLfloat value, GLboolean invert));
typedef void (APIENTRY PFNGLCOMPRESSEDTEXIMAGE3D_PROC (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void * data));
typedef void (APIENTRY PFNGLCOMPRESSEDTEXIMAGE2D_PROC (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void * data));
typedef void (APIENTRY PFNGLCOMPRESSEDTEXIMAGE1D_PROC (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void * data));
typedef void (APIENTRY PFNGLCOMPRESSEDTEXSUBIMAGE3D_PROC (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void * data));
typedef void (APIENTRY PFNGLCOMPRESSEDTEXSUBIMAGE2D_PROC (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void * data));
typedef void (APIENTRY PFNGLCOMPRESSEDTEXSUBIMAGE1D_PROC (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void * data));
typedef void (APIENTRY PFNGLGETCOMPRESSEDTEXIMAGE_PROC (GLenum target, GLint level, void * img));
    
/* GL_VERSION_1_4 */
  
typedef void (APIENTRY PFNGLBLENDFUNCSEPARATE_PROC (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha));
typedef void (APIENTRY PFNGLMULTIDRAWARRAYS_PROC (GLenum mode, const GLint * first, const GLsizei * count, GLsizei drawcount));
typedef void (APIENTRY PFNGLMULTIDRAWELEMENTS_PROC (GLenum mode, const GLsizei * count, GLenum type, const void *const* indices, GLsizei drawcount));
typedef void (APIENTRY PFNGLPOINTPARAMETERF_PROC (GLenum pname, GLfloat param));
typedef void (APIENTRY PFNGLPOINTPARAMETERFV_PROC (GLenum pname, const GLfloat * params));
typedef void (APIENTRY PFNGLPOINTPARAMETERI_PROC (GLenum pname, GLint param));
typedef void (APIENTRY PFNGLPOINTPARAMETERIV_PROC (GLenum pname, const GLint * params));
typedef void (APIENTRY PFNGLBLENDCOLOR_PROC (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha));
typedef void (APIENTRY PFNGLBLENDEQUATION_PROC (GLenum mode));
    
/* GL_VERSION_1_5 */
  
typedef void (APIENTRY PFNGLGENQUERIES_PROC (GLsizei n, GLuint * ids));
typedef void (APIENTRY PFNGLDELETEQUERIES_PROC (GLsizei n, const GLuint * ids));
typedef GLboolean (APIENTRY PFNGLISQUERY_PROC (GLuint id));
typedef void (APIENTRY PFNGLBEGINQUERY_PROC (GLenum target, GLuint id));
typedef void (APIENTRY PFNGLENDQUERY_PROC (GLenum target));
typedef void (APIENTRY PFNGLGETQUERYIV_PROC (GLenum target, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETQUERYOBJECTIV_PROC (GLuint id, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETQUERYOBJECTUIV_PROC (GLuint id, GLenum pname, GLuint * params));
typedef void (APIENTRY PFNGLBINDBUFFER_PROC (GLenum target, GLuint buffer));
typedef void (APIENTRY PFNGLDELETEBUFFERS_PROC (GLsizei n, const GLuint * buffers));
typedef void (APIENTRY PFNGLGENBUFFERS_PROC (GLsizei n, GLuint * buffers));
typedef GLboolean (APIENTRY PFNGLISBUFFER_PROC (GLuint buffer));
typedef void (APIENTRY PFNGLBUFFERDATA_PROC (GLenum target, GLsizeiptr size, const void * data, GLenum usage));
typedef void (APIENTRY PFNGLBUFFERSUBDATA_PROC (GLenum target, GLintptr offset, GLsizeiptr size, const void * data));
typedef void (APIENTRY PFNGLGETBUFFERSUBDATA_PROC (GLenum target, GLintptr offset, GLsizeiptr size, void * data));
typedef void * (APIENTRY PFNGLMAPBUFFER_PROC (GLenum target, GLenum access));
typedef GLboolean (APIENTRY PFNGLUNMAPBUFFER_PROC (GLenum target));
typedef void (APIENTRY PFNGLGETBUFFERPARAMETERIV_PROC (GLenum target, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETBUFFERPOINTERV_PROC (GLenum target, GLenum pname, void ** params));
    
/* GL_VERSION_2_0 */
  
typedef void (APIENTRY PFNGLBLENDEQUATIONSEPARATE_PROC (GLenum modeRGB, GLenum modeAlpha));
typedef void (APIENTRY PFNGLDRAWBUFFERS_PROC (GLsizei n, const GLenum * bufs));
typedef void (APIENTRY PFNGLSTENCILOPSEPARATE_PROC (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass));
typedef void (APIENTRY PFNGLSTENCILFUNCSEPARATE_PROC (GLenum face, GLenum func, GLint ref, GLuint mask));
typedef void (APIENTRY PFNGLSTENCILMASKSEPARATE_PROC (GLenum face, GLuint mask));
typedef void (APIENTRY PFNGLATTACHSHADER_PROC (GLuint program, GLuint shader));
typedef void (APIENTRY PFNGLBINDATTRIBLOCATION_PROC (GLuint program, GLuint index, const GLchar * name));
typedef void (APIENTRY PFNGLCOMPILESHADER_PROC (GLuint shader));
typedef GLuint (APIENTRY PFNGLCREATEPROGRAM_PROC (void));
typedef GLuint (APIENTRY PFNGLCREATESHADER_PROC (GLenum type));
typedef void (APIENTRY PFNGLDELETEPROGRAM_PROC (GLuint program));
typedef void (APIENTRY PFNGLDELETESHADER_PROC (GLuint shader));
typedef void (APIENTRY PFNGLDETACHSHADER_PROC (GLuint program, GLuint shader));
typedef void (APIENTRY PFNGLDISABLEVERTEXATTRIBARRAY_PROC (GLuint index));
typedef void (APIENTRY PFNGLENABLEVERTEXATTRIBARRAY_PROC (GLuint index));
typedef void (APIENTRY PFNGLGETACTIVEATTRIB_PROC (GLuint program, GLuint index, GLsizei bufSize, GLsizei * length, GLint * size, GLenum * type, GLchar * name));
typedef void (APIENTRY PFNGLGETACTIVEUNIFORM_PROC (GLuint program, GLuint index, GLsizei bufSize, GLsizei * length, GLint * size, GLenum * type, GLchar * name));
typedef void (APIENTRY PFNGLGETATTACHEDSHADERS_PROC (GLuint program, GLsizei maxCount, GLsizei * count, GLuint * shaders));
typedef GLint (APIENTRY PFNGLGETATTRIBLOCATION_PROC (GLuint program, const GLchar * name));
typedef void (APIENTRY PFNGLGETPROGRAMIV_PROC (GLuint program, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETPROGRAMINFOLOG_PROC (GLuint program, GLsizei bufSize, GLsizei * length, GLchar * infoLog));
typedef void (APIENTRY PFNGLGETSHADERIV_PROC (GLuint shader, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETSHADERINFOLOG_PROC (GLuint shader, GLsizei bufSize, GLsizei * length, GLchar * infoLog));
typedef void (APIENTRY PFNGLGETSHADERSOURCE_PROC (GLuint shader, GLsizei bufSize, GLsizei * length, GLchar * source));
typedef GLint (APIENTRY PFNGLGETUNIFORMLOCATION_PROC (GLuint program, const GLchar * name));
typedef void (APIENTRY PFNGLGETUNIFORMFV_PROC (GLuint program, GLint location, GLfloat * params));
typedef void (APIENTRY PFNGLGETUNIFORMIV_PROC (GLuint program, GLint location, GLint * params));
typedef void (APIENTRY PFNGLGETVERTEXATTRIBDV_PROC (GLuint index, GLenum pname, GLdouble * params));
typedef void (APIENTRY PFNGLGETVERTEXATTRIBFV_PROC (GLuint index, GLenum pname, GLfloat * params));
typedef void (APIENTRY PFNGLGETVERTEXATTRIBIV_PROC (GLuint index, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETVERTEXATTRIBPOINTERV_PROC (GLuint index, GLenum pname, void ** pointer));
typedef GLboolean (APIENTRY PFNGLISPROGRAM_PROC (GLuint program));
typedef GLboolean (APIENTRY PFNGLISSHADER_PROC (GLuint shader));
typedef void (APIENTRY PFNGLLINKPROGRAM_PROC (GLuint program));
typedef void (APIENTRY PFNGLSHADERSOURCE_PROC (GLuint shader, GLsizei count, const GLchar *const* string, const GLint * length));
typedef void (APIENTRY PFNGLUSEPROGRAM_PROC (GLuint program));
typedef void (APIENTRY PFNGLUNIFORM1F_PROC (GLint location, GLfloat v0));
typedef void (APIENTRY PFNGLUNIFORM2F_PROC (GLint location, GLfloat v0, GLfloat v1));
typedef void (APIENTRY PFNGLUNIFORM3F_PROC (GLint location, GLfloat v0, GLfloat v1, GLfloat v2));
typedef void (APIENTRY PFNGLUNIFORM4F_PROC (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3));
typedef void (APIENTRY PFNGLUNIFORM1I_PROC (GLint location, GLint v0));
typedef void (APIENTRY PFNGLUNIFORM2I_PROC (GLint location, GLint v0, GLint v1));
typedef void (APIENTRY PFNGLUNIFORM3I_PROC (GLint location, GLint v0, GLint v1, GLint v2));
typedef void (APIENTRY PFNGLUNIFORM4I_PROC (GLint location, GLint v0, GLint v1, GLint v2, GLint v3));
typedef void (APIENTRY PFNGLUNIFORM1FV_PROC (GLint location, GLsizei count, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORM2FV_PROC (GLint location, GLsizei count, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORM3FV_PROC (GLint location, GLsizei count, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORM4FV_PROC (GLint location, GLsizei count, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORM1IV_PROC (GLint location, GLsizei count, const GLint * value));
typedef void (APIENTRY PFNGLUNIFORM2IV_PROC (GLint location, GLsizei count, const GLint * value));
typedef void (APIENTRY PFNGLUNIFORM3IV_PROC (GLint location, GLsizei count, const GLint * value));
typedef void (APIENTRY PFNGLUNIFORM4IV_PROC (GLint location, GLsizei count, const GLint * value));
typedef void (APIENTRY PFNGLUNIFORMMATRIX2FV_PROC (GLint location, GLsizei count, GLboolean transpose, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORMMATRIX3FV_PROC (GLint location, GLsizei count, GLboolean transpose, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORMMATRIX4FV_PROC (GLint location, GLsizei count, GLboolean transpose, const GLfloat * value));
typedef void (APIENTRY PFNGLVALIDATEPROGRAM_PROC (GLuint program));
typedef void (APIENTRY PFNGLVERTEXATTRIB1D_PROC (GLuint index, GLdouble x));
typedef void (APIENTRY PFNGLVERTEXATTRIB1DV_PROC (GLuint index, const GLdouble * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB1F_PROC (GLuint index, GLfloat x));
typedef void (APIENTRY PFNGLVERTEXATTRIB1FV_PROC (GLuint index, const GLfloat * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB1S_PROC (GLuint index, GLshort x));
typedef void (APIENTRY PFNGLVERTEXATTRIB1SV_PROC (GLuint index, const GLshort * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB2D_PROC (GLuint index, GLdouble x, GLdouble y));
typedef void (APIENTRY PFNGLVERTEXATTRIB2DV_PROC (GLuint index, const GLdouble * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB2F_PROC (GLuint index, GLfloat x, GLfloat y));
typedef void (APIENTRY PFNGLVERTEXATTRIB2FV_PROC (GLuint index, const GLfloat * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB2S_PROC (GLuint index, GLshort x, GLshort y));
typedef void (APIENTRY PFNGLVERTEXATTRIB2SV_PROC (GLuint index, const GLshort * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB3D_PROC (GLuint index, GLdouble x, GLdouble y, GLdouble z));
typedef void (APIENTRY PFNGLVERTEXATTRIB3DV_PROC (GLuint index, const GLdouble * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB3F_PROC (GLuint index, GLfloat x, GLfloat y, GLfloat z));
typedef void (APIENTRY PFNGLVERTEXATTRIB3FV_PROC (GLuint index, const GLfloat * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB3S_PROC (GLuint index, GLshort x, GLshort y, GLshort z));
typedef void (APIENTRY PFNGLVERTEXATTRIB3SV_PROC (GLuint index, const GLshort * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4NBV_PROC (GLuint index, const GLbyte * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4NIV_PROC (GLuint index, const GLint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4NSV_PROC (GLuint index, const GLshort * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4NUB_PROC (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w));
typedef void (APIENTRY PFNGLVERTEXATTRIB4NUBV_PROC (GLuint index, const GLubyte * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4NUIV_PROC (GLuint index, const GLuint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4NUSV_PROC (GLuint index, const GLushort * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4BV_PROC (GLuint index, const GLbyte * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4D_PROC (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w));
typedef void (APIENTRY PFNGLVERTEXATTRIB4DV_PROC (GLuint index, const GLdouble * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4F_PROC (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w));
typedef void (APIENTRY PFNGLVERTEXATTRIB4FV_PROC (GLuint index, const GLfloat * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4IV_PROC (GLuint index, const GLint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4S_PROC (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w));
typedef void (APIENTRY PFNGLVERTEXATTRIB4SV_PROC (GLuint index, const GLshort * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4UBV_PROC (GLuint index, const GLubyte * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4UIV_PROC (GLuint index, const GLuint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIB4USV_PROC (GLuint index, const GLushort * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBPOINTER_PROC (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void * pointer));
    
/* GL_VERSION_2_1 */
  
typedef void (APIENTRY PFNGLUNIFORMMATRIX2X3FV_PROC (GLint location, GLsizei count, GLboolean transpose, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORMMATRIX3X2FV_PROC (GLint location, GLsizei count, GLboolean transpose, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORMMATRIX2X4FV_PROC (GLint location, GLsizei count, GLboolean transpose, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORMMATRIX4X2FV_PROC (GLint location, GLsizei count, GLboolean transpose, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORMMATRIX3X4FV_PROC (GLint location, GLsizei count, GLboolean transpose, const GLfloat * value));
typedef void (APIENTRY PFNGLUNIFORMMATRIX4X3FV_PROC (GLint location, GLsizei count, GLboolean transpose, const GLfloat * value));
    
/* GL_VERSION_3_0 */
  
typedef void (APIENTRY PFNGLCOLORMASKI_PROC (GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a));
typedef void (APIENTRY PFNGLGETBOOLEANI_V_PROC (GLenum target, GLuint index, GLboolean * data));
typedef void (APIENTRY PFNGLGETINTEGERI_V_PROC (GLenum target, GLuint index, GLint * data));
typedef void (APIENTRY PFNGLENABLEI_PROC (GLenum target, GLuint index));
typedef void (APIENTRY PFNGLDISABLEI_PROC (GLenum target, GLuint index));
typedef GLboolean (APIENTRY PFNGLISENABLEDI_PROC (GLenum target, GLuint index));
typedef void (APIENTRY PFNGLBEGINTRANSFORMFEEDBACK_PROC (GLenum primitiveMode));
typedef void (APIENTRY PFNGLENDTRANSFORMFEEDBACK_PROC (void));
typedef void (APIENTRY PFNGLBINDBUFFERRANGE_PROC (GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size));
typedef void (APIENTRY PFNGLBINDBUFFERBASE_PROC (GLenum target, GLuint index, GLuint buffer));
typedef void (APIENTRY PFNGLTRANSFORMFEEDBACKVARYINGS_PROC (GLuint program, GLsizei count, const GLchar *const* varyings, GLenum bufferMode));
typedef void (APIENTRY PFNGLGETTRANSFORMFEEDBACKVARYING_PROC (GLuint program, GLuint index, GLsizei bufSize, GLsizei * length, GLsizei * size, GLenum * type, GLchar * name));
typedef void (APIENTRY PFNGLCLAMPCOLOR_PROC (GLenum target, GLenum clamp));
typedef void (APIENTRY PFNGLBEGINCONDITIONALRENDER_PROC (GLuint id, GLenum mode));
typedef void (APIENTRY PFNGLENDCONDITIONALRENDER_PROC (void));
typedef void (APIENTRY PFNGLVERTEXATTRIBIPOINTER_PROC (GLuint index, GLint size, GLenum type, GLsizei stride, const void * pointer));
typedef void (APIENTRY PFNGLGETVERTEXATTRIBIIV_PROC (GLuint index, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETVERTEXATTRIBIUIV_PROC (GLuint index, GLenum pname, GLuint * params));
typedef void (APIENTRY PFNGLVERTEXATTRIBI1I_PROC (GLuint index, GLint x));
typedef void (APIENTRY PFNGLVERTEXATTRIBI2I_PROC (GLuint index, GLint x, GLint y));
typedef void (APIENTRY PFNGLVERTEXATTRIBI3I_PROC (GLuint index, GLint x, GLint y, GLint z));
typedef void (APIENTRY PFNGLVERTEXATTRIBI4I_PROC (GLuint index, GLint x, GLint y, GLint z, GLint w));
typedef void (APIENTRY PFNGLVERTEXATTRIBI1UI_PROC (GLuint index, GLuint x));
typedef void (APIENTRY PFNGLVERTEXATTRIBI2UI_PROC (GLuint index, GLuint x, GLuint y));
typedef void (APIENTRY PFNGLVERTEXATTRIBI3UI_PROC (GLuint index, GLuint x, GLuint y, GLuint z));
typedef void (APIENTRY PFNGLVERTEXATTRIBI4UI_PROC (GLuint index, GLuint x, GLuint y, GLuint z, GLuint w));
typedef void (APIENTRY PFNGLVERTEXATTRIBI1IV_PROC (GLuint index, const GLint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI2IV_PROC (GLuint index, const GLint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI3IV_PROC (GLuint index, const GLint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI4IV_PROC (GLuint index, const GLint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI1UIV_PROC (GLuint index, const GLuint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI2UIV_PROC (GLuint index, const GLuint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI3UIV_PROC (GLuint index, const GLuint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI4UIV_PROC (GLuint index, const GLuint * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI4BV_PROC (GLuint index, const GLbyte * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI4SV_PROC (GLuint index, const GLshort * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI4UBV_PROC (GLuint index, const GLubyte * v));
typedef void (APIENTRY PFNGLVERTEXATTRIBI4USV_PROC (GLuint index, const GLushort * v));
typedef void (APIENTRY PFNGLGETUNIFORMUIV_PROC (GLuint program, GLint location, GLuint * params));
typedef void (APIENTRY PFNGLBINDFRAGDATALOCATION_PROC (GLuint program, GLuint color, const GLchar * name));
typedef GLint (APIENTRY PFNGLGETFRAGDATALOCATION_PROC (GLuint program, const GLchar * name));
typedef void (APIENTRY PFNGLUNIFORM1UI_PROC (GLint location, GLuint v0));
typedef void (APIENTRY PFNGLUNIFORM2UI_PROC (GLint location, GLuint v0, GLuint v1));
typedef void (APIENTRY PFNGLUNIFORM3UI_PROC (GLint location, GLuint v0, GLuint v1, GLuint v2));
typedef void (APIENTRY PFNGLUNIFORM4UI_PROC (GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3));
typedef void (APIENTRY PFNGLUNIFORM1UIV_PROC (GLint location, GLsizei count, const GLuint * value));
typedef void (APIENTRY PFNGLUNIFORM2UIV_PROC (GLint location, GLsizei count, const GLuint * value));
typedef void (APIENTRY PFNGLUNIFORM3UIV_PROC (GLint location, GLsizei count, const GLuint * value));
typedef void (APIENTRY PFNGLUNIFORM4UIV_PROC (GLint location, GLsizei count, const GLuint * value));
typedef void (APIENTRY PFNGLTEXPARAMETERIIV_PROC (GLenum target, GLenum pname, const GLint * params));
typedef void (APIENTRY PFNGLTEXPARAMETERIUIV_PROC (GLenum target, GLenum pname, const GLuint * params));
typedef void (APIENTRY PFNGLGETTEXPARAMETERIIV_PROC (GLenum target, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETTEXPARAMETERIUIV_PROC (GLenum target, GLenum pname, GLuint * params));
typedef void (APIENTRY PFNGLCLEARBUFFERIV_PROC (GLenum buffer, GLint drawbuffer, const GLint * value));
typedef void (APIENTRY PFNGLCLEARBUFFERUIV_PROC (GLenum buffer, GLint drawbuffer, const GLuint * value));
typedef void (APIENTRY PFNGLCLEARBUFFERFV_PROC (GLenum buffer, GLint drawbuffer, const GLfloat * value));
typedef void (APIENTRY PFNGLCLEARBUFFERFI_PROC (GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil));
typedef const GLubyte * (APIENTRY PFNGLGETSTRINGI_PROC (GLenum name, GLuint index));
typedef GLboolean (APIENTRY PFNGLISRENDERBUFFER_PROC (GLuint renderbuffer));
typedef void (APIENTRY PFNGLBINDRENDERBUFFER_PROC (GLenum target, GLuint renderbuffer));
typedef void (APIENTRY PFNGLDELETERENDERBUFFERS_PROC (GLsizei n, const GLuint * renderbuffers));
typedef void (APIENTRY PFNGLGENRENDERBUFFERS_PROC (GLsizei n, GLuint * renderbuffers));
typedef void (APIENTRY PFNGLRENDERBUFFERSTORAGE_PROC (GLenum target, GLenum internalformat, GLsizei width, GLsizei height));
typedef void (APIENTRY PFNGLGETRENDERBUFFERPARAMETERIV_PROC (GLenum target, GLenum pname, GLint * params));
typedef GLboolean (APIENTRY PFNGLISFRAMEBUFFER_PROC (GLuint framebuffer));
typedef void (APIENTRY PFNGLBINDFRAMEBUFFER_PROC (GLenum target, GLuint framebuffer));
typedef void (APIENTRY PFNGLDELETEFRAMEBUFFERS_PROC (GLsizei n, const GLuint * framebuffers));
typedef void (APIENTRY PFNGLGENFRAMEBUFFERS_PROC (GLsizei n, GLuint * framebuffers));
typedef GLenum (APIENTRY PFNGLCHECKFRAMEBUFFERSTATUS_PROC (GLenum target));
typedef void (APIENTRY PFNGLFRAMEBUFFERTEXTURE1D_PROC (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level));
typedef void (APIENTRY PFNGLFRAMEBUFFERTEXTURE2D_PROC (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level));
typedef void (APIENTRY PFNGLFRAMEBUFFERTEXTURE3D_PROC (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset));
typedef void (APIENTRY PFNGLFRAMEBUFFERRENDERBUFFER_PROC (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer));
typedef void (APIENTRY PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIV_PROC (GLenum target, GLenum attachment, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGENERATEMIPMAP_PROC (GLenum target));
typedef void (APIENTRY PFNGLBLITFRAMEBUFFER_PROC (GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter));
typedef void (APIENTRY PFNGLRENDERBUFFERSTORAGEMULTISAMPLE_PROC (GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height));
typedef void (APIENTRY PFNGLFRAMEBUFFERTEXTURELAYER_PROC (GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer));
typedef void * (APIENTRY PFNGLMAPBUFFERRANGE_PROC (GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access));
typedef void (APIENTRY PFNGLFLUSHMAPPEDBUFFERRANGE_PROC (GLenum target, GLintptr offset, GLsizeiptr length));
typedef void (APIENTRY PFNGLBINDVERTEXARRAY_PROC (GLuint array));
typedef void (APIENTRY PFNGLDELETEVERTEXARRAYS_PROC (GLsizei n, const GLuint * arrays));
typedef void (APIENTRY PFNGLGENVERTEXARRAYS_PROC (GLsizei n, GLuint * arrays));
typedef GLboolean (APIENTRY PFNGLISVERTEXARRAY_PROC (GLuint array));
    
/* GL_VERSION_3_1 */
  
typedef void (APIENTRY PFNGLDRAWARRAYSINSTANCED_PROC (GLenum mode, GLint first, GLsizei count, GLsizei instancecount));
typedef void (APIENTRY PFNGLDRAWELEMENTSINSTANCED_PROC (GLenum mode, GLsizei count, GLenum type, const void * indices, GLsizei instancecount));
typedef void (APIENTRY PFNGLTEXBUFFER_PROC (GLenum target, GLenum internalformat, GLuint buffer));
typedef void (APIENTRY PFNGLPRIMITIVERESTARTINDEX_PROC (GLuint index));
typedef void (APIENTRY PFNGLCOPYBUFFERSUBDATA_PROC (GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size));
typedef void (APIENTRY PFNGLGETUNIFORMINDICES_PROC (GLuint program, GLsizei uniformCount, const GLchar *const* uniformNames, GLuint * uniformIndices));
typedef void (APIENTRY PFNGLGETACTIVEUNIFORMSIV_PROC (GLuint program, GLsizei uniformCount, const GLuint * uniformIndices, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETACTIVEUNIFORMNAME_PROC (GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei * length, GLchar * uniformName));
typedef GLuint (APIENTRY PFNGLGETUNIFORMBLOCKINDEX_PROC (GLuint program, const GLchar * uniformBlockName));
typedef void (APIENTRY PFNGLGETACTIVEUNIFORMBLOCKIV_PROC (GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint * params));
typedef void (APIENTRY PFNGLGETACTIVEUNIFORMBLOCKNAME_PROC (GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei * length, GLchar * uniformBlockName));
typedef void (APIENTRY PFNGLUNIFORMBLOCKBINDING_PROC (GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding));
    
struct OpenGLBindings
{
    
  /* GL_VERSION_1_2 */

  PFNGLDRAWRANGEELEMENTS_PROC* glDrawRangeElements;
  PFNGLTEXIMAGE3D_PROC* glTexImage3D;
  PFNGLTEXSUBIMAGE3D_PROC* glTexSubImage3D;
  PFNGLCOPYTEXSUBIMAGE3D_PROC* glCopyTexSubImage3D;
    
  /* GL_VERSION_1_3 */

  PFNGLACTIVETEXTURE_PROC* glActiveTexture;
  PFNGLSAMPLECOVERAGE_PROC* glSampleCoverage;
  PFNGLCOMPRESSEDTEXIMAGE3D_PROC* glCompressedTexImage3D;
  PFNGLCOMPRESSEDTEXIMAGE2D_PROC* glCompressedTexImage2D;
  PFNGLCOMPRESSEDTEXIMAGE1D_PROC* glCompressedTexImage1D;
  PFNGLCOMPRESSEDTEXSUBIMAGE3D_PROC* glCompressedTexSubImage3D;
  PFNGLCOMPRESSEDTEXSUBIMAGE2D_PROC* glCompressedTexSubImage2D;
  PFNGLCOMPRESSEDTEXSUBIMAGE1D_PROC* glCompressedTexSubImage1D;
  PFNGLGETCOMPRESSEDTEXIMAGE_PROC* glGetCompressedTexImage;
    
  /* GL_VERSION_1_4 */

  PFNGLBLENDFUNCSEPARATE_PROC* glBlendFuncSeparate;
  PFNGLMULTIDRAWARRAYS_PROC* glMultiDrawArrays;
  PFNGLMULTIDRAWELEMENTS_PROC* glMultiDrawElements;
  PFNGLPOINTPARAMETERF_PROC* glPointParameterf;
  PFNGLPOINTPARAMETERFV_PROC* glPointParameterfv;
  PFNGLPOINTPARAMETERI_PROC* glPointParameteri;
  PFNGLPOINTPARAMETERIV_PROC* glPointParameteriv;
  PFNGLBLENDCOLOR_PROC* glBlendColor;
  PFNGLBLENDEQUATION_PROC* glBlendEquation;
    
  /* GL_VERSION_1_5 */

  PFNGLGENQUERIES_PROC* glGenQueries;
  PFNGLDELETEQUERIES_PROC* glDeleteQueries;
  PFNGLISQUERY_PROC* glIsQuery;
  PFNGLBEGINQUERY_PROC* glBeginQuery;
  PFNGLENDQUERY_PROC* glEndQuery;
  PFNGLGETQUERYIV_PROC* glGetQueryiv;
  PFNGLGETQUERYOBJECTIV_PROC* glGetQueryObjectiv;
  PFNGLGETQUERYOBJECTUIV_PROC* glGetQueryObjectuiv;
  PFNGLBINDBUFFER_PROC* glBindBuffer;
  PFNGLDELETEBUFFERS_PROC* glDeleteBuffers;
  PFNGLGENBUFFERS_PROC* glGenBuffers;
  PFNGLISBUFFER_PROC* glIsBuffer;
  PFNGLBUFFERDATA_PROC* glBufferData;
  PFNGLBUFFERSUBDATA_PROC* glBufferSubData;
  PFNGLGETBUFFERSUBDATA_PROC* glGetBufferSubData;
  PFNGLMAPBUFFER_PROC* glMapBuffer;
  PFNGLUNMAPBUFFER_PROC* glUnmapBuffer;
  PFNGLGETBUFFERPARAMETERIV_PROC* glGetBufferParameteriv;
  PFNGLGETBUFFERPOINTERV_PROC* glGetBufferPointerv;
    
  /* GL_VERSION_2_0 */

  PFNGLBLENDEQUATIONSEPARATE_PROC* glBlendEquationSeparate;
  PFNGLDRAWBUFFERS_PROC* glDrawBuffers;
  PFNGLSTENCILOPSEPARATE_PROC* glStencilOpSeparate;
  PFNGLSTENCILFUNCSEPARATE_PROC* glStencilFuncSeparate;
  PFNGLSTENCILMASKSEPARATE_PROC* glStencilMaskSeparate;
  PFNGLATTACHSHADER_PROC* glAttachShader;
  PFNGLBINDATTRIBLOCATION_PROC* glBindAttribLocation;
  PFNGLCOMPILESHADER_PROC* glCompileShader;
  PFNGLCREATEPROGRAM_PROC* glCreateProgram;
  PFNGLCREATESHADER_PROC* glCreateShader;
  PFNGLDELETEPROGRAM_PROC* glDeleteProgram;
  PFNGLDELETESHADER_PROC* glDeleteShader;
  PFNGLDETACHSHADER_PROC* glDetachShader;
  PFNGLDISABLEVERTEXATTRIBARRAY_PROC* glDisableVertexAttribArray;
  PFNGLENABLEVERTEXATTRIBARRAY_PROC* glEnableVertexAttribArray;
  PFNGLGETACTIVEATTRIB_PROC* glGetActiveAttrib;
  PFNGLGETACTIVEUNIFORM_PROC* glGetActiveUniform;
  PFNGLGETATTACHEDSHADERS_PROC* glGetAttachedShaders;
  PFNGLGETATTRIBLOCATION_PROC* glGetAttribLocation;
  PFNGLGETPROGRAMIV_PROC* glGetProgramiv;
  PFNGLGETPROGRAMINFOLOG_PROC* glGetProgramInfoLog;
  PFNGLGETSHADERIV_PROC* glGetShaderiv;
  PFNGLGETSHADERINFOLOG_PROC* glGetShaderInfoLog;
  PFNGLGETSHADERSOURCE_PROC* glGetShaderSource;
  PFNGLGETUNIFORMLOCATION_PROC* glGetUniformLocation;
  PFNGLGETUNIFORMFV_PROC* glGetUniformfv;
  PFNGLGETUNIFORMIV_PROC* glGetUniformiv;
  PFNGLGETVERTEXATTRIBDV_PROC* glGetVertexAttribdv;
  PFNGLGETVERTEXATTRIBFV_PROC* glGetVertexAttribfv;
  PFNGLGETVERTEXATTRIBIV_PROC* glGetVertexAttribiv;
  PFNGLGETVERTEXATTRIBPOINTERV_PROC* glGetVertexAttribPointerv;
  PFNGLISPROGRAM_PROC* glIsProgram;
  PFNGLISSHADER_PROC* glIsShader;
  PFNGLLINKPROGRAM_PROC* glLinkProgram;
  PFNGLSHADERSOURCE_PROC* glShaderSource;
  PFNGLUSEPROGRAM_PROC* glUseProgram;
  PFNGLUNIFORM1F_PROC* glUniform1f;
  PFNGLUNIFORM2F_PROC* glUniform2f;
  PFNGLUNIFORM3F_PROC* glUniform3f;
  PFNGLUNIFORM4F_PROC* glUniform4f;
  PFNGLUNIFORM1I_PROC* glUniform1i;
  PFNGLUNIFORM2I_PROC* glUniform2i;
  PFNGLUNIFORM3I_PROC* glUniform3i;
  PFNGLUNIFORM4I_PROC* glUniform4i;
  PFNGLUNIFORM1FV_PROC* glUniform1fv;
  PFNGLUNIFORM2FV_PROC* glUniform2fv;
  PFNGLUNIFORM3FV_PROC* glUniform3fv;
  PFNGLUNIFORM4FV_PROC* glUniform4fv;
  PFNGLUNIFORM1IV_PROC* glUniform1iv;
  PFNGLUNIFORM2IV_PROC* glUniform2iv;
  PFNGLUNIFORM3IV_PROC* glUniform3iv;
  PFNGLUNIFORM4IV_PROC* glUniform4iv;
  PFNGLUNIFORMMATRIX2FV_PROC* glUniformMatrix2fv;
  PFNGLUNIFORMMATRIX3FV_PROC* glUniformMatrix3fv;
  PFNGLUNIFORMMATRIX4FV_PROC* glUniformMatrix4fv;
  PFNGLVALIDATEPROGRAM_PROC* glValidateProgram;
  PFNGLVERTEXATTRIB1D_PROC* glVertexAttrib1d;
  PFNGLVERTEXATTRIB1DV_PROC* glVertexAttrib1dv;
  PFNGLVERTEXATTRIB1F_PROC* glVertexAttrib1f;
  PFNGLVERTEXATTRIB1FV_PROC* glVertexAttrib1fv;
  PFNGLVERTEXATTRIB1S_PROC* glVertexAttrib1s;
  PFNGLVERTEXATTRIB1SV_PROC* glVertexAttrib1sv;
  PFNGLVERTEXATTRIB2D_PROC* glVertexAttrib2d;
  PFNGLVERTEXATTRIB2DV_PROC* glVertexAttrib2dv;
  PFNGLVERTEXATTRIB2F_PROC* glVertexAttrib2f;
  PFNGLVERTEXATTRIB2FV_PROC* glVertexAttrib2fv;
  PFNGLVERTEXATTRIB2S_PROC* glVertexAttrib2s;
  PFNGLVERTEXATTRIB2SV_PROC* glVertexAttrib2sv;
  PFNGLVERTEXATTRIB3D_PROC* glVertexAttrib3d;
  PFNGLVERTEXATTRIB3DV_PROC* glVertexAttrib3dv;
  PFNGLVERTEXATTRIB3F_PROC* glVertexAttrib3f;
  PFNGLVERTEXATTRIB3FV_PROC* glVertexAttrib3fv;
  PFNGLVERTEXATTRIB3S_PROC* glVertexAttrib3s;
  PFNGLVERTEXATTRIB3SV_PROC* glVertexAttrib3sv;
  PFNGLVERTEXATTRIB4NBV_PROC* glVertexAttrib4Nbv;
  PFNGLVERTEXATTRIB4NIV_PROC* glVertexAttrib4Niv;
  PFNGLVERTEXATTRIB4NSV_PROC* glVertexAttrib4Nsv;
  PFNGLVERTEXATTRIB4NUB_PROC* glVertexAttrib4Nub;
  PFNGLVERTEXATTRIB4NUBV_PROC* glVertexAttrib4Nubv;
  PFNGLVERTEXATTRIB4NUIV_PROC* glVertexAttrib4Nuiv;
  PFNGLVERTEXATTRIB4NUSV_PROC* glVertexAttrib4Nusv;
  PFNGLVERTEXATTRIB4BV_PROC* glVertexAttrib4bv;
  PFNGLVERTEXATTRIB4D_PROC* glVertexAttrib4d;
  PFNGLVERTEXATTRIB4DV_PROC* glVertexAttrib4dv;
  PFNGLVERTEXATTRIB4F_PROC* glVertexAttrib4f;
  PFNGLVERTEXATTRIB4FV_PROC* glVertexAttrib4fv;
  PFNGLVERTEXATTRIB4IV_PROC* glVertexAttrib4iv;
  PFNGLVERTEXATTRIB4S_PROC* glVertexAttrib4s;
  PFNGLVERTEXATTRIB4SV_PROC* glVertexAttrib4sv;
  PFNGLVERTEXATTRIB4UBV_PROC* glVertexAttrib4ubv;
  PFNGLVERTEXATTRIB4UIV_PROC* glVertexAttrib4uiv;
  PFNGLVERTEXATTRIB4USV_PROC* glVertexAttrib4usv;
  PFNGLVERTEXATTRIBPOINTER_PROC* glVertexAttribPointer;
    
  /* GL_VERSION_2_1 */

  PFNGLUNIFORMMATRIX2X3FV_PROC* glUniformMatrix2x3fv;
  PFNGLUNIFORMMATRIX3X2FV_PROC* glUniformMatrix3x2fv;
  PFNGLUNIFORMMATRIX2X4FV_PROC* glUniformMatrix2x4fv;
  PFNGLUNIFORMMATRIX4X2FV_PROC* glUniformMatrix4x2fv;
  PFNGLUNIFORMMATRIX3X4FV_PROC* glUniformMatrix3x4fv;
  PFNGLUNIFORMMATRIX4X3FV_PROC* glUniformMatrix4x3fv;
    
  /* GL_VERSION_3_0 */

  PFNGLCOLORMASKI_PROC* glColorMaski;
  PFNGLGETBOOLEANI_V_PROC* glGetBooleani_v;
  PFNGLGETINTEGERI_V_PROC* glGetIntegeri_v;
  PFNGLENABLEI_PROC* glEnablei;
  PFNGLDISABLEI_PROC* glDisablei;
  PFNGLISENABLEDI_PROC* glIsEnabledi;
  PFNGLBEGINTRANSFORMFEEDBACK_PROC* glBeginTransformFeedback;
  PFNGLENDTRANSFORMFEEDBACK_PROC* glEndTransformFeedback;
  PFNGLBINDBUFFERRANGE_PROC* glBindBufferRange;
  PFNGLBINDBUFFERBASE_PROC* glBindBufferBase;
  PFNGLTRANSFORMFEEDBACKVARYINGS_PROC* glTransformFeedbackVaryings;
  PFNGLGETTRANSFORMFEEDBACKVARYING_PROC* glGetTransformFeedbackVarying;
  PFNGLCLAMPCOLOR_PROC* glClampColor;
  PFNGLBEGINCONDITIONALRENDER_PROC* glBeginConditionalRender;
  PFNGLENDCONDITIONALRENDER_PROC* glEndConditionalRender;
  PFNGLVERTEXATTRIBIPOINTER_PROC* glVertexAttribIPointer;
  PFNGLGETVERTEXATTRIBIIV_PROC* glGetVertexAttribIiv;
  PFNGLGETVERTEXATTRIBIUIV_PROC* glGetVertexAttribIuiv;
  PFNGLVERTEXATTRIBI1I_PROC* glVertexAttribI1i;
  PFNGLVERTEXATTRIBI2I_PROC* glVertexAttribI2i;
  PFNGLVERTEXATTRIBI3I_PROC* glVertexAttribI3i;
  PFNGLVERTEXATTRIBI4I_PROC* glVertexAttribI4i;
  PFNGLVERTEXATTRIBI1UI_PROC* glVertexAttribI1ui;
  PFNGLVERTEXATTRIBI2UI_PROC* glVertexAttribI2ui;
  PFNGLVERTEXATTRIBI3UI_PROC* glVertexAttribI3ui;
  PFNGLVERTEXATTRIBI4UI_PROC* glVertexAttribI4ui;
  PFNGLVERTEXATTRIBI1IV_PROC* glVertexAttribI1iv;
  PFNGLVERTEXATTRIBI2IV_PROC* glVertexAttribI2iv;
  PFNGLVERTEXATTRIBI3IV_PROC* glVertexAttribI3iv;
  PFNGLVERTEXATTRIBI4IV_PROC* glVertexAttribI4iv;
  PFNGLVERTEXATTRIBI1UIV_PROC* glVertexAttribI1uiv;
  PFNGLVERTEXATTRIBI2UIV_PROC* glVertexAttribI2uiv;
  PFNGLVERTEXATTRIBI3UIV_PROC* glVertexAttribI3uiv;
  PFNGLVERTEXATTRIBI4UIV_PROC* glVertexAttribI4uiv;
  PFNGLVERTEXATTRIBI4BV_PROC* glVertexAttribI4bv;
  PFNGLVERTEXATTRIBI4SV_PROC* glVertexAttribI4sv;
  PFNGLVERTEXATTRIBI4UBV_PROC* glVertexAttribI4ubv;
  PFNGLVERTEXATTRIBI4USV_PROC* glVertexAttribI4usv;
  PFNGLGETUNIFORMUIV_PROC* glGetUniformuiv;
  PFNGLBINDFRAGDATALOCATION_PROC* glBindFragDataLocation;
  PFNGLGETFRAGDATALOCATION_PROC* glGetFragDataLocation;
  PFNGLUNIFORM1UI_PROC* glUniform1ui;
  PFNGLUNIFORM2UI_PROC* glUniform2ui;
  PFNGLUNIFORM3UI_PROC* glUniform3ui;
  PFNGLUNIFORM4UI_PROC* glUniform4ui;
  PFNGLUNIFORM1UIV_PROC* glUniform1uiv;
  PFNGLUNIFORM2UIV_PROC* glUniform2uiv;
  PFNGLUNIFORM3UIV_PROC* glUniform3uiv;
  PFNGLUNIFORM4UIV_PROC* glUniform4uiv;
  PFNGLTEXPARAMETERIIV_PROC* glTexParameterIiv;
  PFNGLTEXPARAMETERIUIV_PROC* glTexParameterIuiv;
  PFNGLGETTEXPARAMETERIIV_PROC* glGetTexParameterIiv;
  PFNGLGETTEXPARAMETERIUIV_PROC* glGetTexParameterIuiv;
  PFNGLCLEARBUFFERIV_PROC* glClearBufferiv;
  PFNGLCLEARBUFFERUIV_PROC* glClearBufferuiv;
  PFNGLCLEARBUFFERFV_PROC* glClearBufferfv;
  PFNGLCLEARBUFFERFI_PROC* glClearBufferfi;
  PFNGLGETSTRINGI_PROC* glGetStringi;
  PFNGLISRENDERBUFFER_PROC* glIsRenderbuffer;
  PFNGLBINDRENDERBUFFER_PROC* glBindRenderbuffer;
  PFNGLDELETERENDERBUFFERS_PROC* glDeleteRenderbuffers;
  PFNGLGENRENDERBUFFERS_PROC* glGenRenderbuffers;
  PFNGLRENDERBUFFERSTORAGE_PROC* glRenderbufferStorage;
  PFNGLGETRENDERBUFFERPARAMETERIV_PROC* glGetRenderbufferParameteriv;
  PFNGLISFRAMEBUFFER_PROC* glIsFramebuffer;
  PFNGLBINDFRAMEBUFFER_PROC* glBindFramebuffer;
  PFNGLDELETEFRAMEBUFFERS_PROC* glDeleteFramebuffers;
  PFNGLGENFRAMEBUFFERS_PROC* glGenFramebuffers;
  PFNGLCHECKFRAMEBUFFERSTATUS_PROC* glCheckFramebufferStatus;
  PFNGLFRAMEBUFFERTEXTURE1D_PROC* glFramebufferTexture1D;
  PFNGLFRAMEBUFFERTEXTURE2D_PROC* glFramebufferTexture2D;
  PFNGLFRAMEBUFFERTEXTURE3D_PROC* glFramebufferTexture3D;
  PFNGLFRAMEBUFFERRENDERBUFFER_PROC* glFramebufferRenderbuffer;
  PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIV_PROC* glGetFramebufferAttachmentParameteriv;
  PFNGLGENERATEMIPMAP_PROC* glGenerateMipmap;
  PFNGLBLITFRAMEBUFFER_PROC* glBlitFramebuffer;
  PFNGLRENDERBUFFERSTORAGEMULTISAMPLE_PROC* glRenderbufferStorageMultisample;
  PFNGLFRAMEBUFFERTEXTURELAYER_PROC* glFramebufferTextureLayer;
  PFNGLMAPBUFFERRANGE_PROC* glMapBufferRange;
  PFNGLFLUSHMAPPEDBUFFERRANGE_PROC* glFlushMappedBufferRange;
  PFNGLBINDVERTEXARRAY_PROC* glBindVertexArray;
  PFNGLDELETEVERTEXARRAYS_PROC* glDeleteVertexArrays;
  PFNGLGENVERTEXARRAYS_PROC* glGenVertexArrays;
  PFNGLISVERTEXARRAY_PROC* glIsVertexArray;
    
  /* GL_VERSION_3_1 */

  PFNGLDRAWARRAYSINSTANCED_PROC* glDrawArraysInstanced;
  PFNGLDRAWELEMENTSINSTANCED_PROC* glDrawElementsInstanced;
  PFNGLTEXBUFFER_PROC* glTexBuffer;
  PFNGLPRIMITIVERESTARTINDEX_PROC* glPrimitiveRestartIndex;
  PFNGLCOPYBUFFERSUBDATA_PROC* glCopyBufferSubData;
  PFNGLGETUNIFORMINDICES_PROC* glGetUniformIndices;
  PFNGLGETACTIVEUNIFORMSIV_PROC* glGetActiveUniformsiv;
  PFNGLGETACTIVEUNIFORMNAME_PROC* glGetActiveUniformName;
  PFNGLGETUNIFORMBLOCKINDEX_PROC* glGetUniformBlockIndex;
  PFNGLGETACTIVEUNIFORMBLOCKIV_PROC* glGetActiveUniformBlockiv;
  PFNGLGETACTIVEUNIFORMBLOCKNAME_PROC* glGetActiveUniformBlockName;
  PFNGLUNIFORMBLOCKBINDING_PROC* glUniformBlockBinding;
    
};

typedef struct OpenGLBindings OpenGLBindings;

/* --------------------------- CATEGORY DEFINES ------------------------------ */

#define GL_VERSION_1_0
#define GL_VERSION_1_1
#define GL_VERSION_1_2
#define GL_VERSION_1_3
#define GL_VERSION_1_4
#define GL_VERSION_1_5
#define GL_VERSION_2_0
#define GL_VERSION_2_1
#define GL_VERSION_3_0
#define GL_VERSION_3_1

/* ---------------------- Flags for optional extensions ---------------------- */

void flextInit(OpenGLBindings *bindings);

#define FLEXT_MAJOR_VERSION 3
#define FLEXT_MINOR_VERSION 1
#define FLEXT_CORE_PROFILE 1

#ifdef __cplusplus
}
#endif

#endif /* _gl_h_ */
