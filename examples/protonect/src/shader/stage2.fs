struct Parameters
{
  float ab_multiplier;
  vec3 ab_multiplier_per_frq;
  float ab_output_multiplier;
  
  vec3 phase_in_rad;
  
  float joint_bilateral_ab_threshold;
  float joint_bilateral_max_edge;
  float joint_bilateral_exp;
  mat3 gaussian_kernel;
  
  float phase_offset;
  float unambigious_dist;
  float individual_ab_threshold;
  float ab_threshold;
  float ab_confidence_slope;
  float ab_confidence_offset;
  float min_dealias_confidence;
  float max_dealias_confidence;
  
  float edge_ab_avg_min_value;
  float edge_ab_std_dev_threshold;
  float edge_close_delta_threshold;
  float edge_far_delta_threshold;
  float edge_max_delta_threshold;
  float edge_avg_delta_threshold;
  float max_edge_count;
  
  float min_depth;
  float max_depth;
};

uniform sampler2DRect A;
uniform sampler2DRect B;
uniform sampler2DRect XTable;
uniform sampler2DRect ZTable;

uniform Parameters Params;

in vec2 TexCoord;

/*layout(location = 0)*/ out vec4 Debug;
/*layout(location = 1)*/ out float Depth;
/*layout(location = 2)*/ out vec2 DepthAndIrSum;

#define M_PI 3.1415926535897932384626433832795

void main(void)
{
  ivec2 uv = ivec2(TexCoord.x, TexCoord.y);
      
  vec3 a = texelFetch(A, uv).xyz;
  vec3 b = texelFetch(B, uv).xyz;
  
  vec3 phase = atan(b, a);
  phase = mix(phase, phase + 2.0 * M_PI, lessThan(phase, vec3(0.0)));
  phase = mix(phase, vec3(0.0), isnan(phase));
  vec3 ir = sqrt(a * a + b * b) * Params.ab_multiplier;
  
  float ir_sum = ir.x + ir.y + ir.z;
  float ir_min = min(ir.x, min(ir.y, ir.z));
  float ir_max = max(ir.x, max(ir.y, ir.z));
  
  float phase_final = 0;
  
  if(ir_min >= Params.individual_ab_threshold && ir_sum >= Params.ab_threshold)
  {
    vec3 t = phase / (2.0 * M_PI) * vec3(3.0, 15.0, 2.0);
  
    float t0 = t.x;
    float t1 = t.y;
    float t2 = t.z;

    float t5 = (floor((t1 - t0) * 0.333333f + 0.5f) * 3.0f + t0);
    float t3 = (-t2 + t5);
    float t4 = t3 * 2.0f;

    bool c1 = t4 >= -t4; // true if t4 positive

    float f1 = c1 ? 2.0f : -2.0f;
    float f2 = c1 ? 0.5f : -0.5f;
    t3 *= f2;
    t3 = (t3 - floor(t3)) * f1;

    bool c2 = 0.5f < abs(t3) && abs(t3) < 1.5f;

    float t6 = c2 ? t5 + 15.0f : t5;
    float t7 = c2 ? t1 + 15.0f : t1;

    float t8 = (floor((-t2 + t6) * 0.5f + 0.5f) * 2.0f + t2) * 0.5f;

    t6 *= 0.333333f; // = / 3
    t7 *= 0.066667f; // = / 15

    float t9 = (t8 + t6 + t7); // transformed phase measurements (they are transformed and divided by the values the original values were multiplied with)
    float t10 = t9 * 0.333333f; // some avg

    t6 *= 2.0f * M_PI;
    t7 *= 2.0f * M_PI;
    t8 *= 2.0f * M_PI;

    // some cross product
    float t8_new = t7 * 0.826977f - t8 * 0.110264f;
    float t6_new = t8 * 0.551318f - t6 * 0.826977f;
    float t7_new = t6 * 0.110264f - t7 * 0.551318f;

    t8 = t8_new;
    t6 = t6_new;
    t7 = t7_new;

    float norm = t8 * t8 + t6 * t6 + t7 * t7;
    float mask = t9 >= 0.0f ? 1.0f : 0.0f;
    t10 *= mask;

    bool slope_positive = 0 < Params.ab_confidence_slope;

    float ir_x = slope_positive ? ir_min : ir_max;

    ir_x = log(ir_x);
    ir_x = (ir_x * Params.ab_confidence_slope * 0.301030f + Params.ab_confidence_offset) * 3.321928f;
    ir_x = exp(ir_x);
    ir_x = min(Params.max_dealias_confidence, max(Params.min_dealias_confidence, ir_x));
    ir_x *= ir_x;

    float mask2 = ir_x >= norm ? 1.0f : 0.0f;

    float t11 = t10 * mask2;

    float mask3 = Params.max_dealias_confidence * Params.max_dealias_confidence >= norm ? 1.0f : 0.0f;
    t10 *= mask3;
    phase_final = true/*(modeMask & 2) != 0*/ ? t11 : t10;
  }
  
  float zmultiplier = texelFetch(ZTable, uv).x;
  float xmultiplier = texelFetch(XTable, uv).x;

  phase_final = 0.0 < phase_final ? phase_final + Params.phase_offset : phase_final;

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * Params.unambigious_dist * 2.0;

  bool cond1 = /*(modeMask & 32) != 0*/ true && 0.0 < depth_linear && 0.0 < max_depth;

  xmultiplier = (xmultiplier * 90.0) / (max_depth * max_depth * 8192.0);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);
  depth_fit = depth_fit < 0.0 ? 0.0 : depth_fit;
  
  Depth = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z
  DepthAndIrSum = vec2(Depth, ir_sum);
  
  Debug = vec4(vec3(Depth / Params.max_depth), 1.0);
}
