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
uniform sampler2DRect Norm;

uniform Parameters Params;

in vec2 TexCoord;

/*layout(location = 0)*/ out vec4 Debug;
/*layout(location = 1)*/ out vec3 FilterA;
/*layout(location = 2)*/ out vec3 FilterB;
/*layout(location = 3)*/ out uint MaxEdgeTest;

void applyBilateralFilter(ivec2 uv)
{
  vec3 threshold = vec3((Params.joint_bilateral_ab_threshold * Params.joint_bilateral_ab_threshold) / (Params.ab_multiplier * Params.ab_multiplier));
  vec3 joint_bilateral_exp = vec3(Params.joint_bilateral_exp);
  
  vec3 self_a = texelFetch(A, uv).xyz;
  vec3 self_b = texelFetch(B, uv).xyz;
  vec3 self_norm = texelFetch(Norm, uv).xyz;
  vec3 self_normalized_a = self_a / self_norm;
  vec3 self_normalized_b = self_b / self_norm;
  
  vec4 weight_acc = vec4(0.0);
  vec4 weighted_a_acc = vec4(0.0);
  vec4 weighted_b_acc = vec4(0.0);
  
  bvec3 c0 = lessThan(self_norm * self_norm, threshold);
  
  threshold = mix(threshold, vec3(0.0), c0);
  joint_bilateral_exp = mix(joint_bilateral_exp, vec3(0.0), c0);
  
  for(int y = 0; y < 3; ++y)
  {
    for(int x = 0; x < 3; ++x)
    {
      ivec2 ouv = uv + ivec2(x - 1, y - 1);
    
      vec3 other_a = texelFetch(A, ouv).xyz;
      vec3 other_b = texelFetch(B, ouv).xyz;
      vec3 other_norm = texelFetch(Norm, ouv).xyz;
      
      vec3 other_normalized_a = other_a / other_norm;
      vec3 other_normalized_b = other_b / other_norm;
            
      bvec3 c1 = lessThan(other_norm * other_norm, threshold);
      
      vec3 dist = 0.5f * (1.0f - (self_normalized_a * other_normalized_a + self_normalized_b * other_normalized_b));
      vec3 weight = mix(Params.gaussian_kernel[x][y] * exp(-1.442695 * joint_bilateral_exp * dist), vec3(0.0), c1);
      
      weighted_a_acc.xyz += weight * other_a;
      weighted_b_acc.xyz += weight * other_b;
      weight_acc.xyz += weight;
      
      // TODO: this sucks, but otherwise opengl reports error: temporary registers exceeded :(
      weighted_a_acc.w += mix(dist.x, 0, c1.x);
      weighted_b_acc.w += mix(dist.y, 0, c1.y);
      weight_acc.w += mix(dist.z, 0, c1.z);
    }
  }
  
  bvec3 c2 = lessThan(vec3(0.0), weight_acc.xyz);
  FilterA = mix(vec3(0.0), weighted_a_acc.xyz / weight_acc.xyz, c2);
  FilterB = mix(vec3(0.0), weighted_b_acc.xyz / weight_acc.xyz, c2);
  
  if(uv.x < 1 || uv.y < 1 || uv.x > 510 || uv.y > 422)
  {
    FilterA = self_a;
    FilterB = self_b;
  }
  
  vec3 dist_acc = vec3(weighted_a_acc.w, weighted_b_acc.w, weight_acc.w);
  MaxEdgeTest = uint(all(lessThan(dist_acc, vec3(Params.joint_bilateral_max_edge))));
  //Debug = vec4(vec3(MaxEdgeTest), 1);
}

void main(void)
{
  ivec2 uv = ivec2(TexCoord.x, TexCoord.y);
    
  applyBilateralFilter(uv);
  
  vec3 norm = sqrt(FilterA * FilterA + FilterB * FilterB);
  float i = min(dot(norm, vec3(0.333333333  * Params.ab_multiplier * Params.ab_output_multiplier)), 65535.0);
  
  Debug = vec4(vec3(i, i, i) / 65535.0, 1);
}
