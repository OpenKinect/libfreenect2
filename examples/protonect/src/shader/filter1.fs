#version 330

uniform sampler2DRect A;
uniform sampler2DRect B;
uniform sampler2DRect Norm;

in VertexData {
    vec2 TexCoord;
} FragmentIn;

out layout(location = 0) vec4 Debug;
out layout(location = 1) vec3 FilterA;
out layout(location = 2) vec3 FilterB;
out layout(location = 3) int MaxEdgeTest;

void filter(ivec2 uv)
{
  const float joint_bilateral_ab_threshold = 3.0;
  const float joint_bilateral_max_edge = 2.5;
  const float ab_multiplier = 0.6666667;

  vec3 threshold = vec3((joint_bilateral_ab_threshold * joint_bilateral_ab_threshold) / (ab_multiplier * ab_multiplier));
  vec3 joint_bilateral_exp = vec3(5.0);
  
  vec3 self_a = texelFetch(A, uv).xyz;
  vec3 self_b = texelFetch(B, uv).xyz;
  vec3 self_norm = texelFetch(Norm, uv).xyz;
  vec3 self_normalized_a = self_a / self_norm;
  vec3 self_normalized_b = self_b / self_norm;
  
  vec4 weight_acc = vec4(0.0);
  vec4 weighted_a_acc = vec4(0.0);
  vec4 weighted_b_acc = vec4(0.0);
  
  const mat3 kernel = mat3(
    0.1069973, 0.1131098, 0.1069973,
    0.1131098, 0.1195715, 0.1131098,
    0.1069973, 0.1131098, 0.1069973
  );
  
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
      vec3 weight = mix(kernel[x][y] * exp(-1.442695 * joint_bilateral_exp * dist), vec3(0.0), c1);
      
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
  
  if(uv.x < 1 || uv.y < 1 || uv.x > 510 || uv.y > 510)
  {
    FilterA = self_a;
    FilterB = self_b;
  }
  
  vec3 dist_acc = vec3(weighted_a_acc.w, weighted_b_acc.w, weight_acc.w);
  MaxEdgeTest = int(all(lessThan(dist_acc, vec3(joint_bilateral_max_edge))));
  //Debug = vec4(vec3(MaxEdgeTest), 1);
}

void main(void)
{
  ivec2 uv = ivec2(FragmentIn.TexCoord.x, FragmentIn.TexCoord.y);
  
  float ab_multiplier = 0.6666667;
  float ab_output_multiplier = 16.0;
  
  filter(uv);
  
  vec3 norm = sqrt(FilterA * FilterA + FilterB * FilterB);
  float i = min(dot(norm, vec3(0.333333333  * ab_multiplier * ab_output_multiplier)), 65535.0);
  
  Debug = vec4(vec3(i, i, i) / 65535.0, 1);
}