#version 330

uniform sampler2DRect DepthAndIrSum;
uniform usampler2DRect MaxEdgeTest;

in VertexData {
    vec2 TexCoord;
} FragmentIn;

layout(location = 0) out vec4 Debug;
layout(location = 1) out float FilterDepth;

void filter(ivec2 uv)
{
  const float min_depth = 100.0f;
  const float max_depth = 10000.0f;
  const float edge_ab_avg_min_value = 50;
  const float edge_ab_std_dev_threshold = 0.05;
  const float edge_close_delta_threshold = 50;
  const float edge_far_delta_threshold = 30;
  const float edge_max_delta_threshold = 100;
  const float edge_avg_delta_threshold = 0;
  const float max_edge_count = 5.0f;

  vec2 v = texelFetch(DepthAndIrSum, uv).xy;
  
  if(v.x >= min_depth && v.x <= max_depth)
  {
    if(uv.x < 1 || uv.y < 1 || uv.x > 510 || uv.y > 422)
    {
      FilterDepth = v.x;
    }
    else
    {
      bool max_edge_test_ok = texelFetch(MaxEdgeTest, uv).x > 0u;
      
      float ir_sum_acc = v.y, squared_ir_sum_acc = v.y * v.y, min_depth = v.x, max_depth = v.x;

      for(int yi = -1; yi < 2; ++yi)
      {
        for(int xi = -1; xi < 2; ++xi)
        {
          if(yi == 0 && xi == 0) continue;

          vec2 other = texelFetch(DepthAndIrSum, uv + ivec2(xi, yi)).xy;

          ir_sum_acc += other.y;
          squared_ir_sum_acc += other.y * other.y;

          if(0.0f < other.x)
          {
            min_depth = min(min_depth, other.x);
            max_depth = max(max_depth, other.x);
          }
        }
      }

      float tmp0 = sqrt(squared_ir_sum_acc * 9.0f - ir_sum_acc * ir_sum_acc) / 9.0f;
      float edge_avg = max(ir_sum_acc / 9.0f, edge_ab_avg_min_value);
      tmp0 /= edge_avg;

      float abs_min_diff = abs(v.x - min_depth);
      float abs_max_diff = abs(v.x - max_depth);

      float avg_diff = (abs_min_diff + abs_max_diff) * 0.5f;
      float max_abs_diff = max(abs_min_diff, abs_max_diff);

      bool cond0 =
          0.0f < v.x &&
          tmp0 >= edge_ab_std_dev_threshold &&
          edge_close_delta_threshold < abs_min_diff &&
          edge_far_delta_threshold < abs_max_diff &&
          edge_max_delta_threshold < max_abs_diff &&
          edge_avg_delta_threshold < avg_diff;

      FilterDepth = cond0 ? 0.0f : v.x;

      if(!cond0)
      {
        if(max_edge_test_ok)
        {
          float tmp1 = 1500.0f > v.x ? 30.0f : 0.02f * v.x;
          float edge_count = 0.0f;

          FilterDepth = edge_count > max_edge_count ? 0.0f : v.x;
        }
        else
        {
          FilterDepth = !max_edge_test_ok ? 0.0f : v.x;
          //FilterDepth = true ? FilterDepth : v.x;
        }
      }
    }
  }
  else
  {
    FilterDepth = 0.0f;
  }
}

void main(void)
{
  ivec2 uv = ivec2(FragmentIn.TexCoord.x, FragmentIn.TexCoord.y);
  
  filter(uv);
  
  Debug = vec4(vec3(FilterDepth / 4500.0), 1);
}
