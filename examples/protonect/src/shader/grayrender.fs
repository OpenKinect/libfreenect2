#version 330

uniform sampler2DRect Data;

vec4 tempColor;

in VertexData {
    vec2 TexCoord;
} FragmentIn;

layout(location = 0) out vec4 Color;

void main(void)
{
  ivec2 uv = ivec2(FragmentIn.TexCoord.x, FragmentIn.TexCoord.y);
  
  tempColor = texelFetch(Data, uv);

  Color = vec4(tempColor.x,tempColor.x,tempColor.x,1);
}