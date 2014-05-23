#version 330

uniform sampler2DRect Data;

in VertexData {
    vec2 TexCoord;
} FragmentIn;

out layout(location = 0) vec4 Color;

void main(void)
{
  ivec2 uv = ivec2(FragmentIn.TexCoord.x, FragmentIn.TexCoord.y);
  
  Color = texelFetch(Data, uv);
}