uniform sampler2DRect Data;

in vec2 TexCoord;

out vec4 Color;

void main(void)
{
  ivec2 uv = ivec2(TexCoord.x, TexCoord.y);
  
  Color = texelFetch(Data, uv);
}
