#version 330

in vec2 Position;
in vec2 TexCoord;

out VertexData {
  vec2 TexCoord;
} VertexOut;

void main(void)
{
  gl_Position = vec4(Position, 0.0, 1.0);
  VertexOut.TexCoord = TexCoord;
}