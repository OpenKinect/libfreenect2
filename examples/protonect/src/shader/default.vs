in vec2 InputPosition;
in vec2 InputTexCoord;

out vec2 TexCoord;

void main(void)
{
  gl_Position = vec4(InputPosition, 0.0, 1.0);
  TexCoord = InputTexCoord;
}
