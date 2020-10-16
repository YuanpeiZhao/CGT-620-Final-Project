#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

out vec3 texCoord;

void main()
{
    texCoord = (aPos+vec3(1.0f)) / 2.0f;
    gl_Position = P * V * M * vec4(aPos, 1.0);
}