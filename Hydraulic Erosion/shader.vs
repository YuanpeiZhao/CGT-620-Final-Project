#version 460 core
layout (location = 0) in vec3 aPos;

layout(location = 0) uniform mat4 M;
layout(location = 1) uniform mat4 V;
layout(location = 2) uniform mat4 P;

out vec3 vPos;

void main()
{
    vPos = (M * vec4(aPos, 1.0f)).zxy;
    gl_Position = P * V * M * vec4(aPos, 1.0);
}