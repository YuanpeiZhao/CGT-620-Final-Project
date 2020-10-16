#version 330 core
out vec4 FragColor;

in vec3 texCoord;

uniform sampler3D tex;

void main()
{
    float t = texture(tex, texCoord).x * 100;
    FragColor = vec4(vec3(t) ,1.0f);
}