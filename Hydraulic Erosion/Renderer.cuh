#ifndef RENDERER_CUH
#define RENDERER_CUH

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Shader.h"

using namespace std;

class Renderer {

private:
	unsigned int VBO, VAO, EBO;
	unsigned int tex3d;
	unsigned int SCR_WIDTH = 800;
	unsigned int SCR_HEIGHT = 600;

public:
	GLFWwindow* window;

	int Init(unsigned char* data, int x, int y, int z);
	void Render(Shader shader);
	void End();
};

#endif // !RENDERER_CUH
