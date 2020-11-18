#ifndef RENDERER_CUH
#define RENDERER_CUH

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glext.h>

#include <windows.h>
#include <string>
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cuda_gl_interop.h>

#include "InitShader.h"
#include "ShaderLocs.h"

using namespace std;

class Renderer {

private:
	
	unsigned int SCR_WIDTH;
	unsigned int SCR_HEIGHT;

	GLuint tex3d;
	GLuint VBO, VAO, EBO, FBO, PBO;

	GLuint fbo_texture = -1;
	GLuint texture_front = -1;
	GLuint texture_back = -1;
	GLuint fbo_texture_width;
	GLuint fbo_texture_height;

	cudaArray* texArrayFront;
	cudaArray* texArrayBack;
	cudaGraphicsResource* cudaResource[2];
	cudaTextureObject_t	frontTexObject;
	cudaTextureObject_t	backTexObject;

	cudaError_t error;

	float4* pixelPtr;
	cudaGraphicsResource* cudaResourcePBO;

	const string vertex_shader = "shader.vs";
	const string fragment_shader = "shader.fs";
	GLuint shader_program = -1;

	void ReloadShader();

public:
	
	bool showWater = true;

	Renderer(unsigned int w, unsigned int h);
	int Init(int x, int y, int z);
	void Render(int voxelX, int voxelY, int voxelZ, float height, float angle, float FOV, unsigned char* voxel);
	void End();
};

#endif // !RENDERER_CUH
