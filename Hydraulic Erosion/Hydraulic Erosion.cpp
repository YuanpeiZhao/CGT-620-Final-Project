#include "Renderer.cuh"
#include "Simulator.cuh"

#include <time.h>
#include <stdio.h>
#include <string>
#include <iostream>

using namespace std;

#define LAYER_NUMBER 2

unsigned int SCR_WIDTH = 1280;
unsigned int SCR_HEIGHT = 720;

int voxelX = 128;
int voxelY = 128;
int voxelZ = 128;

float delta_t = 0.01f;
float gravity = 9.8f;

const string matHeightFileNames[LAYER_NUMBER] = { 
	"rockHeightMap.png",
	"sandHeightMap.png" };

Simulator sim = Simulator(voxelX, voxelY, voxelZ, LAYER_NUMBER, delta_t, gravity);
Renderer ren = Renderer(SCR_WIDTH, SCR_HEIGHT);

void Idle()
{
	glutPostRedisplay();

	//long t1 = clock();

	sim.SimulateShallowWater();
	ren.Render(voxelX, voxelY, voxelZ, sim.d_voxelState);

	//long t2 = clock();

	//cout << 1000 / (t2 - t1 + 1) << endl;
}

void Display(void)
{
}

int main(int argc, char** argv) {

	glutInitWindowSize(SCR_WIDTH, SCR_HEIGHT);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow("CUDA OpenGL interoperability");
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));

	}

	sim.Init(matHeightFileNames);
	ren.Init(voxelX, voxelY, voxelZ);	

	glutDisplayFunc(Display);
	glutIdleFunc(Idle);
	glutMainLoop();

	return 0;
}