#include "Renderer.cuh"
#include "Simulator.cuh"

#include <time.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glut.h"

using namespace std;

#define LAYER_NUMBER 2

unsigned int SCR_WIDTH = 1280;
unsigned int SCR_HEIGHT = 720;

Setting set;

void initParameters(Setting* set) {

	set->voxelX = 256;
	set->voxelY = 256;
	set->voxelZ = 256;
	set->layerNum = 2;
	set->erosionFreqInFrame = 5;
	set->scene = 0;

	set->delta_t = 0.004f;
	set->g = 9.8f;
	set->cMaxRate = 0.1f;
	set->erosionRate = 0.004f;
	set->pipeLen = 0.2f;
	set->matTransRate = 0.5f;

	set->erosion = true;
	set->deposition = true;
	set->transportation = true;

	set->camHeight = 3.0f;
	set->camAngle = 0.0f;
	set->camFOV = 0.75f;
}

const string matHeightFileNames[LAYER_NUMBER * 3] = { 
	"rockHeightMap1.png",
	"sandHeightMap.png",
	"rockHeightMap2.png",
	"sandHeightMap.png",
	"rockHeightMap3.png",
	"sandHeightMap.png",
};

Simulator sim = Simulator();
Renderer ren = Renderer(SCR_WIDTH, SCR_HEIGHT);

//Draw the user interface using ImGui
void draw_gui()
{
	ImGui_ImplGlut_NewFrame();

	ImGui::Text("Press and hold 'a' to add water");
	if (ImGui::Button("Replay"))
	{
		sim.Replay(matHeightFileNames, set);
	}

	if (ImGui::Button("Debug"))
	{
		sim.Debug();
	}

	ImGui::Checkbox("Show water", &ren.showWater);

	ImGui::Text("Camera");

	ImGui::SliderFloat("Height", &set.camHeight, -10.0f, +10.0f);
	ImGui::SliderFloat("View angle", &set.camAngle, -3.141592f, +3.141592f);
	ImGui::SliderFloat("FOV", &set.camFOV, 0.0f, +3.141592f);

	ImGui::Text("Need Replay to update the setting below");

	ImGui::RadioButton("Scene 0", &set.scene, 0);
	ImGui::RadioButton("Scene 1", &set.scene, 1);
	ImGui::RadioButton("Scene 2", &set.scene, 2);

	ImGui::SliderFloat("Pipe Length", &set.pipeLen, 0.01f, 2.0f);
	ImGui::SliderFloat("Erosion Rate", &set.erosionRate, 0.001f, 0.1f);
	ImGui::SliderFloat("Water Capasity Rate", &set.cMaxRate, 0.01f, 1.0f);
	ImGui::SliderFloat("Material Transform Rate", &set.matTransRate, 0.1f, 10.0f);
	ImGui::SliderInt("Erosion Frame Step", &set.erosionFreqInFrame, 3, 10);
	ImGui::Checkbox("Erosion", &set.erosion);
	if (set.erosion) {
		ImGui::Checkbox("Deposition", &set.deposition);
		if (set.deposition) {
			ImGui::Checkbox("Mass transfer", &set.transportation);
		}
		else {
			set.transportation = false;
		}
	}
	else {
		set.deposition = false;
		set.transportation = false;
	}

	//ImGui::ShowDemoWindow();
	ImGui::Render();
}

void Idle()
{
	glutPostRedisplay();
}

void Display(void)
{
	//long t1 = clock();

	sim.SimulateShallowWater();
	ren.Render(set.voxelX, set.voxelY, set.voxelZ, set.camHeight, set.camAngle, set.camFOV, sim.d_voxelState);

	draw_gui();
	glutSwapBuffers();

	//long t2 = clock();

	//cout << 1000 / (t2 - t1 + 1) << endl;
}


void keyboard(unsigned char key, int x, int y)
{
	ImGui_ImplGlut_KeyCallback(key);
	if (key == 'a') {
		sim.AddWater();
	}

}

void keyboard_up(unsigned char key, int x, int y)
{
	ImGui_ImplGlut_KeyUpCallback(key);
}

void special_up(int key, int x, int y)
{
	ImGui_ImplGlut_SpecialUpCallback(key);
}

void passive(int x, int y)
{
	ImGui_ImplGlut_PassiveMouseMotionCallback(x, y);
}

void special(int key, int x, int y)
{
	ImGui_ImplGlut_SpecialCallback(key);
}

void motion(int x, int y)
{
	ImGui_ImplGlut_MouseMotionCallback(x, y);
}

void mouse(int button, int state, int x, int y)
{
	ImGui_ImplGlut_MouseButtonCallback(button, state);
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

	ImGui_ImplGlut_Init(); // initialize the imgui system

	initParameters(&set);
	sim.Init(matHeightFileNames, set);
	ren.Init(set.voxelX, set.voxelY, set.voxelZ);

	glutDisplayFunc(Display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutKeyboardUpFunc(keyboard_up);
	glutSpecialUpFunc(special_up);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(motion);
	glutIdleFunc(Idle);
	glutMainLoop();

	return 0;
}