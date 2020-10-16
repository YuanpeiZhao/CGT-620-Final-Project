#include "Simulator.cuh"
#include "Renderer.cuh"
#include "Shader.h"

#include <stdio.h>
#include <string>
#include <iostream>

using namespace std;

#define LAYER_NUMBER 2

int voxelX = 1024;
int voxelY = 1024;
int voxelZ = 1024;

const string matHeightFileNames[LAYER_NUMBER] = { 
	"rockHeightMap.png",
	"sandHeightMap.png" };

string vertexShader = "shader.vs";
string fragmentShader = "shader.fs";

int main() {

	Simulator sim = Simulator(voxelX, voxelY, voxelZ, LAYER_NUMBER);
	sim.Init(matHeightFileNames);

	//int cnt[3] = { 0 };
	//for (int i = 0; i < voxelX; i++) {
	//	for (int j = 0; j < voxelY; j++) {
	//		for (int k = 0; k < voxelZ; k++) {
	//			cnt[(int)sim.voxelState[i * voxelY * voxelZ + j * voxelZ + k]-1] ++;
	//		}
	//		
	//	}
	//	
	//}
	//cout << cnt[0] << ' ' << cnt[1] << ' ' << cnt[2] << endl;

	Renderer ren = Renderer();
	ren.Init(sim.voxelState, voxelX, voxelY, voxelZ);

	Shader shader = Shader(vertexShader.c_str(), fragmentShader.c_str());

	while (!glfwWindowShouldClose(ren.window)) {
		ren.Render(shader);
	}
	
	ren.End();

	return 0;
}