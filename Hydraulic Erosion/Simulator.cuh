#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include <windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cuda_gl_interop.h>
#include <string>

#define EMPTY 0
#define FULL 1
#define MAT_ROCK 2
#define MAT_SAND 3

using namespace std;

struct Material {
	float sedimentCapacityConstant;
	float color[4];
};

class Simulator {

private:
	float* d_heightState;		// 2*voxelX * voxelY, heightState[x][y][0] for material height, heightState[x][y][0] for water height
	float* d_flow;

	float delta_t;
	float g;

	int voxelX, voxelY, voxelZ;
	int matLayerNum;
	int imgWidth, imgHeight;
	
	unsigned char* LoadImage(const string fileName);
	void InitWaterFixed(unsigned char* voxelState, float* heightState);

public:
	unsigned char* d_voxelState;

	Simulator(int vX, int vY, int vZ, int layerNum, float delta_t, float g);
	int Init(const string fileNames[]);
	int SimulateShallowWater();
};

#endif // !SIMULATOR_CUH
