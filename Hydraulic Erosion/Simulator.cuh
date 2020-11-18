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

struct Setting {

	int voxelX;
	int voxelY;
	int voxelZ;
	int layerNum;
	int scene;

	int erosionFreqInFrame;

	double delta_t;
	double g;
	float cMaxRate;
	float erosionRate;
	float pipeLen;
	float matTransRate;

	bool erosion;
	bool deposition;
	bool transportation;

	float camHeight;
	float camAngle;
	float camFOV;
};

class Simulator {

private:
	double* d_heightState;		// 2*voxelX * voxelY, heightState[x][y][0] for material height, heightState[x][y][1] for water height
	double* d_flow;
	double* d_m;
	double* d_mWater;
	double* d_mDensity;

	int frameCnt;
	double erosionRate;
	int imgWidth, imgHeight;

	unsigned char* LoadImage(const string fileName);
	void InitWater(unsigned char* voxelState, double* heightState);

public:
	unsigned char* d_voxelState;
	//bool erosion = false;
	//bool deposition = false;
	//bool transportation = false;
	Setting preset;

	Simulator();
	int Init(const string fileNames[], Setting set);
	int SimulateShallowWater();
	int Replay(const string fileNames[], Setting set);
	int Debug();
	void UpdateSetting(Setting set);
	int AddWater();
};

#endif // !SIMULATOR_CUH
