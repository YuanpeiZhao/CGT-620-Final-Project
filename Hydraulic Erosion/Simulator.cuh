#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include <string>

#define EMPTY 0
#define FULL 1
#define MAT_SAND 2
#define MAT_ROCK 3

using namespace std;

struct Material {
	float sedimentCapacityConstant;
	float color[4];
};

class Simulator {

private:
	unsigned char* heightState; // 2*voxelX * voxelY, heightState[x][y][0] for material height, heightState[x][y][0] for water height
	int voxelX, voxelY, voxelZ;
	int matLayerNum;
	int imgWidth, imgHeight;
	
	unsigned char* LoadImage(const string fileName);
	void InitWaterFixed();

public:
	unsigned char* voxelState;	// 3D voxel state
	Simulator(int vX, int vY, int vZ, int layerNum);
	void Init(const string fileNames[]);

};

#endif // !SIMULATOR_CUH
