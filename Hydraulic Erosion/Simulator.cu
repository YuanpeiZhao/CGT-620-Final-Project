#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "cuda_gl_interop.h"

#include "Simulator.cuh"

#include <stdio.h>
#include <string>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

Simulator::Simulator(int x, int y, int z, int layerNum) {
    this->voxelX = x;
    this->voxelY = y;
    this->voxelZ = z;

    this->matLayerNum = layerNum;
}

void Simulator::InitWaterFixed() {
    for (int x = 400; x < 600; x++) {
        for (int y = 400; y < 600; y++) {
            int h = 200;
            for (int z = 0; z < voxelZ; z++) {
                if (h <= 0) break;
                int idx = x * voxelY * voxelZ + y * voxelZ + z;
                if (voxelState[idx] == 0) {
                    voxelState[idx] = 1;
                    h--;
                    heightState[x * voxelY * 2 + voxelY * 2 + 1] ++;
                }
            }
        }
    }
}

void Simulator::Init(const string fileNames[]) {

    voxelState = new unsigned char[voxelX * voxelY * voxelZ]();
    heightState = new unsigned char[voxelX * voxelY * 2]();

    unsigned char* img;
    // Load height map and init voxel state;
    for (int i = 0; i < matLayerNum; i++) {

        img = LoadImage(fileNames[i]);

        for (int x = 0; x < voxelX; x++) {
            for (int y = 0; y < voxelY; y++) {
                int imgX = x * imgHeight / voxelX;
                int imgY = y * imgWidth / voxelY;
                int h = img[imgWidth * imgX + imgY];

                for (int z = 0; z < voxelZ; z++) {
                    if (h <= 0) break;
                    int idx = x * voxelY * voxelZ + y * voxelZ + z;
                    if (voxelState[idx] == 0) {
                        voxelState[idx] = i + 2;
                        h--;
                        heightState[x * voxelY * 2 + voxelY * 2] ++;
                    }
                }
            }
        }
        free(img);
    }
    
    // Init water initial state
    InitWaterFixed();
}

unsigned char* Simulator::LoadImage(const string file) {

    string fileName = "images/" + file;
    int components;

    unsigned char* result = stbi_load(fileName.c_str(), &(imgWidth), &(imgHeight), &components, 1);
    if (!result) {
        fprintf(stderr, "Cannot read input image, invalid path?\n");
        exit(-1);
    }

    return result;
}