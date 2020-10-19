#include "Simulator.cuh"

#include <stdio.h>
#include <string>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

__global__ void simulateWaterKernel(float* flow, float* heightMap, unsigned char* voxel, int voxelX, int voxelY, int voxelZ, float delta_t, float g) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= voxelX || y >= voxelY) return;

    float w = heightMap[x * voxelY * 2 + y * 2 + 1];
    float d = heightMap[x * voxelY * 2 + y * 2];
    float h = w + d;

    int fidx = 0;
    float vout = 0.0f;
    for (int i = -1; i <= 1; i ++) {
        for (int j = -1; j <= 1; j ++) {
            if ((i == 0 && j == 0) || (i != 0 && j != 0)) continue;

            float w_nei, d_nei, h_nei;
            int idx = (x + i) * voxelY * 2 + (y + j) * 2;

            // boundary condition
            if (idx < 0 || idx + 1 >= voxelX * voxelY * 2 || (heightMap[idx] >= h && heightMap[idx+1] == 0.0f) || (d >= heightMap[idx + 1] + heightMap[idx] && w == 0.0f)) {
                //flow[x * voxelY * 4 + y * 4 + fidx] = 0.0f;
                fidx++;
                continue;
            }
            //w_nei = heightMap[idx+1];
            d_nei = heightMap[idx];
            //h_nei = w_nei + d_nei;

            flow[x * voxelY * 4 + y * 4 + fidx] += delta_t * g * (h - d_nei);
            vout += delta_t * flow[x * voxelY * 4 + y * 4 + fidx];
            fidx++;
        }
    }

    if (vout > w) {
        flow[x * voxelY * 4 + y * 4] *= (w / vout);
        flow[x * voxelY * 4 + y * 4 + 1] *= (w / vout);
        flow[x * voxelY * 4 + y * 4 + 2] *= (w / vout);
        flow[x * voxelY * 4 + y * 4 + 3] *= (w / vout);
    }
    
}

__global__ void simulation_cont(float* flow, float* heightMap, unsigned char* voxel, int voxelX, int voxelY, int voxelZ, float delta_t, float g) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= voxelX || y >= voxelY) return;

    float w = heightMap[x * voxelY * 2 + y * 2 + 1];
    float d = heightMap[x * voxelY * 2 + y * 2];
    float h = w + d;

    int fidx = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if ((i == 0 && j == 0) || (i != 0 && j != 0)) continue;

            int idx = (x + i) * voxelY * 2 + (y + j) * 2;
            // boundary condition
            if (idx < 0 || idx + 1 >= voxelX * voxelY * 2 || (heightMap[idx] >= h && heightMap[idx + 1] == 0.0f) || (d >= heightMap[idx + 1] + heightMap[idx] && w == 0.0f)) continue;

            heightMap[x * voxelY * 2 + y * 2 + 1] += delta_t * (flow[(x + i) * voxelY * 4 + (y + j) * 4 + 3 - fidx] - flow[x * voxelY * 4 + y * 4 + fidx]);
            if (heightMap[x * voxelY * 2 + y * 2 + 1] < 0) heightMap[x * voxelY * 2 + y * 2 + 1] = 0;
            if (heightMap[x * voxelY * 2 + y * 2 + 1] > voxelZ - d - 1) heightMap[x * voxelY * 2 + y * 2 + 1] = voxelZ - d - 1;
        }
    }

    // update voxel
    float w_new = heightMap[x * voxelY * 2 + y * 2 + 1];

    if (w <= w_new) {
        for (int z = d + (int)w; z < d + (int)w_new; z++) {
            voxel[x * voxelY * voxelZ + y * voxelZ + z] = 1;
        }
    }

    else {
        for (int z = d + (int)w_new; z < d + (int)w; z++) {
            voxel[x * voxelY * voxelZ + y * voxelZ + z] = 0;
        }
    }
}

Simulator::Simulator(int x, int y, int z, int layerNum, float delta_t, float g) {
    this->voxelX = x;
    this->voxelY = y;
    this->voxelZ = z;

    this->matLayerNum = layerNum;
    this->delta_t = delta_t;
    this->g = g;
}

void Simulator::InitWaterFixed(unsigned char* voxelState, float* heightState) {
    for (int x = 60; x < 75; x++) {
        for (int y = 60; y < 75; y++) {
            int h = 50;
            for (int z = 0; z < voxelZ; z++) {
                if (h <= 0) break;
                int idx = x * voxelY * voxelZ + y * voxelZ + z;
                if (voxelState[idx] == 0) {
                    voxelState[idx] = 1;
                    h--;
                    heightState[x * voxelY * 2 + y * 2 + 1] +=1.0f;
                }
            }
        }
    }
}

int Simulator::Init(const string fileNames[]) {

    unsigned char* voxelState = new unsigned char[voxelX * voxelY * voxelZ]();
    float* heightState = new float[voxelX * voxelY * 2]();
    float* flow = new float[voxelX * voxelY * 4]();

    unsigned char* img;
    // Load height map and init voxel state;
    for (int i = 0; i < matLayerNum; i++) {

        img = LoadImage(fileNames[i]);

        for (int x = 0; x < voxelX; x++) {
            for (int y = 0; y < voxelY; y++) {
                int imgX = x * imgHeight / voxelX;
                int imgY = y * imgWidth / voxelY;
                int h = img[imgWidth * imgX + imgY] * voxelZ / 255;

                for (int z = 0; z < voxelZ; z++) {
                    if (h <= 0) break;
                    int idx = x * voxelY * voxelZ + y * voxelZ + z;
                    if (voxelState[idx] == 0) {
                        voxelState[idx] = i + 2;
                        h--;
                        heightState[x * voxelY * 2 + y * 2] += 1.0f;
                    }
                }
            }
        }
        free(img);
    }
    
    // Init water initial state
    InitWaterFixed(voxelState, heightState);

    // Init Cuda
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&d_voxelState, voxelX * voxelY * voxelZ * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_heightState, voxelX * voxelY * 2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_flow, voxelX * voxelY * 4 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_voxelState, voxelState, voxelX * voxelY * voxelZ * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(d_heightState, heightState, voxelX * voxelY * 2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(d_flow, flow, voxelX * voxelY * 4 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    free(voxelState);
    free(heightState);
    free(flow);

    return 0;
}

int Simulator::SimulateShallowWater() {

    cudaError_t cudaStatus;

    // Launch a kernel on the GPU.
    const int TILE = 16;
    dim3 dimGrid(ceil((float)imgWidth / TILE), ceil((float)imgHeight / TILE));
    dim3 dimBlock(TILE, TILE, 1);

    simulation_cont << <dimGrid, dimBlock >> > (d_flow, d_heightState, d_voxelState, voxelX, voxelY, voxelZ, delta_t, g);
    simulateWaterKernel<<<dimGrid, dimBlock>>>(d_flow, d_heightState, d_voxelState, voxelX, voxelY, voxelZ, delta_t, g);
    

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "!!! Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return -1;
    }
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