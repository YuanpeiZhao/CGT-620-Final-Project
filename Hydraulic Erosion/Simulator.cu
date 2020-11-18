#include "Simulator.cuh"

#include <stdio.h>
#include <string>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

__global__ void simulateWaterKernel(double* flow, double* heightMap, unsigned char* voxel, Setting set) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= set.voxelX || y >= set.voxelY) return;

    double w = heightMap[x * set.voxelY * 2 + y * 2 + 1];
    double d = heightMap[x * set.voxelY * 2 + y * 2];
    double h = w + d;



    int fidx = 0;
    double vout = 0.0f;
    for (int i = -1; i <= 1; i ++) {
        for (int j = -1; j <= 1; j ++) {
            if (i == 0 && j == 0) continue;

            double w_nei, d_nei, h_nei;
            int idx = (x + i) * set.voxelY * 2 + (y + j) * 2;

            // boundary condition
            /*if (idx < 0 || idx + 1 >= set.voxelX * set.voxelY * 2 || (heightMap[idx] >= h) || (d >= heightMap[idx + 1] + heightMap[idx] && w == 0.0f)) {
                flow[x * set.voxelY * 8 + y * 8 + fidx] = 0.0f;
                fidx++;
                continue;
            }*/
            if (x + i < 1 || x + i >= set.voxelX - 1 || y + j < 1 || y + j >= set.voxelY - 1) {
                flow[x * set.voxelY * 8 + y * 8 + fidx] = 0.0f;
                fidx++;
                continue;
            }
            d_nei = heightMap[idx];
            flow[x * set.voxelY * 8 + y * 8 + fidx] += set.delta_t * set.g * set.pipeLen * (h - d_nei);
            if (flow[x * set.voxelY * 8 + y * 8 + fidx] < 0.0f) flow[x * set.voxelY * 8 + y * 8 + fidx] = 0.0f;

            vout += set.delta_t * flow[x * set.voxelY * 8 + y * 8 + fidx];
            fidx++;
        }
    }

    double vmax = set.pipeLen * set.pipeLen * w;
    if (vout > vmax) {
        flow[x * set.voxelY * 8 + y * 8] *= (vmax / vout);
        flow[x * set.voxelY * 8 + y * 8 + 1] *= (vmax / vout);
        flow[x * set.voxelY * 8 + y * 8 + 2] *= (vmax / vout);
        flow[x * set.voxelY * 8 + y * 8 + 3] *= (vmax / vout);
        flow[x * set.voxelY * 8 + y * 8 + 4] *= (vmax / vout);
        flow[x * set.voxelY * 8 + y * 8 + 5] *= (vmax / vout);
        flow[x * set.voxelY * 8 + y * 8 + 6] *= (vmax / vout);
        flow[x * set.voxelY * 8 + y * 8 + 7] *= (vmax / vout);
    }
    
}

__global__ void simulation_cont(double* flow, double* heightMap, unsigned char* voxel, double* m, double* mWater, Setting set, int frameCnt) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= set.voxelX || y >= set.voxelY) return;

    double w = heightMap[x * set.voxelY * 2 + y * 2 + 1];
    double d = heightMap[x * set.voxelY * 2 + y * 2];
    double h = w + d;

    double vx = 0.0f, vy = 0.0f, v = 0.0f;

    int fidx = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            int idx = (x + i) * set.voxelY * 2 + (y + j) * 2;
            // boundary condition
            if (x + i < 1 || x + i >= set.voxelX - 1 || y + j < 1 || y + j >= set.voxelY - 1)
            {
                fidx++;
                continue;
            }

            double f = flow[(x + i) * set.voxelY * 8 + (y + j) * 8 + 7 - fidx] - flow[x * set.voxelY * 8 + y * 8 + fidx];
            double transferM = 0.0f;

            if (set.transportation) {
                if (f < 0.0f) {
                    if(w > 5.0f)
                    transferM = (set.delta_t / set.pipeLen / set.pipeLen * f) / w * mWater[x * set.voxelY + y];
                }
                else {
                    if (heightMap[(x + i) * set.voxelY * 2 + (y + j) * 2 + 1] > 5.0f)
                        transferM = (set.delta_t / set.pipeLen / set.pipeLen * f) / heightMap[(x + i) * set.voxelY * 2 + (y + j) * 2 + 1] * mWater[(x + i) * set.voxelY + (y + j)];
                }

                mWater[x * set.voxelY + y] += transferM;
            }

            heightMap[x * set.voxelY * 2 + y * 2 + 1] += set.delta_t / set.pipeLen / set.pipeLen * f;

            vx += f * i;
            vy += f * j;

            if (heightMap[x * set.voxelY * 2 + y * 2 + 1] < 0) heightMap[x * set.voxelY * 2 + y * 2 + 1] = 0;
            if (heightMap[x * set.voxelY * 2 + y * 2 + 1] > set.voxelZ - d - 1) heightMap[x * set.voxelY * 2 + y * 2 + 1] = set.voxelZ - d - 1;

            fidx++;
        }
    }

    double w_new = heightMap[x * set.voxelY * 2 + y * 2 + 1];

    // update voxel
    if (w <= w_new) {
        for (int z = (int)(d+w); z < (int)(d+w_new); z++) {
            voxel[x * set.voxelY * set.voxelZ + y * set.voxelZ + z] = 1;
        }
    }

    else {
        for (int z = (int)(d + w_new); z < (int)(d + w); z++) {
            voxel[x * set.voxelY * set.voxelZ + y * set.voxelZ + z] = 0;
        }
    }

    if (frameCnt) return;

    // max sediment capasity
    v = sqrt(vx * vx + vy * vy) / 4;
    double cMax = v * set.cMaxRate;
    if (cMax < 1.0f) cMax = 1.0f;

    // erosion
    if (set.erosion && w_new > 1.0f && d > 1.0f && mWater[x * set.voxelY + y] < cMax) {
        m[x * set.voxelY + y] -= set.erosionRate;
        mWater[x * set.voxelY + y] += set.erosionRate;

        if (m[x * set.voxelY + y] <= 0.0f) {
            m[x * set.voxelY + y] = 1.0f;
            voxel[x * set.voxelY * set.voxelZ + y * set.voxelZ + (int)d - 1] = 1;
            voxel[x * set.voxelY * set.voxelZ + y * set.voxelZ + (int)(d+w_new) - 1] = 0;
            heightMap[x * set.voxelY * 2 + y * 2] -= 1.0f;
            d -= 1.0f;
            //heightMap[x * set.voxelY * 2 + y * 2 + 1] += 1.0f;
        }
    }

    

    // deposition
    if (set.deposition && w_new > 1.0f) {

        if (mWater[x * set.voxelY + y] >= cMax) {
            mWater[x * set.voxelY + y] -= 1.0f;
            //m[x * voxelY + y] = 1.0f;
            voxel[x * set.voxelY * set.voxelZ + y * set.voxelZ + (int)d] = 4;
            heightMap[x * set.voxelY * 2 + y * 2] += 1.0f;
            //heightMap[x * voxelY * 2 + y * 2 + 1] -= 1.0f;
            voxel[x * set.voxelY * set.voxelZ + y * set.voxelZ + (int)(d + w_new)] = 1;
        }
    }

}

__global__ void MaterialTransform(double* heightMap, double* mWater, double* mDensity, Setting set) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= set.voxelX || y >= set.voxelY) return;

    double w = heightMap[x * set.voxelY * 2 + y * 2 + 1];
    if (w < 2.0f) return;

    double density = mWater[x * set.voxelY + y] / w;

    int fidx = 0;
    double vout = 0.0f;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {

            if (i == 0 && j == 0) continue;
            if (x + i < 1 || x + i >= set.voxelX - 1 || y + j < 1 || y + j >= set.voxelY - 1) {
                fidx++;
                continue;
            }

            double w_nei = heightMap[(x + i) * set.voxelY * 2 + (y + j) * 2 + 1];
            if (w_nei < 2.0f) {
                fidx++;
                continue;
            }

            double den_nei = mWater[(x + i) * set.voxelY + (y + j)] / w_nei;
            mDensity[x * set.voxelY * 8 + y * 8 + fidx] = set.delta_t * set.matTransRate * (density - den_nei);
            if (mDensity[x * set.voxelY * 8 + y * 8 + fidx] < 0.0f) mDensity[x * set.voxelY * 8 + y * 8 + fidx] = 0.0f;

            vout += mDensity[x * set.voxelY * 8 + y * 8 + fidx];
            fidx++;
        }
    }

    if (vout > density) {
        mDensity[x * set.voxelY * 8 + y * 8] *= (density / vout);
        mDensity[x * set.voxelY * 8 + y * 8 + 1] *= (density / vout);
        mDensity[x * set.voxelY * 8 + y * 8 + 2] *= (density / vout);
        mDensity[x * set.voxelY * 8 + y * 8 + 3] *= (density / vout);
        mDensity[x * set.voxelY * 8 + y * 8 + 4] *= (density / vout);
        mDensity[x * set.voxelY * 8 + y * 8 + 5] *= (density / vout);
        mDensity[x * set.voxelY * 8 + y * 8 + 6] *= (density / vout);
        mDensity[x * set.voxelY * 8 + y * 8 + 7] *= (density / vout);
    }
}

__global__ void MaterialTransform_cont(double* heightMap, double* mWater, double* mDensity, Setting set) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= set.voxelX || y >= set.voxelY) return;

    double w = heightMap[x * set.voxelY * 2 + y * 2 + 1];
    if (w < 2.0f) return;

    int fidx = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {

            if (i == 0 && j == 0) continue;
            if (x + i < 1 || x + i >= set.voxelX - 1 || y + j < 1 || y + j >= set.voxelY - 1) {
                fidx++;
                continue;
            }

            double transDen = mDensity[(x + i) * set.voxelY * 8 + (y + j) * 8 + 7 - fidx] - mDensity[x * set.voxelY * 8 + y * 8 + fidx];
            mWater[x * set.voxelY + y] += transDen * w;
            if (mWater[x * set.voxelY + y] < 0.0f) mWater[x * set.voxelY + y] = 0.0f;

            fidx++;
        }
    }
}

__global__ void AddWaterKernel(double* heightMap, unsigned char* voxel, Setting set) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (set.scene == 0) {
        if ((x >= 35 && x < 60) && (y >= 175 && y < 200)) {

            heightMap[x * set.voxelY * 2 + y * 2 + 1] += 1.0f;

            double w = heightMap[x * set.voxelY * 2 + y * 2 + 1];
            double d = heightMap[x * set.voxelY * 2 + y * 2];
            double h = w + d;
            voxel[x * set.voxelY * set.voxelZ + y * set.voxelZ + (int)h -1] = 1;
        }
    }

    else if (set.scene == 1) {
        if ((x >= 103 && x< 153) && ((y >= 40 && y < 60) || (y >=196 && y < 216))) {
            heightMap[x * set.voxelY * 2 + y * 2 + 1] += 1.0f;

            double w = heightMap[x * set.voxelY * 2 + y * 2 + 1];
            double d = heightMap[x * set.voxelY * 2 + y * 2];
            double h = w + d;
            voxel[x * set.voxelY * set.voxelZ + y * set.voxelZ + (int)h-1] = 1;
        }
    }

    else if (set.scene == 2) {
        if (((x >= 150 && x < 200) && ((y >= 50 && y < 100) || (y >= 156 && y < 206))) ||
            (x >= 50 && x < 100) && (y >= 103 && y < 153)) {
            heightMap[x * set.voxelY * 2 + y * 2 + 1] += 1.0f;

            double w = heightMap[x * set.voxelY * 2 + y * 2 + 1];
            double d = heightMap[x * set.voxelY * 2 + y * 2];
            double h = w + d;
            voxel[x * set.voxelY * set.voxelZ + y * set.voxelZ + (int)h-1] = 1;
        }
    }
}

void Simulator::UpdateSetting(Setting set) {

    this->preset.voxelX = set.voxelX;
    this->preset.voxelY = set.voxelY;
    this->preset.voxelZ = set.voxelZ;
    this->preset.layerNum = set.layerNum;
    this->preset.erosionFreqInFrame = set.erosionFreqInFrame;
    this->preset.scene = set.scene;

    this->preset.delta_t = set.delta_t;
    this->preset.g = set.g;
    this->preset.cMaxRate = set.cMaxRate;
    this->preset.erosionRate = set.erosionRate;
    this->preset.pipeLen = set.pipeLen;
    this->preset.matTransRate = set.matTransRate;

    this->preset.erosion = set.erosion;
    this->preset.deposition = set.deposition;
    this->preset.transportation = set.transportation;
}

Simulator::Simulator() {
    frameCnt = 0;
}

void Simulator::InitWater(unsigned char* voxelState, double* heightState) {
    if (preset.scene == 0) {

        for (int x = 35; x < 60; x++) {
            for (int y = 175; y < 200; y++) {
                int h = 100;
                for (int z = 0; z < preset.voxelZ; z++) {
                    if (h <= 0) break;
                    int idx = x * preset.voxelY * preset.voxelZ + y * preset.voxelZ + z;
                    if (voxelState[idx] == 0) {
                        voxelState[idx] = 1;
                        h--;
                        heightState[x * preset.voxelY * 2 + y * 2 + 1] += 1.0f;
                    }
                }
            }
        }
    }

    else if (preset.scene == 1) {
        for (int x = 103; x < 153; x++) {
            for (int y = 40; y < 60; y++) {
                int h = 130;
                for (int z = 0; z < preset.voxelZ; z++) {
                    if (h <= 0) break;
                    int idx = x * preset.voxelY * preset.voxelZ + y * preset.voxelZ + z;
                    if (voxelState[idx] == 0) {
                        voxelState[idx] = 1;
                        h--;
                        heightState[x * preset.voxelY * 2 + y * 2 + 1] += 1.0f;
                    }
                }
            }
        }

        for (int x = 103; x < 153; x++) {
            for (int y = 196; y < 216; y++) {
                int h = 130;
                for (int z = 0; z < preset.voxelZ; z++) {
                    if (h <= 0) break;
                    int idx = x * preset.voxelY * preset.voxelZ + y * preset.voxelZ + z;
                    if (voxelState[idx] == 0) {
                        voxelState[idx] = 1;
                        h--;
                        heightState[x * preset.voxelY * 2 + y * 2 + 1] += 1.0f;
                    }
                }
            }
        }
    }

    else if (preset.scene == 2) {
        for (int x = 150; x < 200; x++) {
            for (int y = 50; y < 100; y++) {
                int h = 80;
                for (int z = 0; z < preset.voxelZ; z++) {
                    if (h <= 0) break;
                    int idx = x * preset.voxelY * preset.voxelZ + y * preset.voxelZ + z;
                    if (voxelState[idx] == 0) {
                        voxelState[idx] = 1;
                        h--;
                        heightState[x * preset.voxelY * 2 + y * 2 + 1] += 1.0f;
                    }
                }
            }
        }

        for (int x = 150; x < 200; x++) {
            for (int y = 156; y < 206; y++) {
                int h = 80;
                for (int z = 0; z < preset.voxelZ; z++) {
                    if (h <= 0) break;
                    int idx = x * preset.voxelY * preset.voxelZ + y * preset.voxelZ + z;
                    if (voxelState[idx] == 0) {
                        voxelState[idx] = 1;
                        h--;
                        heightState[x * preset.voxelY * 2 + y * 2 + 1] += 1.0f;
                    }
                }
            }
        }

        for (int x = 50; x < 100; x++) {
            for (int y = 103; y < 153; y++) {
                int h = 80;
                for (int z = 0; z < preset.voxelZ; z++) {
                    if (h <= 0) break;
                    int idx = x * preset.voxelY * preset.voxelZ + y * preset.voxelZ + z;
                    if (voxelState[idx] == 0) {
                        voxelState[idx] = 1;
                        h--;
                        heightState[x * preset.voxelY * 2 + y * 2 + 1] += 1.0f;
                    }
                }
            }
        }
    }
}

int Simulator::AddWater() {

    cudaError_t cudaStatus;

    // Launch a kernel on the GPU.
    const int TILE = 16;
    dim3 dimGrid(ceil((float)imgWidth / TILE), ceil((float)imgHeight / TILE));
    dim3 dimBlock(TILE, TILE, 1);
    
    AddWaterKernel << <dimGrid, dimBlock >> > (d_heightState, d_voxelState, preset);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "!!! Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
}

int Simulator::Replay(const string fileNames[], Setting set) {
    cudaFree(d_voxelState);
    cudaFree(d_heightState);
    cudaFree(d_flow);
    cudaFree(d_m);
    cudaFree(d_mWater);

    return Init(fileNames, set);
}

int Simulator::Debug() {
    unsigned char* voxelState = new unsigned char[preset.voxelX * preset.voxelY * preset.voxelZ]();
    double* heightState = new double[preset.voxelX * preset.voxelY * 2]();
    double* mWater = new double[preset.voxelX * preset.voxelY]();

    // Init Cuda
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(voxelState, d_voxelState, preset.voxelX * preset.voxelY * preset.voxelZ * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(heightState, d_heightState, preset.voxelX * preset.voxelY * 2 * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(mWater, d_mWater, preset.voxelX * preset.voxelY * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    double water = 0.0f;
    double material = 0.0f;
    for (int i = 1; i < preset.voxelX * preset.voxelY * 2; i += 2) {
        water += heightState[i];
        material += mWater[(i-1)/2];
    }

    

    cout << "Total material in Water: " << material << endl;
    cout << "Water total volume: " << water << endl;
}

int Simulator::Init(const string fileNames[], Setting set) {

    UpdateSetting(set);

    unsigned char* voxelState = new unsigned char[set.voxelX * set.voxelY * set.voxelZ]();
    double* heightState = new double[set.voxelX * set.voxelY * 2]();
    double* flow = new double[set.voxelX * set.voxelY * 8]();
    double* m = new double[set.voxelX * set.voxelY]();
    double* mWater = new double[set.voxelX * set.voxelY]();
    double* mDensity = new double[set.voxelX * set.voxelY * 8]();

    unsigned char* img;
    // Load height map and init voxel state;
    for (int i = 0; i < set.layerNum; i++) {

        img = LoadImage(fileNames[set.scene * set.layerNum + i]);

        for (int x = 1; x < set.voxelX-1; x++) {
            for (int y = 1; y < set.voxelY-1; y++) {

                // init material status
                m[x * preset.voxelY + y] = 1.0f;

                int imgX = x * imgHeight / set.voxelX;
                int imgY = y * imgWidth / set.voxelY;
                int h = img[imgWidth * imgX + imgY] * set.voxelZ / 255;

                for (int z = 0; z < set.voxelZ; z++) {
                    if (h <= 0) break;
                    int idx = x * set.voxelY * set.voxelZ + y * set.voxelZ + z;
                    if (voxelState[idx] == 0) {
                        voxelState[idx] = i + 2;
                        h--;
                        heightState[x * set.voxelY * 2 + y * 2] += 1.0f;
                    }
                }
            }
        }
        free(img);
    }
    
    // Init water initial state
    InitWater(voxelState, heightState);

    // Init Cuda
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&d_voxelState, set.voxelX * set.voxelY * set.voxelZ * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_heightState, set.voxelX * set.voxelY * 2 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_flow, set.voxelX * set.voxelY * 8 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_m, set.voxelX * set.voxelY * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_mWater, set.voxelX * set.voxelY * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_mDensity, set.voxelX * set.voxelY * 8 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_voxelState, voxelState, set.voxelX * set.voxelY * set.voxelZ * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(d_heightState, heightState, set.voxelX * set.voxelY * 2 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(d_flow, flow, set.voxelX * set.voxelY * 8 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(d_mDensity, mDensity, set.voxelX * set.voxelY * 8 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(d_m, m, set.voxelX * set.voxelY * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(d_mWater, mWater, set.voxelX * set.voxelY * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }

    free(voxelState);
    free(heightState);
    free(flow);
    free(m);
    free(mWater);
    free(mDensity);

    return 0;
}

int Simulator::SimulateShallowWater() {

    frameCnt = (frameCnt + 1) % preset.erosionFreqInFrame;
    
    cudaError_t cudaStatus;

    // Launch a kernel on the GPU.
    const int TILE = 16;
    dim3 dimGrid(ceil((float)imgWidth / TILE), ceil((float)imgHeight / TILE));
    dim3 dimBlock(TILE, TILE, 1);

    MaterialTransform << <dimGrid, dimBlock >> > (d_heightState, d_mWater, d_mDensity, preset);
    MaterialTransform_cont << <dimGrid, dimBlock >> > (d_heightState, d_mWater, d_mDensity, preset);
    
    simulateWaterKernel<<<dimGrid, dimBlock>>>(d_flow, d_heightState, d_voxelState, preset);
    simulation_cont << <dimGrid, dimBlock >> > (d_flow, d_heightState, d_voxelState, d_m, d_mWater, preset, frameCnt);

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