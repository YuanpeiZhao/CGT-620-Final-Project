#include "Renderer.cuh"
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>


//#include <thrust/device_vector.h>

texture<float4, cudaTextureType2D, cudaReadModeElementType> texRefFront;
texture<float4, cudaTextureType2D, cudaReadModeElementType> texRefBack;

__device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 time(float3 a, float t) {
    return make_float3(a.x * t, a.y * t, a.z * t);
}

__device__ float distance(float3 a, float3 b) {
    float3 tmp = sub(a, b);
    return sqrt(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z);
}

__device__ float3 normalize(float3 a, float3 b) {
    float3 tmp = sub(a, b);
    if (distance(a, b) == 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    return time(tmp, 1.0f / distance(a, b));
}

__device__ int clamp(int x, int a, int b) {
    if (x < a) return a;
    if (x > b) return b;
    return x;
}

__device__ float3 normal(float3 pos, int voxelX, int voxelY, int voxelZ, unsigned char* voxel)
{
    float3 texCoord = add(time(pos, 0.5f), make_float3(0.5f, 0.5f, 0.5f));

	float3 n = make_float3(0.0f, 0.0f, 0.0f);

	for(int i=-1; i<=1; i++){
		for(int j=-1; j<=1;j++){
			for(int k=-1; k<=1; k++){

                int X = clamp((int)(texCoord.x * voxelX + i), 0, voxelX - 1);
                int Y = clamp((int)(texCoord.y * voxelY + j), 0, voxelY - 1);
                int Z = clamp((int)(texCoord.z * voxelZ + k), 0, voxelZ - 1);

                unsigned char t = voxel[Z * voxelY * voxelZ + Y * voxelZ + X];

				if(t == 0) n = add(n, make_float3(i, j, k));
			}
		}
	}

	return normalize(n, make_float3(0.0f, 0.0f, 0.0f));
}

__device__ float3 RenderMat(float3 color, float3 pos, float3 rayDir, int voxelX, int voxelY, int voxelZ, unsigned char* voxel) {

    float4 La = make_float4(0.75, 0.75, 0.75, 1.0);
    float4 Ld = make_float4(0.74, 0.74, 0.74, 1.0);
    float3 lightPos = make_float3(5.0, 3.0, 1.0);

    float3 light = normalize(lightPos, pos); //light direction from surface
    float3 n = normal(pos, voxelX, voxelY, voxelZ, voxel);

	float diff = n.x*light.x + n.y*light.y + n.z+light.z;
    if (diff < 0.0) diff = 0.0;

	return add(make_float3(La.x*color.x, La.y * color.y, La.z * color.z), make_float3(Ld.x * color.x * diff, Ld.y * color.y * diff, Ld.z * color.z * diff));
}

__global__ void rayMarching(float4* pixelPtr, unsigned char* voxel, int width, int height, int voxelX, int voxelY, int voxelZ, bool showWater) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float u = x / (float)width;
    float v = y / (float)height; 

    float4 rayStart4 = tex2D(texRefFront, u, v);
    float4 rayStop4 = tex2D(texRefBack, u, v);

    float3 rayStart = make_float3(rayStart4.x, rayStart4.y, rayStart4.z);
    float3 rayStop = make_float3(rayStop4.x, rayStop4.y, rayStop4.z);

    int MaxSamples = 1024; //max number of steps along ray
    float waterSurfRate = 0.6f;
    float waterStepRate = 0.0f;
    float matRate = 0.4f;

    float travel = distance(rayStart, rayStop);

    if (travel < 0.001f) {
        pixelPtr[x + y * width] = make_float4(0.2f, 0.3f, 0.4f, 1.0f);
        return;
    }

	float3 rayDir = normalize(rayStop, rayStart);	//ray direction unit vector
		
	float stepSize = travel/MaxSamples;	//initial raymarch step size
	float3 pos = rayStart;				//position along the ray
	float3 step = time(rayDir, stepSize);		//displacement vector along ray

    float3 waterSurf = make_float3(0.2f, 0.3f, 0.4f);
    float3 result = make_float3(0.2f, 0.3f, 0.4f);
    float3 waterColor = make_float3(0.2f, 0.5f, 0.5f);
    bool hitWater = false;
	
	for (int i=0; i < MaxSamples && travel > 0.0001; ++i, pos = add(pos, step), travel -= stepSize)
	{
		float3 texCoord = add(time(pos, 0.5f), make_float3(0.5f, 0.5f, 0.5f));
        int X = clamp((int)(texCoord.x * voxelX), 0, voxelX - 1);
        int Y = clamp((int)(texCoord.y * voxelY), 0, voxelY - 1);
        int Z = clamp((int)(texCoord.z * voxelZ), 0, voxelZ - 1);

        unsigned char t = voxel[Z * voxelY * voxelZ + Y * voxelZ + X];

        if (t == 1) {
            if(!showWater) continue;

            if (hitWater == false) {
                hitWater = true;
                waterSurf = RenderMat(make_float3(0.2f, 0.5f, 0.5f), pos, rayDir, voxelX, voxelY, voxelZ, voxel);

            }
            else {
                waterStepRate += 1.0f / MaxSamples;
            }
            /*float3 result = RenderMat(make_float3(0.2f, 0.5f, 0.5f), pos, rayDir, voxelX, voxelY, voxelZ, voxel);
            pixelPtr[x + y * width] = make_float4(result.x ,result.y, result.z, 1.0f);
            return;*/
        }

        else if (t == 2) {
            result = RenderMat(make_float3(0.5f, 0.5f, 0.5f), pos, rayDir, voxelX, voxelY, voxelZ, voxel);
            if (hitWater == false) waterSurf = result; 
            break;
            /*pixelPtr[x + y * width] = make_float4(result.x, result.y, result.z, 1.0f);
            return;*/
        }

        else if (t == 3) {
            result = RenderMat(make_float3(0.5f, 0.5f, 0.2f), pos, rayDir, voxelX, voxelY, voxelZ, voxel);
            if (hitWater == false) waterSurf = result;
            break;
            /*pixelPtr[x + y * width] = make_float4(result.x, result.y, result.z, 1.0f);
            return;*/
        }

        else if (t == 4) {
            result = RenderMat(make_float3(0.5f, 0.2f, 0.2f), pos, rayDir, voxelX, voxelY, voxelZ, voxel);
            if (hitWater == false) waterSurf = result;
            break;
            /*pixelPtr[x + y * width] = make_float4(result.x, result.y, result.z, 1.0f);
            return;*/
        }
	}
    float xx = (result.x * matRate + waterSurf.x * waterSurfRate) * (1.0f - waterStepRate) + waterColor.x * waterStepRate;
    float yy = (result.y * matRate + waterSurf.y * waterSurfRate) * (1.0f - waterStepRate) + waterColor.y * waterStepRate;
    float zz = (result.z * matRate + waterSurf.z * waterSurfRate) * (1.0f - waterStepRate) + waterColor.z * waterStepRate;
	//If the ray never intersects the scene then output clear color
    pixelPtr[x + y * width] = make_float4(xx, yy, zz, 1.0f);
	return;
}

Renderer::Renderer(unsigned int w, unsigned int h) {
    SCR_WIDTH = w;
    SCR_HEIGHT = h;
    fbo_texture_width = w;
    fbo_texture_height = h;
}

int Renderer::Init(int x, int y, int z) {

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    ReloadShader();

    // set up vertex data (and buffer(s)) and configure vertex attributes
    float vertices[] = {
        // positions         
        -1.0f, -1.0f, -1.0f,  
        -1.0f, -1.0f, +1.0f,
        -1.0f, +1.0f, -1.0f,  
        -1.0f, +1.0f, +1.0f,
        +1.0f, -1.0f, -1.0f,  
        +1.0f, -1.0f, +1.0f,
        +1.0f, +1.0f, -1.0f,
        +1.0f, +1.0f, +1.0f
    };

    unsigned int indices[] = {
        0, 2, 6, 
        0, 6, 4,
        2, 3, 7,
        2, 7, 6,
        4, 6, 7,
        4, 7, 5,
        1, 5, 7,
        1, 7, 3,
        0, 1, 3,
        0, 3, 2,
        0, 4 ,5,
        0, 5, 1
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Create a texture object and set initial wrapping and filtering state.
    glGenTextures(1, &fbo_texture);
    glBindTexture(GL_TEXTURE_2D, fbo_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, fbo_texture_width, fbo_texture_height, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &texture_front);
    glBindTexture(GL_TEXTURE_2D, texture_front);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, fbo_texture_width, fbo_texture_height, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &texture_back);
    glBindTexture(GL_TEXTURE_2D, texture_back);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, fbo_texture_width, fbo_texture_height, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create the framebuffer object
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    //attach the texture we just created to color attachment 1
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture, 0);
    //unbind the fbo
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    cudaGraphicsGLRegisterImage(&cudaResource[0], texture_back, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
    cudaGraphicsGLRegisterImage(&cudaResource[1], texture_front, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);

    //// Set PBO and cuda pixel texture

    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    //this actually creates the buffer
    glBufferData(GL_PIXEL_UNPACK_BUFFER, fbo_texture_width * fbo_texture_height * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    //CUDA part
    error = cudaGraphicsGLRegisterBuffer(&cudaResourcePBO, PBO, cudaGraphicsMapFlagsNone);
    if (error != cudaSuccess) printf("Something went wrong: %s\n", cudaGetErrorString(error));

    return 0;
}

void Renderer::Render(int voxelX, int voxelY, int voxelZ, float height, float angle, float FOV, unsigned char* voxel)
{
    glEnable(GL_CULL_FACE);

    const int w = glutGet(GLUT_WINDOW_WIDTH);
    const int h = glutGet(GLUT_WINDOW_HEIGHT);
    const float aspect_ratio = float(w) / float(h);

    // render the triangle
    glUseProgram(shader_program);

    glm::mat4 M = glm::mat4(1.0f);
    glm::mat4 V = glm::lookAt(glm::vec3(3.0f, 3.0f, height), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)) * glm::rotate(angle, glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 P = glm::perspective(FOV, aspect_ratio, 0.1f, 100.0f);

    glUniformMatrix4fv(UniformLoc::M, 1, false, glm::value_ptr(M));
    glUniformMatrix4fv(UniformLoc::V, 1, false, glm::value_ptr(V));
    glUniformMatrix4fv(UniformLoc::P, 1, false, glm::value_ptr(P));

    //// Pass 1
    glUniform1i(UniformLoc::Pass, 1);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO); // Render to FBO.
    glDrawBuffer(GL_COLOR_ATTACHMENT0); //Out variable in frag shader will be written to the texture attached to GL_COLOR_ATTACHMENT0.

    //Clear the FBO attached texture.
    glClear(GL_COLOR_BUFFER_BIT);

    glCullFace(GL_BACK);   // Don't draw front faces

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    // copy FBO texture to free texture
    glBindTexture(GL_TEXTURE_2D, texture_front);
    /// copy from framebuffer (here, the FBO!) to the bound texture
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, fbo_texture_width, fbo_texture_height);

    //// Pass 2
    glUniform1i(UniformLoc::Pass, 2);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO); // Render to FBO.

    //Clear the FBO attached texture.
    glClear(GL_COLOR_BUFFER_BIT);

    glCullFace(GL_FRONT);   // Don't draw front faces

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    // copy FBO texture to free texture
    glBindTexture(GL_TEXTURE_2D, texture_back);
    /// copy from framebuffer (here, the FBO!) to the bound texture
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, fbo_texture_width, fbo_texture_height);

    //// Use cuda to do ray marching
    error = cudaGraphicsMapResources(2, cudaResource, 0);
    if (error != cudaSuccess) printf("Something went wrong: %s\n", cudaGetErrorString(error));
    error = cudaGraphicsSubResourceGetMappedArray(&texArrayBack, cudaResource[0], 0, 0);
    if (error != cudaSuccess) printf("Something went wrong: %s\n", cudaGetErrorString(error));
    error = cudaGraphicsSubResourceGetMappedArray(&texArrayFront, cudaResource[1], 0, 0);
    if (error != cudaSuccess) printf("Something went wrong: %s\n", cudaGetErrorString(error));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    // Set texture parameters
    texRefFront.addressMode[0] = cudaAddressModeClamp;
    texRefFront.addressMode[1] = cudaAddressModeClamp;
    texRefFront.filterMode = cudaFilterModeLinear;
    texRefFront.normalized = true;

    texRefBack.addressMode[0] = cudaAddressModeClamp;
    texRefBack.addressMode[1] = cudaAddressModeClamp;
    texRefBack.filterMode = cudaFilterModeLinear;
    texRefBack.normalized = true;

    // Bind the array to the texture reference
    error = cudaBindTextureToArray(texRefBack, texArrayBack, channelDesc);
    if (error != cudaSuccess) printf("Something went wrong: %s\n", cudaGetErrorString(error));
    error = cudaBindTextureToArray(texRefFront, texArrayFront, channelDesc);
    if (error != cudaSuccess) printf("Something went wrong: %s\n", cudaGetErrorString(error));

    const int TILE = 16;
    dim3 dimGrid(ceil((float)fbo_texture_width / TILE), ceil((float)fbo_texture_height / TILE));
    dim3 dimBlock(TILE, TILE);

    size_t size;
    error = cudaGraphicsMapResources(1, &cudaResourcePBO, 0);
    if (error != cudaSuccess) printf("Something went wrong: %s\n", cudaGetErrorString(error));

    error = cudaGraphicsResourceGetMappedPointer((void**)&pixelPtr, &size, cudaResourcePBO);
    if (error != cudaSuccess) printf("Something went wrong: %s\n", cudaGetErrorString(error));

    rayMarching << <dimGrid, dimBlock >> > (pixelPtr, voxel, fbo_texture_width, fbo_texture_height, voxelX, voxelY, voxelZ, showWater);

    error = cudaGraphicsUnmapResources(1, &cudaResourcePBO, 0);
    if (error != cudaSuccess) printf("Something went wrong: %s\n", cudaGetErrorString(error));

    cudaGraphicsUnmapResources(1, &cudaResource[0], 0);
    cudaGraphicsUnmapResources(1, &cudaResource[1], 0);

    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // Do not render the next pass to FBO.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glDrawPixels(fbo_texture_width, fbo_texture_height, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    //glutSwapBuffers();

    //glUseProgram(shader_program);
    //glDisable(GL_CULL_FACE);

    //// Pass 3

    //glUseProgram(shader_program);

    //glUniform1i(UniformLoc::Pass, 3);
    //
    //glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, fbo_texture_back);
    ////Bind texture so we can read the texture in the shader
    //glActiveTexture(GL_TEXTURE1);
    //glBindTexture(GL_TEXTURE_2D, fbo_texture_front);

    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //Clear the back buffer

    //glCullFace(GL_BACK);    // Don't draw back faces
    //glBindVertexArray(VAO);
    //glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    //glutSwapBuffers();
}

void Renderer::End() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

void Renderer::ReloadShader()
{
    GLuint new_shader = InitShader(vertex_shader.c_str(), fragment_shader.c_str());

    if (new_shader == -1) // loading failed
    {
        glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
    }
    else
    {
        glClearColor(0.15f, 0.15f, 0.15f, 0.0f);

        if (shader_program != -1)
        {
            glDeleteProgram(shader_program);
        }
        shader_program = new_shader;
    }
}