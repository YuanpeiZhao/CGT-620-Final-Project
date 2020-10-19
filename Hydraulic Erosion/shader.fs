#version 460 core

layout(location = 3) uniform int pass;

in vec3 vPos;

layout(binding = 0) uniform sampler2D frontfaces_tex;
layout(binding = 1) uniform sampler2D backfaces_tex;

out vec4 FragColor;

const vec3 lightPos = vec3(5.0, 3.0, 1.0);

const vec4 La = vec4(0.75, 0.75, 0.75, 1.0);
const vec4 Ld = vec4(0.74, 0.74, 0.74, 1.0);
const vec4 Ls = vec4(1.0, 1.0, 0.74, 1.0);

const vec4 Ka = vec4(0.4, 0.4, 0.34, 1.0);
const vec4 Kd = vec4(1.0, 1.0, 0.73, 1.0);
const vec4 Ks = vec4(0.1, 0.1, 0.073, 1.0);

vec4 raytracedcolor(vec3 rayStart, vec3 rayStop);
vec4 renderMat(vec4 matColor, vec3 pos, vec3 rayDir);
vec4 renderWater();

vec3 normal(vec3 pos);

void main()
{
    if(pass == 1 || pass == 2)
	{
		FragColor = vec4((vPos), 1.0); //write cube positions to texture
	}
	else if(pass == 3) 
	{
		//vec3 rayStart = vPos.xyz;
		//vec3 rayStop = texelFetch(backfaces_tex, ivec2(gl_FragCoord.xy), 0).xyz;
		//FragColor = raytracedcolor(rayStart, rayStop);
		FragColor = texelFetch(frontfaces_tex, ivec2(gl_FragCoord), 0);
	}
}

//vec4 raytracedcolor(vec3 rayStart, vec3 rayStop)
//{
//	const int MaxSamples = 1024; //max number of steps along ray
//
//	vec3 rayDir = normalize(rayStop-rayStart);	//ray direction unit vector
//	float travel = distance(rayStop, rayStart);	
//	float stepSize = travel/MaxSamples;	//initial raymarch step size
//	vec3 pos = rayStart;				//position along the ray
//	vec3 step = rayDir*stepSize;		//displacement vector along ray
//	
//	for (int i=0; i < MaxSamples && travel > 0.0; ++i, pos += step, travel -= stepSize)
//	{
//		vec3 texCoord = pos / 2.0f + vec3(0.5f);
//		float t = texture(voxel_tex, texCoord).x;
//
//		if(t == 1.0f / 255.0f) return renderMat(vec4(0.5f, 1.0f , 1.0f, 1.0f), pos, rayDir);
//		else if(t == 2.0f / 255.0f) return renderMat(vec4(0.5f, 0.5f, 0.5f, 1.0f), pos, rayDir);
//	}
//	//If the ray never intersects the scene then output clear color
//	return vec4(0.2f, 0.3f, 0.3f, 1.0f);
//}
//
////Compute lighting on the raycast surface using Phong lighting model
//vec4 renderMat(vec4 matColor, vec3 pos, vec3 rayDir) 
//{
//	const vec3 light = normalize(lightPos-pos); //light direction from surface
//	vec3 n = normal(pos);
//
//	float diff = max(0.0, dot(n, light));             
//
//	return La*matColor + Ld*matColor*diff;	
//
//}
//
//vec3 normal(vec3 pos)
//{
//	vec3 texCoord = pos / 2.0f + vec3(0.5f);
//
//	const float h = 1.0f / 255.0f;
//	const float t = 1.0f / 1024.0f;
//
//	vec3 n = vec3(0.0f);
//
//	for(int i=-1; i<=1; i++){
//		for(int j=-1; j<=1;j++){
//			for(int k=-1; k<=1; k++){
//				if(texture(voxel_tex, texCoord + vec3(i, j, k) * t).x == 0.0f) n += vec3(i, j, k);
//			}
//		}
//	}
//
//	return normalize(n);
//}






//__device__ float3 normal(float3 pos, int voxelX, int voxelY, int voxelZ, unsigned char* voxel)
//{
//    float3 texCoord = add(time(pos, 0.5f), make_float3(0.5f, 0.5f, 0.5f));
//
//	float3 n = make_float3(0.0f, 0.0f, 0.0f);
//
//	for(int i=-1; i<=1; i++){
//		for(int j=-1; j<=1;j++){
//			for(int k=-1; k<=1; k++){
//
//                int X = clamp((int)(texCoord.x * voxelX + i), 0, voxelX - 1);
//                int Y = clamp((int)(texCoord.y * voxelY + j), 0, voxelY - 1);
//                int Z = clamp((int)(texCoord.z * voxelZ + k), 0, voxelZ - 1);
//
//                unsigned char t = voxel[Z * voxelY * voxelZ + Y * voxelZ + X];
//
//				if(t == 0) n = add(n, make_float3(i, j, k));
//			}
//		}
//	}
//
//	return normalize(n, make_float3(0.0f, 0.0f, 0.0f));
//}
//
//__device__ float3 RenderMat(float3 color, float3 pos, float3 rayDir, int voxelX, int voxelY, int voxelZ, unsigned char* voxel) {
//
//    float4 La = make_float4(0.75, 0.75, 0.75, 1.0);
//    float4 Ld = make_float4(0.74, 0.74, 0.74, 1.0);
//    float3 lightPos = make_float3(5.0, 3.0, 1.0);
//
//    float3 light = normalize(lightPos, pos); //light direction from surface
//	float3 n = make_float3(0.75, 0.75, 0.75); //normal(pos, voxelX, voxelY, voxelZ, voxel);
//
//	float diff = n.x*light.x + n.y*light.y + n.z+light.z;
//    if (diff < 0.0) diff = 0.0;
//
//	return add(make_float3(La.x*color.x, La.y * color.y, La.z * color.z), make_float3(Ld.x * color.x * diff, Ld.y * color.y * diff, Ld.z * color.z * diff));
//}