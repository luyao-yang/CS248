#include "webgpu.h"
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
// You can use glm or using your implementation in previous assignments for calculation
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>
#define GLg_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform2.hpp>
#define STB_IMAGE_IMPLEMENTATION

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "rasterization/Vec3.h"
#include "rasterization/Matrix4.h"
#include "rasterization/Light.h"
#include "rasterization/Color.h"
#include "rasterization/Camera.h"
#include "rasterization/Mesh.h"

#include <thread>
#include <chrono>
#include <stdlib.h>
#include <set>
#include <list>
#include <vector>

using namespace std;

//----------- WEBGPU variables ---------------------------------------

WGPUDevice device;
WGPUQueue queue;
WGPUSwapChain swapchain;

WGPURenderPipeline pipeline;
WGPUBuffer vertBuf; // vertex buffer with triangle position and colours
WGPUBuffer indxBuf; // index buffer


// TODO  新定义的uniforms
WGPUBuffer Buf; // Matrices buffer
WGPUBuffer CamBuf;
WGPUBuffer LitBuf;

WGPUBindGroup* bindGroup;

WGPUTexture depthTexture;
WGPUTextureView depthView;
WGPUSampler depthSampler;

WGPUTexture tex; // Texture
WGPUSampler samplerTex;
WGPUTextureView texView;
WGPUExtent3D texSize = {};
WGPUTextureDescriptor texDesc = {};
WGPUTextureDataLayout texDataLayout = {};
WGPUImageCopyTexture texCopy = {};


/**
 * Current rotation angle (in degrees, updated per frame).
 */

float rotDeg = 0.0f;

// const int viewWidth = 1024, viewHeight = 768;

std::string simple_vert_wgsl;
std::string simple_frag_wgsl;

vector <glm::vec3> points;
vector <glm::vec3> colors;
// 定义得到的点的向量
vector <glm::vec3> vnormals;
vector <glm::vec2> textures;

glm::vec3 tangent;
vector <glm::vec3> tangents;

Mesh sphere;
Mesh beacon;
Mesh teapot;
Mesh object;

Matrix4 sphereModel;
Matrix4 normalSphere;

Matrix4 beaconModel;
Matrix4 normalBeacon;

Matrix4 teapotModel;
Matrix4 normalTeapot;

Matrix4 objModel;
Matrix4 normalObj;

//----------- Camera -----------------------------------------------
Vec3 eye_postion(0.0f, 0.0f, 1.0f);
Vec3 gaze_direction(0.0f, 0.0f, -1.0f);
Vec3 view_up(0.0f, 1.0f, 0.0f);
Camera camera(eye_postion, gaze_direction, view_up);

const int viewWidth = 1024, viewHeight = 768;
const double viewLeft = -2.0f, viewRight = 2.0f, viewBottom = -1.5f, viewTop = 1.5f;
const double aspect = (double)viewWidth / (double)viewHeight;
const double FoV = 45.f;
const double nearPlane = -1.0f, farPlane = -1000.0f;
unsigned char* img = new unsigned char[viewHeight * viewWidth * 4];

Material emerald(Color(0.0215f, 0.1745f, 0.0215f), Color(0.07568f, 0.61424f, 0.07568f),
                 Color(0.633f, 0.727811f, 0.633f), 76.8f);

Matrix4 View = camera.calculateViewMatrix();
Matrix4 Projection = camera.calculatePerspectiveMatrix(FoV, aspect, nearPlane, farPlane);

Light light(Vec3(5.0f, 5.0f, 0.0f), Color(1.0f));
Matrix4 lightModel;


float temp[16*4];

Vec3 computeUV(Vec3 point){
    int type=0;
    Vec3 pc=point-Vec3(0.0);
    Vec3 result=Vec3(0.0);

    if (type == 0){
        // sphere
        result.x = float(0.5 + (atan2(pc.y, pc.x)) / (2*3.14159));
        result.y = float(acos( pc.z / sqrt(pc.x*pc.x+pc.y*pc.y+pc.z*pc.z) )/3.14159);

    } else if (type == 1){
        // cylindrical
        result.x = float(0.5 + (atan2(pc.z, pc.x)) / (2*3.14159));
        result.y = float(0.5 + pc.y/2.0);

    }else if (type == 2){
        // perspective mapping
        Vec4 temp = Projection*View*Vec4(pc.x, pc.y, pc.z, 1.0);
        result.x = float(temp.x/temp.w)/2+0.5;
        result.y = float(temp.y/temp.w)/2+0.5;
    }

    return result;
}

// This is just an example vertex data
void setVertexData()
{
    // 首先研究一下球体
//    object=sphere;
    vector<Vec3> s_points=object.points;
    vector<Vec3> s_normals=object.normals;

    // 设定一个color的空值，传入一个空的颜色
    // colors.resize(s_points.size());

    for(int i=0; i<s_points.size();i++){
        // cout<<sphere.points[i].y<<endl;
        points.emplace_back(glm::vec3(s_points[i].x,s_points[i].y,s_points[i].z));
        colors.emplace_back(glm::vec3(0.0f, 1.0f, 1.0f));
        vnormals.emplace_back(glm::vec3(s_normals[i].x,s_normals[i].y,s_normals[i].z));
        //############################# tangent #############################
        tangent = glm::normalize(glm::vec3(-s_normals[i].y,s_normals[i].x,0.0));

        if(s_normals[i].x == 0 && s_normals[i].y == 0){
            tangent = glm::vec3(-1.0,0.0,0.0);
        }

        Vec3 result= computeUV(Vec3(s_points[i].x,s_points[i].y,s_points[i].z));
        textures.emplace_back(glm::vec2(result.x,result.y));

        tangents.emplace_back(tangent);

    }
}

static void setupShader()
{
    // Simple shaders
    simple_vert_wgsl = R"(

		[[block]]
		struct VertexIn {
            [[builtin(instance_index)]] instanceIdx : u32;
			[[location(0)]] aPos : vec3<f32>;
			[[location(1)]] aColor : vec3<f32>;
			// ADD
			[[location(2)]] aNorm : vec3<f32>;
            [[location(3)]] aTex : vec2<f32>;
            [[location(4)]] aTan : vec3<f32>;
		};
		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] vColor : vec3<f32>;
			// ADD
			[[location(1)]] vPos : vec3<f32>;
			[[location(2)]] vNorm : vec3<f32>;
            [[location(3)]] vTex : vec2<f32>;
            [[location(4)]] vTan : vec3<f32>;
		};

		[[block]]
		struct Matrices {
			// degs : f32;
			Objmodel: mat4x4<f32>;
			Normodel: mat4x4<f32>;
			Proj: mat4x4<f32>;
			View: mat4x4<f32>;
            instanceObjmodel : [[stride(64)]] array<mat4x4<f32>, 2>;
            instanceObjnorm : [[stride(64)]] array<mat4x4<f32>, 2>;
		};
		[[group(0), binding(0)]] var<uniform> m : Matrices;

		[[block]]
		struct Camera {
			// 80
			position : vec3<f32>;
            lookAt : vec3<f32>;
            u : vec3<f32>;
            v : vec3<f32>;
            w : vec3<f32>;
		};

		[[group(0), binding(1)]] var<uniform> camera : Camera;

		[[block]]
		struct Light {
			// 96
			position : vec3<f32>;
            intensity : vec3<f32>;
			ka : vec3<f32>;
            kd : vec3<f32>;
            ks : vec3<f32>;
			shiness : vec3<f32>;
		};
		[[group(0), binding(2)]] var<uniform> light : Light;

        [[group(1), binding(0)]] var tex: texture_2d<f32>;
        [[group(1), binding(1)]] var sam: sampler;

		[[stage(vertex)]]
		fn main(input : VertexIn) -> VertexOut {

			// ############################### switch Phong or Guod ############################
			let guod=0;
            let ins=0;

			var output : VertexOut;
            var instanceModel : mat4x4<f32> = m.instanceObjmodel[input.instanceIdx];
            var instanceNormal : mat4x4<f32> = m.instanceObjnorm[input.instanceIdx];

            var after_pos : vec4<f32> = transpose(m.Objmodel)*vec4<f32>(input.aPos,1.0);
            var after_norm : vec4<f32> = transpose(m.Normodel)*vec4<f32>(input.aNorm, 0.0);
            //############################ bump tangent vector ###############################
            var after_tan : vec4<f32> = transpose(m.Objmodel)*vec4<f32>(input.aTan,0.0);

            // ############################## Instance ######################################
            if(ins==1){
                after_pos  = transpose(instanceModel)*vec4<f32>(input.aPos,1.0);
                after_norm  = transpose(instanceNormal)*vec4<f32>(input.aNorm, 0.0);
            }

			var p : vec3<f32> = after_pos.xyz;
            var v : vec3<f32> = normalize(camera.position - p);
            var l : vec3<f32> = normalize(light.position - p);
            var n : vec3<f32> = normalize(after_norm.xyz);
            var h : vec3<f32> = normalize(l+v);

			output.Position = transpose(m.Proj)*transpose(m.View)*vec4<f32>(p,1.0);
			output.Position = output.Position/output.Position.w;

			// ############################## Phong Shading ######################################

			output.vColor = input.aColor;


			// ############################## Guod Shading ######################################
			if(guod==1){
				output.vColor = input.aColor*(light.ka*light.intensity + light.kd*light.intensity*max(dot(n, l), 0.0)
				+ light.ks*light.intensity*pow(max(dot(n, h), 0.0), light.shiness.x));
			}

			output.vPos = after_pos.xyz;
			output.vNorm = after_norm.xyz;
			// ADD
            output.vTex = input.aTex;
            // ########################### tangent vector ##########################################
            output.vTan = after_tan.xyz;
			return output;
		}
	)";

    simple_frag_wgsl = R"(
        [[group(1), binding(0)]] var tex: texture_2d<f32>;
        [[group(1), binding(1)]] var sam: sampler;

		[[block]]
		struct Matrices {
			// degs : f32;
			Objmodel: mat4x4<f32>;
			Normodel: mat4x4<f32>;
			Proj: mat4x4<f32>;
			View: mat4x4<f32>;
            instanceObjmodel : [[stride(64)]] array<mat4x4<f32>, 2>;
            instanceObjnorm : [[stride(64)]] array<mat4x4<f32>, 2>;
		};
		[[group(0), binding(0)]] var<uniform> m : Matrices;

		[[block]]
		struct Camera {
			// 80
			position : vec3<f32>;
            lookAt : vec3<f32>;
            u : vec3<f32>;
            v : vec3<f32>;
            w : vec3<f32>;
		};

		[[group(0), binding(1)]] var<uniform> camera : Camera;

		[[block]]
		struct Light {
			// 96
			position : vec3<f32>;
            intensity : vec3<f32>;
			ka : vec3<f32>;
            kd : vec3<f32>;
            ks : vec3<f32>;
			shiness : vec3<f32>;
		};
		[[group(0), binding(2)]] var<uniform> light : Light;

		struct FragOut {
			[[location(0)]] oColor : vec4<f32>;
		};
		// var ocolor : vec4<f32>;

		[[stage(fragment)]]
		fn main([[location(0)]] vColor : vec3<f32>, [[location(1)]] vPos : vec3<f32> ,
		[[location(2)]] vNorm : vec3<f32>, [[location(3)]] vTex : vec2<f32>,
        [[location(4)]] vTan : vec3<f32>)
		-> [[location(0)]] vec4<f32> {
			let phong=0;
            let te=0;
            let proced=0;
            let onlytex=0;
            let bump=1;

            var p : vec3<f32> = vPos;
            var v : vec3<f32> = normalize(camera.position - p);
            var l : vec3<f32> = normalize(light.position - p);
            var n : vec3<f32> = normalize(vNorm);
            var h : vec3<f32> = normalize(l+v);
			// var output : FragOut;
			if(phong==1){


				return vec4<f32>(vColor*(light.ka*light.intensity +
				light.kd*light.intensity*max(dot(n, l), 0.0) +
				light.ks*light.intensity*pow(max(dot(n, h), 0.0), light.shiness.x)), 1.0);
			}
            if(onlytex==1){
                 return textureSample(tex,sam,vTex);
            }
            if(te==1){

                var textureColor : vec3<f32> = textureSample(tex, sam, vTex).xyz;


                return vec4<f32>(
                textureColor*light.intensity*max(dot(n, l), 0.0) +
                textureColor*light.intensity*pow(max(dot(n, h), 0.0), light.shiness.x), 1.0);


                }
            if(bump==1){
                var p1 : vec3<f32> = vPos;
                var v1 : vec3<f32> = normalize(camera.position - p1);
                var l1 : vec3<f32> = normalize(light.position - p1);

                //################# vTan ####################
                var tan1 : vec3<f32> = normalize(vTan);
                var tan2 : vec3<f32> = normalize(cross(n,tan1));
                var h1 : vec3<f32> = normalize(l1+v1);

                var bump_c : vec4<f32> = textureSample(tex, sam, vTex);
                bump_c = bump_c-vec4<f32>(0.5,0.5,0.0,0.0);
                var scale : f32 = 20.0/1.0;
                n = normalize(n+scale*bump_c.x*tan1-scale*bump_c.y*tan2);

                var brown : vec3<f32> = vec3<f32>(0.8, 0.4, 0.2);

                // with phong shading version
                return vec4<f32>((0.1*brown + 0.8*brown*max(dot(n, l1), 0.0) +
                 0.2*brown*pow(max(dot(n, h1), 0.0), light.shiness.x)), 1.0);

            }
            if(proced==1){
                var scale : f32 = 50.0;
                var pattern : f32 = (cos(vTex.x * 2.0 * 3.14 * scale) + 1.0) * 0.5;
                return vec4<f32>(vec3<f32>(1.0,0.3,0.6)*pattern*light.intensity, 1.0);
            }

			return vec4<f32>(vColor,1.0);

		}
	)";

}


/**
 * Helper to create a shader from WGSL source.
 *
 * \param[in] code WGSL shader source
 * \param[in] label optional shader name
 */
static WGPUShaderModule createShader(const char* const code, const char* label = nullptr) {
    WGPUShaderModuleWGSLDescriptor wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgsl.source = code;
    WGPUShaderModuleDescriptor desc = {};
    desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl);
    desc.label = label;
    return wgpuDeviceCreateShaderModule(device, &desc);
}

/**
 * Helper to create a buffer.
 *
 * \param[in] data pointer to the start of the raw data
 * \param[in] size number of bytes in \a data
 * \param[in] usage type of buffer
 */
static WGPUBuffer createBuffer(const void* data, size_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor desc = {};
    desc.usage = WGPUBufferUsage_CopyDst | usage;
    desc.size = size;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
    wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
    return buffer;
}


/**
 * Helper to create a texture.
 *
 * \param[in]
 * \param[in]
 * \param[in]
 */

static WGPUTexture createTexture(unsigned char* data, unsigned int w, unsigned int h) {
    texSize.depthOrArrayLayers = 1;
    texSize.height = h;
    texSize.width = w;

    texDesc.sampleCount = 1;
    texDesc.mipLevelCount = 1;
    texDesc.dimension = WGPUTextureDimension_2D;
    texDesc.size = texSize;
    texDesc.usage = WGPUTextureUsage_Sampled | WGPUTextureUsage_CopyDst;
    texDesc.format = WGPUTextureFormat_RGBA8Unorm;

    texDataLayout.offset = 0;
    texDataLayout.bytesPerRow = 4 * w;
    texDataLayout.rowsPerImage = h;

    texCopy.texture = wgpuDeviceCreateTexture(device, &texDesc);

    wgpuQueueWriteTexture(queue, &texCopy, data, w * h * 4, &texDataLayout, &texSize);
    return texCopy.texture;
//    return nullptr;
}


static void createDepthTexture() {
    WGPUExtent3D textSizeDepth;
    textSizeDepth.depthOrArrayLayers = 1;
    textSizeDepth.height = viewHeight;
    textSizeDepth.width = viewWidth;
    WGPUTextureDescriptor textureDescriptorDepth = {};
    textureDescriptorDepth.sampleCount = 1;
    textureDescriptorDepth.mipLevelCount = 1;
    textureDescriptorDepth.dimension = WGPUTextureDimension_2D;
    textureDescriptorDepth.size = textSizeDepth;
    textureDescriptorDepth.usage = WGPUTextureUsage_RenderAttachment;
    textureDescriptorDepth.format = WGPUTextureFormat_Depth32Float;
    depthTexture = wgpuDeviceCreateTexture(device, &textureDescriptorDepth);


    WGPUTextureViewDescriptor texViewDescDepth = {};
    depthView = wgpuTextureCreateView(depthTexture, &texViewDescDepth);
    return;
}

/**
 * Bare minimum pipeline to draw a triangle using the above shaders.
 */
static void createPipelineAndBuffers() {
    // compile shaders
    // NOTE: these are now the WGSL shaders (tested with Dawn and Chrome Canary)

    setupShader();

    WGPUShaderModule vertMod = createShader(simple_vert_wgsl.c_str());
    WGPUShaderModule fragMod = createShader(simple_frag_wgsl.c_str());


    // bind group layout for uniform
    // ##############################  Layout: Uniform  #####################################
    WGPUBufferBindingLayout buf = {};
    buf.type = WGPUBufferBindingType_Uniform;

    WGPUBindGroupLayoutEntry bglEntry = {};
    bglEntry.binding = 0;
    bglEntry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    bglEntry.buffer = buf;
    bglEntry.sampler = { 0 };

    WGPUBufferBindingLayout buf1 = {};
    buf1.type = WGPUBufferBindingType_Uniform;

    WGPUBindGroupLayoutEntry bglEntry1 = {};
    bglEntry1.binding = 1;
    bglEntry1.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    bglEntry1.buffer = buf1;
    bglEntry1.sampler = {0 };

    WGPUBufferBindingLayout buf2 = {};
    buf2.type = WGPUBufferBindingType_Uniform;

    WGPUBindGroupLayoutEntry bglEntry2 = {};
    bglEntry2.binding = 2;
    bglEntry2.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    bglEntry2.buffer = buf2;
    bglEntry2.sampler = {0 };

    WGPUBindGroupLayoutEntry* allbglEntry = new WGPUBindGroupLayoutEntry[3];
    allbglEntry[0] = bglEntry;
    allbglEntry[1] = bglEntry1;
    allbglEntry[2] = bglEntry2;

    WGPUBindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 3;
    bglDesc.entries = allbglEntry;

    WGPUBindGroupLayout bindGroupLayoutUniform = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    // ##############################  Layout: Texture & Sampler  #####################################
    tex = createTexture(img, viewWidth, viewHeight);

    WGPUTextureViewDescriptor texViewDesc = {};
    texViewDesc.dimension = WGPUTextureViewDimension_2D;
    texViewDesc.format = WGPUTextureFormat_RGBA8Unorm;

    texView = wgpuTextureCreateView(tex, &texViewDesc);

    WGPUSamplerDescriptor samplerDesc = {};
    samplerDesc.addressModeU = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeV = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeW = WGPUAddressMode_ClampToEdge;
    samplerDesc.magFilter = WGPUFilterMode_Linear;
    samplerDesc.minFilter = WGPUFilterMode_Nearest;
    samplerDesc.mipmapFilter = WGPUFilterMode_Nearest;
    samplerDesc.lodMaxClamp = 32;
    samplerDesc.lodMinClamp = 0;
    samplerDesc.compare = WGPUCompareFunction_Undefined;
    samplerDesc.maxAnisotropy = 1;

    samplerTex = wgpuDeviceCreateSampler(device, &samplerDesc);

    WGPUSamplerBindingLayout samplerLayout = {};
    samplerLayout.type = WGPUSamplerBindingType_Filtering;

    WGPUTextureBindingLayout texLayout = {};
    texLayout.sampleType = WGPUTextureSampleType_Float;
    texLayout.viewDimension = WGPUTextureViewDimension_2D;
    texLayout.multisampled = false;

    WGPUBindGroupLayoutEntry bglTexEntry = {};
    bglTexEntry.binding = 0;
    bglTexEntry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    bglTexEntry.texture = texLayout;

    WGPUBindGroupLayoutEntry bglSamplerEntry = {};
    bglSamplerEntry.binding = 1;
    bglSamplerEntry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    bglSamplerEntry.sampler = samplerLayout;

    WGPUBindGroupLayoutEntry* allBgLayoutEntries = new WGPUBindGroupLayoutEntry[2];
    allBgLayoutEntries[0] = bglTexEntry;
    allBgLayoutEntries[1] = bglSamplerEntry;

    WGPUBindGroupLayoutDescriptor bglDesc1 = {};
    bglDesc1.entryCount = 2;
    bglDesc1.entries = allBgLayoutEntries;
    WGPUBindGroupLayout bindGroupLayoutUniform1 = wgpuDeviceCreateBindGroupLayout(device, &bglDesc1);


    // ########################### Layout: Pipeline ####################################
    // add all uniform layouts (shading + texture mapping)
    int numUniformLayouts = 2;
    WGPUBindGroupLayout* allLayouts = new WGPUBindGroupLayout[numUniformLayouts];
//    for (int i = 0; i < numUniformLayouts; i++)
    allLayouts[0] = bindGroupLayoutUniform;
    allLayouts[1] = bindGroupLayoutUniform1;

    // pipeline layout (used by the render pipeline, released after its creation): remember to add all uniform layout to pipeline layout
    WGPUPipelineLayoutDescriptor layoutDesc = {};
    layoutDesc.bindGroupLayoutCount = numUniformLayouts;
    layoutDesc.bindGroupLayouts = allLayouts;
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);

    // ########################### Layout: Vertex ####################################
    // describe vertex buffer layouts: Need to care about the memory layout
    WGPUVertexAttribute vertAttrs[5] = {};
    vertAttrs[0].format = WGPUVertexFormat_Float32x3;
    vertAttrs[0].offset = 0;
    vertAttrs[0].shaderLocation = 0;
    vertAttrs[1].format = WGPUVertexFormat_Float32x3;
    vertAttrs[1].offset = 4 * sizeof(float);
    vertAttrs[1].shaderLocation = 1;
    vertAttrs[2].format = WGPUVertexFormat_Float32x3;
    vertAttrs[2].offset = 4*2 * sizeof(float);
    vertAttrs[2].shaderLocation = 2;
    // TODO texture
    vertAttrs[3].format = WGPUVertexFormat_Float32x2;
    vertAttrs[3].offset = 4*3 * sizeof(float);
    vertAttrs[3].shaderLocation = 3;

    //Tangent
    vertAttrs[4].format = WGPUVertexFormat_Float32x2;
    vertAttrs[4].offset = 14 * sizeof(float);
    vertAttrs[4].shaderLocation = 4;

    WGPUVertexBufferLayout vertexBufferLayout = {};
    vertexBufferLayout.arrayStride = 18 * sizeof(float);
    vertexBufferLayout.attributeCount = 5;
    vertexBufferLayout.attributes = vertAttrs;

    //#################### choose the object ############################
//    object = sphere;
//    objModel = sphereModel;
//    normalObj = normalSphere;

    setVertexData();
    float* vertData = new float[points.size() * 18];
    int index = 0;


    // Memory layout
    for (int i = 0; i < points.size(); i++)
    {
        vertData[index] = float(points[i].x);
        vertData[index + 1] = float(points[i].y);
        vertData[index + 2] = float(points[i].z);
        vertData[index + 3] = 0.0f;
        vertData[index + 4] = float(colors[i].x);
        vertData[index + 5] = float(colors[i].y);
        vertData[index + 6] = float(colors[i].z);
        vertData[index + 7] = 1.0f;
        vertData[index + 8] = float(vnormals[i].x);
        vertData[index + 9] = float(vnormals[i].y);
        vertData[index + 10] = float(vnormals[i].z);
        vertData[index + 11] = 0.0f;

        vertData[index + 12] = float(textures[i].x);
        vertData[index + 13] = float(textures[i].y);

        //##################### texture #########################

        vertData[index + 14] = float(tangents[i].x);
        vertData[index + 15] = float(tangents[i].y);
        vertData[index + 16] = float(tangents[i].z);
        vertData[index + 17] = 0.0f;

        index += 18;
    }

    uint16_t *indxData = new uint16_t[object.faces.size()];
    // 得到一系列vertexindex的值
    for(int i=0; i<object.faces.size(); i++){
        // cout<<object.faces[i]<<endl;
        indxData[i]=(uint16_t)(object.faces[i]-1);
    }

    // ################################# create cam light matric buffer and bind group #################################
    vertBuf = createBuffer(vertData, points.size() * 18 * sizeof(float), WGPUBufferUsage_Vertex);
    indxBuf = createBuffer(indxData, object.faces.size() * sizeof(uint16_t), WGPUBufferUsage_Index);

    // TODO 将三个矩阵写进去
    // ObjmBuf = createBuffer(&objModel, sizeof(objModel),WGPUBufferUsage_Uniform);
    // ProjBuf = createBuffer(&Projection, sizeof(Projection),WGPUBufferUsage_Uniform);
    // CamBuf = createBuffer(&View, sizeof(View),WGPUBufferUsage_Uniform);

    // create the uniform bind group (note 'rotDeg' is copied here, not bound in any way)
    // TODO 创建矩阵的buffer
    // cout<<(sizeof(objModel)+sizeof(normalObj)+sizeof(Projection)+sizeof(View))/2<<endl;
    Buf = createBuffer("", 512, WGPUBufferUsage_Uniform);
    CamBuf = createBuffer("",80,WGPUBufferUsage_Uniform);
    LitBuf = createBuffer("",96,WGPUBufferUsage_Uniform);

    WGPUBindGroupEntry bgEntry = {};
    bgEntry.binding = 0;
    bgEntry.buffer = Buf;
    bgEntry.offset = 0;
    bgEntry.size = 512;

    WGPUBindGroupEntry bgEntry1 = {};
    bgEntry1.binding = 1;
    bgEntry1.buffer = CamBuf;
    bgEntry1.offset = 0;
    bgEntry1.size = 80;

    WGPUBindGroupEntry bgEntry2 = {};
    bgEntry2.binding = 2;
    bgEntry2.buffer = LitBuf;
    bgEntry2.offset = 0;
    bgEntry2.size = 96;

    WGPUBindGroupEntry* allbgEntry = new WGPUBindGroupEntry[3];
    allbgEntry[0] = bgEntry;
    allbgEntry[1] = bgEntry1;
    allbgEntry[2] = bgEntry2;

    WGPUBindGroupDescriptor bgDesc = {};
    bgDesc.layout = bindGroupLayoutUniform;
    bgDesc.entryCount = 3;
    bgDesc.entries = allbgEntry;

    // ################################# Texture and bind group #################################
    // TODO For the texture part
    WGPUBindGroupEntry bgTexEntry = {};
    bgTexEntry.binding = 0;
    bgTexEntry.textureView = texView;

    WGPUBindGroupEntry bgSamplerEntry = {};
    bgSamplerEntry.binding = 1;
    bgSamplerEntry.sampler = samplerTex;

    WGPUBindGroupEntry* allBgEntries1 = new WGPUBindGroupEntry[2];
    allBgEntries1[0] = bgTexEntry;
    allBgEntries1[1] = bgSamplerEntry;

    WGPUBindGroupDescriptor bgDes1 = {};
    bgDes1.layout = bindGroupLayoutUniform1;
    bgDes1.entryCount = 2;
    bgDes1.entries = allBgEntries1;

    bindGroup = new WGPUBindGroup[numUniformLayouts];
    bindGroup[0] = wgpuDeviceCreateBindGroup(device, &bgDesc);
    bindGroup[1] = wgpuDeviceCreateBindGroup(device, &bgDes1);

    // last bit of clean-up
    wgpuBindGroupLayoutRelease(bindGroupLayoutUniform);
    wgpuBindGroupLayoutRelease(bindGroupLayoutUniform1);


    // Vertex state
    WGPUVertexState vertex = {};
    vertex.module = vertMod;
    vertex.entryPoint = "main";
    vertex.bufferCount = 1;
    vertex.buffers = &vertexBufferLayout;


    // Fragment state
    WGPUBlendState blend = {};
    blend.color.operation = WGPUBlendOperation_Add;
    blend.color.srcFactor = WGPUBlendFactor_One;
    blend.color.dstFactor = WGPUBlendFactor_Zero;
    blend.alpha.operation = WGPUBlendOperation_Add;
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_Zero;


    WGPUColorTargetState colorTarget = {};
    colorTarget.format = webgpu::getSwapChainFormat(device);
    colorTarget.blend = &blend;
    colorTarget.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragment = {};
    fragment.module = fragMod;
    fragment.entryPoint = "main";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;


#ifdef __EMSCRIPTEN__
    WGPURenderPipelineDescriptor desc = {};
#else
    WGPURenderPipelineDescriptor desc = {};
#endif
    desc.vertex = vertex;
    desc.fragment = &fragment;
    desc.layout = pipelineLayout;
    // Other states

    // Primitive state
    desc.primitive.frontFace = WGPUFrontFace_CCW;
    desc.primitive.cullMode = WGPUCullMode_None;
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;

    // Depth Stencil state: You can add depth test in here
    createDepthTexture();
    WGPUDepthStencilState wgpuDepthStencilState = {};
    wgpuDepthStencilState.format = WGPUTextureFormat_Depth32Float;
    wgpuDepthStencilState.depthWriteEnabled = true;
    wgpuDepthStencilState.depthCompare = WGPUCompareFunction_Greater;
    wgpuDepthStencilState.stencilBack.compare = WGPUCompareFunction_Greater;
    wgpuDepthStencilState.stencilFront.compare = WGPUCompareFunction_Greater;
    wgpuDepthStencilState.depthBias = 1.0;
    desc.depthStencil = &wgpuDepthStencilState;


    // Multisample state
    desc.multisample.count = 1;
    desc.multisample.mask = 0xFFFFFFFF;
    desc.multisample.alphaToCoverageEnabled = false;


#ifdef __EMSCRIPTEN__
    pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
#else
    pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
#endif

    // partial clean-up
    wgpuPipelineLayoutRelease(pipelineLayout);

    wgpuShaderModuleRelease(fragMod);
    wgpuShaderModuleRelease(vertMod);
}

// TODO将四个需要转化的矩阵提前放到一个连续的内存空间里
void Change(Matrix4 m1, Matrix4 m2, Matrix4 m3, Matrix4 m4){
    int c=0;
    int flag=1;
    Matrix4 t=m1;
    while(c<63){
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                temp[c]=t.matrix[i][j];
                c+=1;
            }
        }
        flag=flag+1;
        if(flag==2){
            t=m2;
        }
        if(flag==3){
            t=m3;
        }
        if(flag==4){
            t=m4;
        }
    }
}

/**
 * Draws using the above pipeline and buffers.
 */
static bool redraw() {
    WGPUTextureView backBufView = wgpuSwapChainGetCurrentTextureView(swapchain);			// create textureView

    WGPURenderPassColorAttachment colorDesc = {};
    colorDesc.view = backBufView;
    colorDesc.loadOp = WGPULoadOp_Clear;
    colorDesc.storeOp = WGPUStoreOp_Store;
    colorDesc.clearColor.r = 0.0f;
    colorDesc.clearColor.g = 0.0f;
    colorDesc.clearColor.b = 0.0f;
    colorDesc.clearColor.a = 1.0f;

    // You can add depth texture in here
    WGPURenderPassDepthStencilAttachment stencilDesc = {};
    stencilDesc.view = depthView;
    stencilDesc.depthLoadOp = WGPULoadOp_Clear;
    stencilDesc.depthStoreOp = WGPUStoreOp_Store;
    stencilDesc.clearDepth = 0.0f;

    WGPURenderPassDescriptor renderPass = {};
    renderPass.colorAttachmentCount = 1;
    renderPass.colorAttachments = &colorDesc;
    renderPass.depthStencilAttachment = &stencilDesc;


    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);			// create encoder
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPass);	// create pass

    // TODO
    sphereModel=Matrix4::Translate(Vec3(0,0,-5));
    normalSphere=(Matrix4::TranslateInv(Vec3(0,0,-5))).transpose();

    // TODO Decide to show different objects : sphere
    // TODO switch choices
    Matrix4 R1 = Matrix4::Rotation(Vec3(0,3.14/100,0));

    objModel = objModel*R1;
    normalObj=R1*normalObj;

    // TODO 将四个矩阵写入到buffer里
    Matrix4 m1,n1,m2,n2;
    m1 = Matrix4::Translate(Vec3(-1,0,0))*Matrix4::RotationX(20);
    m2 = Matrix4::Translate(Vec3(1,0,0));
    n1 = (Matrix4::RotationXInv(20)*Matrix4::Translate(Vec3(1,0,0))).transpose();
    n2 = Matrix4::Translate(Vec3(-1,0,0)).transpose();

    Change(objModel,normalObj,Projection,View);
    wgpuQueueWriteBuffer(queue, Buf, 0, &temp, 256);
//    cout<<m1<<endl;
    Change(m2,m1,n2,n1);
    wgpuQueueWriteBuffer(queue, Buf, 256, &temp, 256);

    // Write Camera Buffer
    camera.setCameraFrame();
    float* camPos = new float[3];
    camPos[0]=camera.eye_position.x;
    camPos[1]=camera.eye_position.y;
    camPos[2]=camera.eye_position.z;
    wgpuQueueWriteBuffer(queue, CamBuf, 0, camPos, 12);

    // light and coefficients
    float* lightPosition = new float[3];
    lightPosition[0] = light.position.x;
    lightPosition[1] = light.position.y;
    lightPosition[2] = light.position.z;
    float* lightIntensity = new float[3];
    lightIntensity[0] = light.intensity.red;
    lightIntensity[1] = light.intensity.green;
    lightIntensity[2] = light.intensity.blue;

    float* coefficients = new float[16];
    coefficients[0] = emerald.Ka.red;
    coefficients[1] = emerald.Ka.green;
    coefficients[2] = emerald.Ka.blue;
    coefficients[3] = 1.0f;
    coefficients[4] = emerald.Kd.red;
    coefficients[5] = emerald.Kd.green;
    coefficients[6] = emerald.Kd.blue;
    coefficients[7] = 1.0f;
    coefficients[8] = emerald.Ks.red;
    coefficients[9] = emerald.Ks.green;
    coefficients[10] = emerald.Ks.blue;
    coefficients[11] = 1.0f;
    coefficients[12] = emerald.shininess;
    coefficients[13] = 0.0f;
    coefficients[14] = 1.0f;
    coefficients[15] = 0.0f;

    wgpuQueueWriteBuffer(queue, LitBuf, 0, lightPosition, 12);
    wgpuQueueWriteBuffer(queue, LitBuf, 16, lightIntensity, 12);
    wgpuQueueWriteBuffer(queue, LitBuf, 32, coefficients, 64);

    // wrute Texture buffer
    wgpuQueueWriteTexture(queue, &texCopy, img, viewHeight * viewWidth * 4, &texDataLayout, &texSize);

    // wgpuQueueWriteBuffer(queue, LitBuf, 256, &temp, 256);

    // Change(objModel);
    // wgpuQueueWriteBuffer(queue, Buf, 0, &temp, 256);

    // cout<<(sizeof(normalObj)+sizeof(objModel)+sizeof(Projection)+sizeof(View))/2<<endl;

    // draw the object (comment these five lines to simply clear the screen
    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup[0], 0, 0);
    wgpuRenderPassEncoderSetBindGroup(pass, 1, bindGroup[1], 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertBuf, 0, 0);
    wgpuRenderPassEncoderSetIndexBuffer(pass, indxBuf, WGPUIndexFormat_Uint16, 0, 0);
    // Instancing checking
    wgpuRenderPassEncoderDrawIndexed(pass, object.faces.size(), 2, 0, 0, 0);

    wgpuRenderPassEncoderEndPass(pass);
    wgpuRenderPassEncoderRelease(pass);														// release pass
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);				// create commands
    wgpuCommandEncoderRelease(encoder);														// release encoder

    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);														// release commands

#ifndef __EMSCRIPTEN__
    wgpuSwapChainPresent(swapchain);
#endif
    wgpuTextureViewRelease(backBufView);													// release textureView

    return true;
}



extern "C" int __main__(int /*argc*/, char* /*argv*/[]) {

    // TODO
    sphere.loadOBJ("../../data/sphere.obj");
    sphere.generateFaceNormals(
            sphere.points,
            sphere.pointlink,
            sphere.faces,
            sphere.faceNormals);
    sphere.generateVertexNormals(
            sphere.points,
            sphere.normals,
            sphere.faces);

    // // TODO
    sphereModel=Matrix4::Translate(Vec3(0,0,-5));
    normalSphere=(Matrix4::TranslateInv(Vec3(0,0,-5))).transpose();

    // TODO Decide to show different objects : sphere
    // TODO switch choices
    object=sphere;
    objModel=sphereModel;
    normalObj=normalSphere;


    int width=1024, height=768, bpp=3;
    unsigned char* rgb_image = stbi_load("../../marsbump.jpg", &width, &height, &bpp, 3);
    int total = width*height;
    for (int i = 0; i < total; ++i) {
        img[i*4+0] = rgb_image[i*3+0];
        img[i*4+1] = rgb_image[i*3+1];
        img[i*4+2] = rgb_image[i*3+2];
        img[i*4+3] = 1.0f * 255;
    }

    for (int i = 0; i < viewWidth; ++i) {
        for (int j = 0; j < viewHeight; ++j) {
            float x, y;

            if (i+1>=viewWidth){
                x = (float)(rgb_image[(j*viewHeight+i-1)*3]
                        -rgb_image[(j*viewHeight+i)*3]);
            }else if (i-1<0){
                x = (float)(rgb_image[(j*viewWidth+i)*3]
                        -rgb_image[(j*viewHeight+i+1)*3]);
            }
            else{
                x = (float)(rgb_image[(j*viewWidth+i-1)*3]
                        -rgb_image[(j*viewWidth+i+1)*3]);
            }

            if (j+1>=viewHeight){
                y = (float)(rgb_image[(j*viewWidth+i-viewWidth)*3]
                        -rgb_image[(j*viewWidth+i)*3]);
            }else if (j-1<0){
                y = (float)(rgb_image[(j*viewWidth+i)*3]-
                        rgb_image[(j*viewWidth+i+viewWidth)*3]);
            }
            else{
                y = (float)(rgb_image[(j*viewWidth+i-viewWidth)*3]
                        -rgb_image[(j*viewWidth+i+viewWidth)*3]);
            }


            img[(j*width+i)*4+0] = x+125;
            img[(j*width+i)*4+1] = y+125;
            img[(j*width+i)*4+2] = 0.0;
            img[(j*width+i)*4+3] = 0.0;

        }
    }

    //----------- Draw windows and update scene ------------
    if (window::Handle wHnd = window::create(viewWidth, viewHeight, "Hello CS248")) {
        if ((device = webgpu::create(wHnd))) {

            queue = wgpuDeviceGetQueue(device);
            swapchain = webgpu::createSwapChain(device);

            createPipelineAndBuffers();


            window::show(wHnd);
            window::loop(wHnd, redraw);


#ifndef __EMSCRIPTEN__
            wgpuBindGroupRelease(bindGroup[0]);
            wgpuBufferRelease(Buf);
            wgpuBufferRelease(indxBuf);
            wgpuBufferRelease(vertBuf);
            wgpuRenderPipelineRelease(pipeline);
            wgpuSwapChainRelease(swapchain);
            wgpuQueueRelease(queue);
            wgpuDeviceRelease(device);
#endif
        }
#ifndef __EMSCRIPTEN__
        window::destroy(wHnd);
#endif
    }


    return 0;
}
