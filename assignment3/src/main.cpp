#include "webgpu.h"
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream>
using namespace std;

#include "rasterization/Vec3.h"
#include "rasterization/Matrix4.h"
#include "rasterization/Light.h"
#include "rasterization/Color.h"
#include "rasterization/Camera.h"
#include "rasterization/Mesh.h"
#include <thread>         
#include <chrono>  


// Set up view plane
const int viewWidth = 1024, viewHeight = 768;
const double viewLeft = -2.0f, viewRight = 2.0f, viewBottom = -1.5f, viewTop = 1.5f;
const double aspect = (double)viewWidth / (double)viewHeight;
const double FoV = 45.f;

// #NOTE: If you compute the perspective projection matrix by using viewLeft, viewRight, viewBottom, viewTop,
// you have to find the value of nearPlane to make sure the FoV is 45.f and change the nearPlane in here
const double nearPlane = -1.0f, farPlane = -1000.0f;
const double INF = std::numeric_limits<double>::infinity();

//----------- Objects ----------------------------------------------
// #NOTE: You can find other material in this link: http://www.barradeau.com/nicoptere/dump/materials.html
Material emerald(Color(0.0215f, 0.1745f, 0.0215f), Color(0.07568f, 0.61424f, 0.07568f),
	Color(0.633f, 0.727811f, 0.633f), 76.8f);
// #NOTE: Depth buffer
double* depthBuffer = new double[viewWidth * viewHeight];

//判断是不是灯塔 
bool isbeacon=true;
// #NOTE: Normal buffer
vector<Vec3> normalBuffer;

// #NOTE: Color buffer
vector<Color> colorBuffer;

// #NOTE: You can draw 3 objects in one scene or one object in one scene
// Hint: Testing with sphere first
Mesh sphere;
Mesh beacon;
Mesh teapot;

// #TODO: Change these matrices for each object so that we can see these objects in camera space
Matrix4 sphereModel;
Matrix4 normalSphere;

Matrix4 beaconModel;
Matrix4 normalBeacon;

Matrix4 teapotModel;
Matrix4 normalTeapot;

// 把物体设置到相机观察的空间里，即相机的前面
// beaconModel=Matrix4::Scaling(125)*Matrix4::Translate(Vec3(-125,-125,-125));

// false
// 设置颜色方式
bool phong=false;
bool guad=true;
//----------- Camera -----------------------------------------------
Vec3 eye_postion(0.0f, 0.0f, 1.0f);
Vec3 gaze_direction(0.0f, 0.0f, -1.0f);
Vec3 view_up(0.0f, 1.0f, 0.0f);
Camera camera(eye_postion, gaze_direction, view_up);

// Arcball Matrix: will be composited with View matrix from Camera 
Matrix4 Arcball;
// Arcball.identity();
Matrix4 View = camera.calculateViewMatrix();


// #NOTE: Testing with two projective methods, should start with orthographic first
// You can define additional variables in here for checking which projective method is used.

// #TODO: Uncomment the line below if using orthorgraphic projection
Matrix4 Projection = camera.calculateOrthographicMatrix(viewLeft, viewRight, viewBottom, viewTop, nearPlane, farPlane);

// #TODO: Uncomment the line below if using perspective projection
// Matrix4 Projection = camera.calculatePerspectiveMatrix(FoV, aspect, nearPlane, farPlane);

// #NOTE: M_vp
Matrix4 Viewport;

//----------- Light ------------------------------------------------
Light light(Vec3(5.0f, 5.0f, 0.0f), Color(1.0f));
Matrix4 lightModel;

//----------- Result image ------------------------------------------
unsigned char* img = new unsigned char[viewWidth * viewHeight * 4];

//----------- Function for handling user interaction ----------------
void mouseClickHandler(int, int, int, int);
void keyPressHandler(int, int);

// 记录一下鼠标坐标变化
int record[2]={0,0};
//----------- Rasterization ------------------------------------------

// #TODO: Compute Viewport matrix for Viewport transformation M_vp
void calculateViewPortMatrix(int viewWidth, int viewHeight, Matrix4& Viewport)
{
	// 这个地方主要计算viewport的矩阵
	// Viewport = Matrix4::Scaling(Vec3(viewWidth/2,viewHeight/2,1))*Matrix4::Translate(Vec3((viewWidth-1)/2,(viewHeight-1)/2,0));
	Viewport.identity();
	Viewport.matrix[0][0]=viewWidth/2;
	Viewport.matrix[1][1]=viewHeight/2;
	Viewport.matrix[0][3]=(viewWidth-1)/2;
	Viewport.matrix[1][3]=(viewHeight-1)/2;
}

// #TODO: Define function to compute Arcball matrix for camera
void computeArcball(int x, int y){
	// calculateViewPortMatrix(viewWidth, viewHeight, Viewport);
	if(record[0]==x && record[1]==y){
		Matrix4 R;
		R.identity();
		Arcball=R*Arcball;
		return;
	}
	double z1,z2;
	double x1,x2,y1,y2;
	double theta;

	// 需要把点点坐标转化为NDC，即归一化坐标
	x1=double(record[0])/viewWidth;
	y1=double(record[1])/viewHeight;
	// 更新记录数组的鼠标坐标
	// record[0]=x;
	// record[1]=y;

	x2=double(x)/viewWidth;
	y2=double(y)/viewHeight;
	// Vec3 p1= Vec4::toVec3(objectModel*Vec4::(x1,y1,1,1));
	// Vec3 p2= Vec4::toVec3(*Vec4::(x2,y2,1,1));
	
	if(x1*x1+y1*y1>1.0){
		z1=0;
	}else{
		z1=sqrt(1-x1*x1-y1*y1);
	}
	// p1.z=z1;
	Vec3 p1=Vec3(x1,y1,z1);
	// Vec3 p1=Vec3(3,4,5);

	if(x2*x2+y2*y2>1){
		z2=0;
	}else{
		z2=sqrt(1-x2*x2-y2*y2);
	}
	// p2.z=z2;
	Vec3 p2=Vec3(x2,y2,z2);

	theta=acos(p1.dotProduct(p2)/(p1.vecLength()*p2.vecLength()));
	// cout<<theta<<endl;
	Vec3 u=p1.crossProduct(p2).normalize();

	Matrix4 R;
	R.identity();
	R.matrix[0][0]=u.x*u.x+(1-u.x*u.x)*cos(theta);
	R.matrix[0][1]=u.x*u.y*(1-cos(theta)-u.z*sin(theta));
	R.matrix[0][2]=u.x*u.z*(1-cos(theta)+u.y*sin(theta));
	R.matrix[0][3]=0;
	R.matrix[1][0]=u.y*u.x*(1-cos(theta)+u.z*sin(theta));
	R.matrix[1][1]=u.y*u.y+(1-u.y*u.y)*cos(theta);
	R.matrix[1][2]=u.y*u.z*(1-cos(theta)-u.x*sin(theta));
	R.matrix[1][3]=0;
	R.matrix[2][0]=u.z*u.x*(1-cos(theta)-u.y*sin(theta));
	R.matrix[2][1]=u.z*u.y*(1-cos(theta)+u.x*sin(theta));
	R.matrix[2][2]=u.z*u.z+(1-u.z*u.z)*cos(theta);
	R.matrix[2][3]=0;
	R.matrix[3][0]=0;
	R.matrix[3][1]=0;
	R.matrix[3][2]=0;
	R.matrix[3][3]=1;
	// cout<<Arcball<<endl;
	// return Arcball;
	Arcball=R*Arcball;
	// Arcball=R;
}

// #TODO: Compute color
// #NOTE: Ambient light intensity is the same with light intensity
Color shade(Vec3 pos, Vec3 normal, Camera cam, Light light, Material m) {
	Color c;
	Color diffuse = Color(0.0f, 0.0f, 0.0f);
	Color specular = Color(0.0f, 0.0f, 0.0f);

	Vec3 v = (cam.eye_position - pos).normalize();
	Vec3 l = (light.position - pos).normalize();
	Vec3 h = (v + l).normalize();	

	// ambient color
	Color ambient = light.intensity*m.Ka;

	diffuse = diffuse + (light.intensity * m.Kd * (max(0.0, (normal.dotProduct(l)))));
	specular = specular + (light.intensity * m.Ks * pow(max(0.0, (normal.dotProduct(h))), m.shininess));
		
	c = ambient + diffuse + specular;
	// 这是一个坑
	if(c.red>1){
		c.red=1;
	}
	if(c.blue>1){
		c.blue=1;
	}
	if(c.green>1){
		c.green=1;
	}

	return c;
}

// #TODO: Implement rasterization 
// rasterization(beacon, beaconModel, normalBeacon)
int c=0;
void rasterization(Mesh object, Matrix4 objectModel, Matrix4 normalMatrix) {
	// cout<<"Arc: "<<Arcball<<endl;
	// cout<<object.triangles.size()<<endl;
	// 每次染色前，初始化整个屏幕为黑色
	for(int i=0;i<viewWidth;i++){
		for(int j=0; j<viewHeight;j++){
			int idx = ((j * viewWidth) + i) * 4;
			img[idx] = floor(0.0);
			img[idx + 1] = floor(0.0);
			img[idx + 2] = floor(0.0);
			img[idx + 3] = 0;
		}
	}	
	// cout<<object.triangles.size()<<endl;
	for(int i=0; i<object.triangles.size();i++){
		// cout<<"check0"<<endl;
		// if(c>20){
		// 	break;
		// }
		Vec3 v0,v1,v2;
		Vec3 v0_norm, v1_norm, v2_norm;
		double alpha, beta, gamma;
		Matrix4 M;
		// Matrix4 Norm_M;
		calculateViewPortMatrix(viewWidth, viewHeight, Viewport);
		// 计算所需要的变化矩阵
		// objectModel=Arcball*objectModel;	
		M=Viewport*Projection*View;
		// 对于projection的变化，矩阵还要除以一个w的值

		// cout<<"check1"<<endl;
		// 还没转化到screen之前才有z的深度
		Vec3 v0_temp=Vec4::toVec3(View*objectModel*Vec4(object.triangles[i].v0,1));
		Vec3 v1_temp=Vec4::toVec3(View*objectModel*Vec4(object.triangles[i].v1,1));
		Vec3 v2_temp=Vec4::toVec3(View*objectModel*Vec4(object.triangles[i].v2,1));

		// 1. 把三角形的三个点映射到screen上
		v0=Vec4::toVec3(M*objectModel*Vec4(object.triangles[i].v0,1));
		v1=Vec4::toVec3(M*objectModel*Vec4(object.triangles[i].v1,1));
		v2=Vec4::toVec3(M*objectModel*Vec4(object.triangles[i].v2,1));
		// cout<<"check2"<<endl;
		// 计算映射之后的norm
		// cout<<object.normals.size()<<endl;
		v0_norm=Vec4::toVec3(normalMatrix*Vec4(object.triangles[i].v0Normal,0)).normalize();
		v1_norm=Vec4::toVec3(normalMatrix*Vec4(object.triangles[i].v1Normal,0)).normalize();
		v2_norm=Vec4::toVec3(normalMatrix*Vec4(object.triangles[i].v2Normal,0)).normalize();
		// cout<<"check3"<<endl;
		// beacon只能用face normal求
		Vec3 f_norm=Vec4::toVec3(normalMatrix*Vec4(object.triangles[i].faceNormal,0)).normalize();
		// cout<<"v0: "<<v0_norm<<endl;
		// cout<<v1_norm<<endl;
		// cout<<v2_norm<<endl;

		// 算出三角形三个顶点的颜色
		Color c0,c1,c2;
		// 设置最终的color
		// Color final_c;
		
		// 2. 找到三角形的bounding边，也就是bounding的像素点的坐标
		double x_min=object.triangles[i].min3(v0.x,v1.x,v2.x);
		int xi=floor(x_min);
		double x_max=object.triangles[i].max3(v0.x,v1.x,v2.x);
		int xj=ceil(x_max);
		double y_min=object.triangles[i].min3(v0.y,v1.y,v2.y);
		int yi=floor(y_min);
		double y_max=object.triangles[i].max3(v0.y,v1.y,v2.y);
		int yj=ceil(y_max);

		// cout<<"check4"<<endl;
		// cout<<xj<<endl;
		// cout<<yi<<endl;
		// cout<<yj<<endl;
		// 3.遍历三角形boungding中的每个像素点

		// line drawing wire frame
	

		// triangle rasterization
		for(int k=yi; k<(yj+1); k++){
			for(int j=xi; j<(xj+1); j++){
				// Color c0,c1,c2;<
				// cout<<"x: "<<j<<endl;
				// cout<<k<<endl;
				c=c+1;

				double k1=k+0.5;
				double j1=j+0.5;
				// cout<<k<<" "<<j<<endl;
				double x0=v0.x, y0=v0.y, x1=v1.x, y1=v1.y, x2=v2.x, y2=v2.y;

				double z;

				// alpha对应的点v0 beta对应的是点v1 gamma对应的点v2			
				double A = ((y1-y2)*x0+(x2-x1)*y0+x1*y2-x2*y1)*((y1-y2)*(-1)+(x2-x1)*(-1)+x1*y2-x2*y1);
				double B = ((y2-y0)*x1+(x0-x2)*y1+x2*y0-x0*y2)*((y2-y0)*(-1)+(x0-x2)*(-1)+x2*y0-x0*y2);
				double C = ((y0-y1)*x2+(x1-x0)*y2+x0*y1-x1*y0)*((y0-y1)*(-1)+(x1-x0)*(-1)+x0*y1-x1*y0);

				// 三角形中每个点都对应一个alpha beta gamma，无论在哪个坐标系下，同一个点对应的alpha beta gamma不变
				// 一旦知道该点对应的alpha beta gamma，就知道该点的坐标，用三个坐标分别乘以alpha beta gamma
				alpha = ((y1-y2)*j1+(x2-x1)*k1+x1*y2-x2*y1)/((y1-y2)*x0+(x2-x1)*y0+x1*y2-x2*y1);
				beta =  ((y2-y0)*j1+(x0-x2)*k1+x2*y0-x0*y2)/((y2-y0)*x1+(x0-x2)*y1+x2*y0-x0*y2);
				// gamma = ((y0-y1)*j1+(x1-x0)*k1+x0*y1-x1*y0)/((y0-y1)*x2+(x1-x0)*y2+x0*y1-x1*y0); 
				gamma = 1-beta-alpha;
				
				if(alpha>=0 && beta>=0 && gamma>=0){
					if( (alpha>0 || A>0)&&
					(beta>0 || B>0)&&
					(gamma>0 || C>0)){
						// phong color
						// 根据一个alpha beta gamma算出一个normal	
						if(phong){					
							Vec3 normal_v0=v0_norm*alpha;
							Vec3 normal_v1=v1_norm*beta;
							Vec3 normal_v2=v2_norm*gamma;
							// Vec3 normal_v0=object.triangles[i].v0Normal*alpha;
							// Vec3 normal_v1=object.triangles[i].v1Normal*beta;
							// Vec3 normal_v2=object.triangles[i].v2Normal*gamma;
							// 计算phong shading的法向量
							Vec3 p_norm=(normal_v2+normal_v1+normal_v0).normalize();

							Vec3 pixel_x = Vec4::toVec3(objectModel*Vec4(object.triangles[i].v0,1))*alpha+
							Vec4::toVec3(objectModel*Vec4(object.triangles[i].v1,1))*beta+Vec4::toVec3(objectModel*Vec4(object.triangles[i].v2,1))*gamma;

							// 更新Z buffer
							z=v0_temp.z*alpha+v1_temp.z*beta+v2_temp.z*gamma;
							if(z>depthBuffer[(k * viewWidth) + j]){
								// 当染色beacon时，用facenorm，其他两个染色用vertexnorm
								if(isbeacon){
									depthBuffer[(k * viewWidth) + j]= z;

									// object.triangles[i].faceNormal
									// Color final_c = shade(pixel_x, object.triangles[i].faceNormal.normalize(), camera, light, emerald);
									// cout<<"check1"<<endl;
									Color final_c = shade(pixel_x, f_norm, camera, light, emerald);
									
									int idx = ((k * viewWidth) + j) * 4;
									img[idx] = floor(final_c.red * 255);
									img[idx + 1] = floor(final_c.green * 255);
									img[idx + 2] = floor(final_c.blue * 255);
									img[idx + 3] = 255;
									// cout<<"check"<<endl;
								}
								else{
									// depthBuffer[(k * viewWidth) + j]= z;
									Color final_c = shade(pixel_x, p_norm.normalize(), camera,light, emerald);

									int idx = ((k * viewWidth) + j) * 4;
									img[idx] = floor(final_c.red * 255);
									img[idx + 1] = floor(final_c.green * 255);
									img[idx + 2] = floor(final_c.blue * 255);
									img[idx + 3] = 255;		
								}
							}
						}
						//guad
						if(guad){
							// 只有更新了Zbuffer之后才能染色
							z=v0_temp.z*alpha+v1_temp.z*beta+v2_temp.z*gamma;
							
							// cout<<depthBuffer[k*viewWidth+j]<<endl;
							if(z>depthBuffer[(k * viewWidth) + j]){
								// cout<<z<<endl;
								depthBuffer[(k * viewWidth) + j]= z;
								c0 = shade(Vec3(j1,k1,1), v0_norm.normalize(), camera,light, emerald);
								c1 = shade(Vec3(j1,k1,1), v1_norm.normalize(), camera,light, emerald);
								c2 = shade(Vec3(j1,k1,1), v2_norm.normalize(), camera,light, emerald);
								// c0 = shade(Vec4::toVec3(objectModel*Vec4(object.triangles[i].v0,1)), object.triangles[i].v0Normal, camera,light, emerald);
								// c1 = shade(Vec4::toVec3(objectModel*Vec4(object.triangles[i].v1,1)), object.triangles[i].v1Normal, camera,light, emerald);
								// c2 = shade(Vec4::toVec3(objectModel*Vec4(object.triangles[i].v2,1)), object.triangles[i].v2Normal, camera,light, emerald);
								Color final_c = c0*alpha+c1*beta+c2*gamma;
								
								int idx = ((k * viewWidth) + j) * 4;
								img[idx] = floor(final_c.red * 255);
								img[idx + 1] = floor(final_c.green * 255);
								img[idx + 2] = floor(final_c.blue * 255);
								img[idx + 3] = 255;
						
							}
						}
					}		

				}

			}
		}
	
		// break;s

	}

}
void drawline(Vec3 v1, Vec3 v2){
	// cout<<"v1"<<v1<<endl;
	// cout<<v2<<endl;
	double x1,x2,y1,y2;
	double step;
	Color final_c=Color (1.0,1.0,0);
	
	x1=v1.x;
	x2=v2.x;
	y1=v1.y;
	y2=v2.y;
	double m;
	if(x1!=x2){
		m=(y2-y1)/(x2-x1);
	}
	if(x1==x2){
		step=0.0;
	}
	// cout<<m<<endl;
	if(m==0){
		step=0.0;
	}
	if(m>0){
		step=1.0;
		// x1=floor(x1);
		// y1=floor(y1);
		// x2=ceil(x2);
		// y2=ceil(y2);
	}
	if(m<0){
		step=-1.0;
		// x1=ceil(x1);
		// y1=ceil(y1);
		// x2=floor(x2);
		// y2=floor(y2);
	}

	int y=y1;
	if(abs(m)<=1){
		for(int i=x1;i<x2;i++){
			if(m>0 && y>y2){
				break;
			}
			if(m<0 && y<y2){
				break;
			}
			int idx = ((y * viewWidth) + i) * 4;
			img[idx] = floor(final_c.red * 255);
			img[idx + 1] = floor(final_c.green * 255);
			img[idx + 2] = floor(final_c.blue * 255);
			img[idx + 3] = 0;	

			// f(x+1,y+0.5)<0
			double temp=(y1-y2)*(i+1)+(x2-x1)*(y+step/2)+x1*y2-x2*y1;
			// cout<<temp<<endl;
			if(temp<0){
				y=y+step;
			}
		}
	}

	int x=x1;
	if(abs(m)>1){
		if(y1<=y2){
		for(int i=y1;i<y2;i++){
			// if(x==x2){
			// 	break;
			// }
			if(x>x2){
				break;
			}
		
			int idx = ((i * viewWidth) + x) * 4;
			img[idx] = floor(final_c.red * 255);
			img[idx + 1] = floor(final_c.green * 255);
			img[idx + 2] = floor(final_c.blue * 255);
			img[idx + 3] = 0;	

			// f(x+1,y+0.5)<0
			double temp=(y1-y2)*(x+step/2)+(x2-x1)*(i+1)+x1*y2-x2*y1;
			// cout<<temp<<endl;
			if(temp>0){
			x=x+step;
			}
		}}
		int x=x2;
		if(y1>y2){
		for(int i=y2;i<y1;i++){
			// if(m>0 && x>y2){
			// 	break;
			// }
			// if(m<0 && y<y2){
			// 	break;
			// }

			int idx = ((i * viewWidth) + x) * 4;
			img[idx] = floor(final_c.red * 255);
			img[idx + 1] = floor(final_c.green * 255);
			img[idx + 2] = floor(final_c.blue * 255);
			img[idx + 3] = 0;	

			// f(x+1,y+0.5)<0
			double temp=(y1-y2)*(x+step/2)+(x2-x1)*(i+1)+x1*y2-x2*y1;
			// cout<<temp<<endl;
			if(temp>0){
			x=x+step;
			}
		}	
		}
	}

}
// #TODO: Define function for wireframe model rendering with a hidden surface removal
void wireframe(Mesh object, Matrix4 objectModel){
	
	// 每次画之前刷新一下屏幕
	for(int i=0;i<viewWidth;i++){
	for(int j=0; j<viewHeight;j++){
		int idx = ((j * viewWidth) + i) * 4;
		img[idx] = floor(0.0);
		img[idx + 1] = floor(0.0);
		img[idx + 2] = floor(0.0);
		img[idx + 3] = 0;
		}
	}

	for(int i=0; i<object.triangles.size();i++){
		// if(c>20){
		// 	break;
		// }
		Vec3 v0,v1,v2;
		Vec3 v0_norm, v1_norm, v2_norm;
		Matrix4 M;
		// Matrix4 Norm_M;
		calculateViewPortMatrix(viewWidth, viewHeight, Viewport);
		// 计算所需要的变化矩阵
		// objectModel=Arcball*objectModel;	
		M=Viewport*Projection*View;	

		v0=Vec4::toVec3(M*objectModel*Vec4(object.triangles[i].v0,1));
		v1=Vec4::toVec3(M*objectModel*Vec4(object.triangles[i].v1,1));
		v2=Vec4::toVec3(M*objectModel*Vec4(object.triangles[i].v2,1));	
		
		// cout<<"1: "<<v0<<endl;
		// cout<<v1<<endl;
		// cout<<v2<<endl;	
		
		double x0,y0,x1,y1,x2,y2;
		// 	加上0.5表示像素点的中心
		x0=v0.x;
		x1=v1.x;
		x2=v2.x;
		y0=v0.y;
		y1=v1.y;
		y2=v2.y;

		Color final_c=Color (1.0,1.0,0);

		if(x0<=x1){
			drawline(v0,v1);
		}else if(x0>x1){
			drawline(v1,v0);
		}

		if(x1<=x2){
			drawline(v1,v2);
		}else if(x1>x2){
			drawline(v2,v1);
		}
		
		if(x2<=x0){
			drawline(v2,v0);
		}else if(x2>x0){
			drawline(v0,v2);
		}

	}

}
//----------- WEBGPU variables ---------------------------------------

WGPUDevice device;
WGPUQueue queue;
WGPUSwapChain swapchain;

WGPURenderPipeline pipeline;
WGPUBuffer vertBuf; // vertex buffer with position
WGPUBuffer indxBuf; // index buffer
WGPUBindGroup bindGroup;

WGPUTexture tex; // Texture
WGPUSampler samplerTex;
WGPUTextureView texView;
WGPUExtent3D texSize = {};
WGPUTextureDescriptor texDesc = {};
WGPUTextureDataLayout texDataLayout = {};
WGPUImageCopyTexture texCopy = {};

/*
* Shaders
*/
static char const rectangle_vert_wgsl[] = R"(
	[[block]]
	struct VertexIn {
		[[location(0)]] aPos : vec2<f32>;
		[[location(1)]] aTex : vec2<f32>;
	};
	struct VertexOut {
		[[location(0)]] vTex : vec2<f32>;
		[[builtin(position)]] Position : vec4<f32>;
	};
	[[stage(vertex)]]
	fn main(input : VertexIn) -> VertexOut {
		var output : VertexOut;
		output.Position = vec4<f32>(vec3<f32>(input.aPos, 1.0), 1.0);
		output.vTex = input.aTex;
		return output;
	}
)";

static char const rectangle_frag_wgsl[] = R"(
	[[group(0), binding(0)]]
	var tex: texture_2d<f32>;
	[[group(0), binding(1)]]
	var sam: sampler;

	[[stage(fragment)]]
	fn main([[location(0)]] vTex : vec2<f32>) -> [[location(0)]] vec4<f32> {
		return textureSample(tex, sam, vTex);
	}
)";

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
	desc.size  = size;
	WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
	wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
	return buffer;
}

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
}

/**
 * Bare minimum pipeline to draw a triangle using the above shaders.
 */
static void createPipelineAndBuffers(unsigned char* data, unsigned int w, unsigned int h) {
	// compile shaders
	// NOTE: these are now the WGSL shaders (tested with Dawn and Chrome Canary)
	WGPUShaderModule vertMod = createShader(rectangle_vert_wgsl);
	WGPUShaderModule fragMod = createShader(rectangle_frag_wgsl);

	//===================================================================

	tex = createTexture(data, w, h);

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
	bglTexEntry.visibility = WGPUShaderStage_Fragment;
	bglTexEntry.texture = texLayout;

	WGPUBindGroupLayoutEntry bglSamplerEntry = {};
	bglSamplerEntry.binding = 1;
	bglSamplerEntry.visibility = WGPUShaderStage_Fragment;
	bglSamplerEntry.sampler = samplerLayout;

	WGPUBindGroupLayoutEntry* allBgLayoutEntries = new WGPUBindGroupLayoutEntry[2];
	allBgLayoutEntries[0] = bglTexEntry;
	allBgLayoutEntries[1] = bglSamplerEntry;

	//=======================================================================

	WGPUBindGroupLayoutDescriptor bglDesc = {};
	bglDesc.entryCount = 2;
	bglDesc.entries = allBgLayoutEntries;
	WGPUBindGroupLayout bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

	// pipeline layout (used by the render pipeline, released after its creation)
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	layoutDesc.bindGroupLayoutCount = 1;
	layoutDesc.bindGroupLayouts = &bindGroupLayout;
	WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);

	// describe buffer layouts
	WGPUVertexAttribute vertAttrs[2] = {};
	vertAttrs[0].format = WGPUVertexFormat_Float32x2;
	vertAttrs[0].offset = 0;
	vertAttrs[0].shaderLocation = 0;
	vertAttrs[1].format = WGPUVertexFormat_Float32x2;
	vertAttrs[1].offset = 2 * sizeof(float);
	vertAttrs[1].shaderLocation = 1;
	WGPUVertexBufferLayout vertexBufferLayout = {};
	vertexBufferLayout.arrayStride = 4 * sizeof(float);
	vertexBufferLayout.attributeCount = 2;
	vertexBufferLayout.attributes = vertAttrs;

	// Fragment state
	WGPUBlendState blend = {};
	blend.color.operation = WGPUBlendOperation_Add;
	blend.color.srcFactor = WGPUBlendFactor_One;
	blend.color.dstFactor = WGPUBlendFactor_One;
	blend.alpha.operation = WGPUBlendOperation_Add;
	blend.alpha.srcFactor = WGPUBlendFactor_One;
	blend.alpha.dstFactor = WGPUBlendFactor_One;

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
	desc.fragment = &fragment;

	// Other state
	desc.layout = pipelineLayout;
	desc.depthStencil = nullptr;

	desc.vertex.module = vertMod;
	desc.vertex.entryPoint = "main";
	desc.vertex.bufferCount = 1;
	desc.vertex.buffers = &vertexBufferLayout;

	desc.multisample.count = 1;
	desc.multisample.mask = 0xFFFFFFFF;
	desc.multisample.alphaToCoverageEnabled = false;

	desc.primitive.frontFace = WGPUFrontFace_CCW;
	desc.primitive.cullMode = WGPUCullMode_None;
	desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	desc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;

#ifdef __EMSCRIPTEN__
	pipeline = wgpuDeviceCreateRenderPipeline (device, &desc);
#else
	pipeline = wgpuDeviceCreateRenderPipeline (device, &desc);
#endif

	// partial clean-up (just move to the end, no?)
	wgpuPipelineLayoutRelease(pipelineLayout);

	wgpuShaderModuleRelease(fragMod);
	wgpuShaderModuleRelease(vertMod);
	
	// create the buffers (position[2], tex_coords[2])
	float const vertData[] = {
		-1.0f, -1.0f, 0.0f, 0.0f, 
		 1.0f, -1.0f, 1.0f, 0.0f, 
		-1.0f,  1.0f, 0.0f, 1.0f, 
		 1.0f,  1.0f, 1.0f, 1.0f, 
	};
	
	// indices buffer
	uint16_t const indxData[] = {
		0, 1, 2, 1, 3, 2, 0, 0 //Two last zero: padding
	};

	vertBuf = createBuffer(vertData, sizeof(vertData), WGPUBufferUsage_Vertex);
	indxBuf = createBuffer(indxData, sizeof(indxData), WGPUBufferUsage_Index);

	WGPUBindGroupEntry bgTexEntry = {};
	bgTexEntry.binding = 0;
	bgTexEntry.textureView = texView;

	WGPUBindGroupEntry bgSamplerEntry = {};
	bgSamplerEntry.binding = 1;
	bgSamplerEntry.sampler = samplerTex;


	WGPUBindGroupEntry* allBgEntries = new WGPUBindGroupEntry[2];
	allBgEntries[0] = bgTexEntry;
	allBgEntries[1] = bgSamplerEntry;

	WGPUBindGroupDescriptor bgDesc = {};
	bgDesc.layout = bindGroupLayout;
	bgDesc.entryCount = 2;
	bgDesc.entries = allBgEntries;


	bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

	// last bit of clean-up
	wgpuBindGroupLayoutRelease(bindGroupLayout);
}

/**
 * Draws using the above pipeline and buffers.
 */
// double* depthBuffer = new double[viewWidth * viewHeight];
static bool redraw() {
	// 初始化z_buffer
	for(int i=0;i<viewWidth;i++)
		for(int j=0; j<viewHeight; j++){
			depthBuffer[j*viewWidth+i]=farPlane;
		}

	
	WGPUTextureView backBufView = wgpuSwapChainGetCurrentTextureView(swapchain);			// create textureView

	WGPURenderPassColorAttachment colorDesc = {};
	colorDesc.view    = backBufView;
	colorDesc.loadOp  = WGPULoadOp_Clear;
	colorDesc.storeOp = WGPUStoreOp_Store;
	colorDesc.clearColor.r = 0.0f;
	colorDesc.clearColor.g = 0.0f;
	colorDesc.clearColor.b = 0.0f;
	colorDesc.clearColor.a = 1.0f;

	WGPURenderPassDescriptor renderPass = {};
	renderPass.colorAttachmentCount = 1;
	renderPass.colorAttachments = &colorDesc;

	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);			// create encoder
	WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPass);	// create pass

	// #NOTE: You can add updates for the scene in here (also wireframe model rendering)

	// 把物体设置到相机观察的空间里，即相机的前面
	// norm的矩阵就是N=(M^-1)^T
	beaconModel=Matrix4::Translate(Vec3(0,0,-3))*Arcball*Matrix4::Scaling(1.0/125)*Matrix4::Translate(Vec3(-125,-125,-125));
	normalBeacon=(Matrix4::TranslateInv(Vec3(-125,-125,-125))*Matrix4::ScalingInv(1.0/125)*Arcball.transpose()*Matrix4::TranslateInv(Vec3(0,0,-3))).transpose();
	// normalBeacon=(Matrix4::TranslateInv(Vec3(-125,-125,-125))*Matrix4::ScalingInv(1.0/125)).transpose();
	
	// 把求往Z轴负方向移动，因为球心在原点，不能显示完整的球
	// 在物体平移到照相机前之前进行Arcball的操作，这样的话Arcball操作的时候，物体才不会变形
	sphereModel=Matrix4::Translate(Vec3(0,0,-5))*Arcball;
	normalSphere=(Arcball.transpose()*Matrix4::TranslateInv(Vec3(0,0,-5))).transpose();

	teapotModel=Matrix4::Translate(Vec3(0,0,-3))*Matrix4::Scaling(Vec3(2.0/7))*Arcball*Matrix4::Translate(Vec3(-0.217,-1.575,0));
	normalTeapot=(Matrix4::TranslateInv(Vec3(-0.217,-1.575,0))*Arcball.transpose()*Matrix4::ScalingInv(Vec3(2.0/7))*Matrix4::TranslateInv(Vec3(0,0,-3))).transpose();

	// rasterization(beacon, beaconModel, normalBeacon);
	// wireframe(beacon, beaconModel);
	rasterization(sphere, sphereModel, normalSphere);
	// wireframe(sphere,sphereModel);
	// rasterization(teapot, teapotModel, normalTeapot);
	// wireframe(teapot,teapotModel);


	// std::this_thread::sleep_for(std::chrono::seconds(1));
	wgpuQueueWriteTexture(queue, &texCopy, img, viewWidth * viewHeight * 4, &texDataLayout, &texSize);	
	

	// draw the triangle (comment these five lines to simply clear the screen)
	wgpuRenderPassEncoderSetPipeline(pass, pipeline);
	wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup, 0, 0);
	wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertBuf, 0, 0);
	wgpuRenderPassEncoderSetIndexBuffer(pass, indxBuf, WGPUIndexFormat_Uint16, 0, 0);
	wgpuRenderPassEncoderDrawIndexed(pass, 8, 1, 0, 0, 0);

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

// #TODO: Using these two functions for tasks in the assignment
/**
 * Mouse handling function.
 */
void mouseClickHandler(int button, int action, int x, int y)
{
	// int x1=0,y1=0,x2=0,y2=0;
	// int temp[4];
	
	printf("button:%d action:%d x:%d y:%d\n", button, action, x, y);
	if(button==0){
		if(action==1){
			// Arcball=computeArcball(x,y,Arcball);
			// computeArcball(x,y,Arcball);
			record[0]=x;
			record[1]=y;
		}
		if(action==0){
			// Arcball=computeArcball(x,y,Arcball);
			computeArcball(x,y);
			// cout<<View<<endl;
			// cout<<Viewport<<endl;
			// cout<<Projection<<endl;
			// cout<<teapotModel<<endl;
			// cout<<normalTeapot<<endl;

		}
		// if(x1!=x2 || y1!=y2){
		// Arcball=computeArcball(x,y);
		// }
		// beaconModel=Arcball*beaconModel;
	}
	// printf("button:%d action:%d x:%d y:%d\n", button, action, x, y);
}

/**
 * Keyboard handling function.
 */
void keyPressHandler(int button, int action)
{
	printf("key:%d action:%d\n", button, action);
}


extern "C" int __main__(int /*argc*/, char* /*argv*/[]) {

	// // //##############teapot##############
	teapot.loadOBJ("../data/teapot.obj");
	teapot.generateFaceNormals(
		teapot.points,
		teapot.pointlink,
		teapot.faceNormals);
	
	teapot.generateVertexNormals(
        teapot.points,
        teapot.normals,
        teapot.faces);

	// ##############sphere###############
	sphere.loadOBJ("../data/sphere.obj");
	sphere.generateFaceNormals(
		sphere.points,
		sphere.pointlink,
		sphere.faceNormals);
	sphere.generateVertexNormals(
        sphere.points,
        sphere.normals,
        sphere.faces);

	
	// ################beacon##############
	beacon.loadOBJ("../data/beacon.obj");
	beacon.generateFaceNormals(
        beacon.points,
        beacon.pointlink,
        // const vector<int>& faces,
        beacon.faceNormals);

	beacon.generateVertexNormals(
        beacon.points,
        beacon.normals,
        beacon.faces);


	calculateViewPortMatrix(viewWidth, viewHeight, Viewport);
	
	//----------- Draw windows and update scene ------------
	if (window::Handle wHnd = window::create(viewWidth, viewHeight, "Hello CS248")) {
		if ((device = webgpu::create(wHnd))) {

			queue = wgpuDeviceGetQueue(device);
			swapchain = webgpu::createSwapChain(device);
			createPipelineAndBuffers(img, viewWidth, viewHeight);

			// bind the user interaction
			window::mouseClicked(mouseClickHandler);
			window::keyPressed(keyPressHandler);

			window::show(wHnd);
			window::loop(wHnd, redraw);


#ifndef __EMSCRIPTEN__
			wgpuBindGroupRelease(bindGroup);
			wgpuBufferRelease(indxBuf);
			wgpuBufferRelease(vertBuf);
			wgpuSamplerRelease(samplerTex);
			wgpuTextureViewRelease(texView);
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
