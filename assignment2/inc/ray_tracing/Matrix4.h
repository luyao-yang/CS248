#pragma once
#include "Vec3.h"
#include <iostream>
#include <cmath>
#include<math.h>

// Vec4 for homogeneous coordinate
class Vec4 {

public:
	double x;
	double y;
	double z;
	double w;
	Vec4() :x(0), y(0), z(0), w(0) {}
	Vec4(double _x, double _y, double _z, double _w) :x(_x), y(_y), z(_z), w(_w) {}
	Vec4(Vec3 a, double _w) :x(a.x), y(a.y), z(a.z), w(_w) {}
	static Vec3 toVec3(Vec4 a)
	{
		return Vec3(a.x, a.y, a.z);
	}
};

// Matrix4 for transformation
class Matrix4 {
public:
	double matrix[4][4];

	Matrix4() {
		identity();
	}
	void identity()
	{
		// #TODO: Implement function to create 4x4 identity matrix]
		// cout<<"inden";
		int i,j;
		for(i=0; i<4; i++)
			for(j=0; j<4; j++){
				matrix[i][j]=0;
				if(i==j)
					matrix[i][j]=1;
				// matrix[i][j]=0;
			}
	}

	Matrix4 operator *(Matrix4 a)
	{
		// #TODO: Implement function for matrix multiplication
		Matrix4 result;
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++){
				result.matrix[i][j]= 0;
				for(int k=0;k<4;k++){
					result.matrix[i][j] += matrix[i][k]*a.matrix[k][j];
				}
			}
		}
	
		return result;
	}
	Vec4 operator *(Vec4 a)
	{
		// #TODO: Implement function for matrix - vector multiplication
		double x=0.0,y=0.0,z=0.0,w=0.0;
		
		x=matrix[0][0]*a.x+matrix[0][1]*a.y+matrix[0][2]*a.z+matrix[0][3]*a.w;
		y=matrix[1][0]*a.x+matrix[1][1]*a.y+matrix[1][2]*a.z+matrix[1][3]*a.w;
		z=matrix[2][0]*a.x+matrix[2][1]*a.y+matrix[2][2]*a.z+matrix[2][3]*a.w;
		w=matrix[3][0]*a.x+matrix[3][1]*a.y+matrix[3][2]*a.z+matrix[3][3]*a.w;
		
		return Vec4(x,y,z,w);
	}
	Matrix4 transpose()
	{
		// #TODO: Implement function to calculate transpose of a matrix
		Matrix4 result;

		for(int i=0;i<4;i++)
			for(int j=0; j<4;j++){
				result.matrix[j][i]=matrix[i][j];
			}
		return result;
	}
	static Matrix4 Translate(Vec3 t)
	{
		// #TODO: Implement function to calculate translation matrix
		Matrix4 result;
		result.identity();

		result.matrix[0][3]=t.x;
		result.matrix[1][3]=t.y;
		result.matrix[2][3]=t.z;

		return result;

	}
	static Matrix4 TranslateInv(Vec3 t)
	{
		// #TODO: Implement function to calculate inverse of translation matrix
		Matrix4 result;
		result.identity();

		result.matrix[0][3]=-t.x;
		result.matrix[1][3]=-t.y;
		result.matrix[2][3]=-t.z;

		return result;

	}
	static Matrix4 Scaling(Vec3 t)
	{
		// #TODO: Implement function to calculate scaling matrix
		Matrix4 result;
		result.identity();

		result.matrix[0][0]=t.x;
		result.matrix[1][1]=t.y;
		result.matrix[2][2]=t.z;

		return result;
	}

	static Matrix4 ScalingInv(Vec3 t)
	{
		// #TODO: Implement function to calculate inverse of scaling matrix
		Matrix4 result;
		result.identity();

		result.matrix[0][0]=1/t.x;
		result.matrix[1][1]=1/t.y;
		result.matrix[2][2]=1/t.z;

		return result;

	}
	static Matrix4 RotationX(double theta)
	{
		// #TODO: Implement function to calculate rotation matrix - X axis
		Matrix4 result;
		result.identity();

		double c = cos(theta*3.1415926/180.0);
		double s = sin(theta*3.1415926/180.0);

		result.matrix[1][1]=c;
		result.matrix[1][2]=s;
		result.matrix[2][1]=-s;
		result.matrix[2][2]=c;

		return result;	

	}
	static Matrix4 RotationXInv(double theta)
	{
		// #TODO: Implement function to calculate inverse of rotation matrix - X axis
		Matrix4 result;
		result.identity();

		result=result.RotationX(theta);
		result=result.transpose();

		return result;

	}
	static Matrix4 RotationY(double theta)
	{
		// #TODO: Implement function to calculate rotation matrix - Y axis
		Matrix4 result;
		result.identity();

		double c = cos(theta*3.1415926/180.0);
		double s = sin(theta*3.1415926/180.0);

		result.matrix[0][0]=c;
		result.matrix[0][2]=-s;
		result.matrix[2][0]=s;
		result.matrix[2][2]=c;

		return result;	

	}
	static Matrix4 RotationYInv(double theta)
	{
		// #TODO: Implement function to calculate inverse of rotation matrix - Y axis
		Matrix4 result;
		result.identity();

		result=result.RotationY(theta);
		result=result.transpose();

		return result;	
	}
	static Matrix4 RotationZ(double theta)
	{
		// #TODO: Implement function to calculate rotation matrix - Z axis
		Matrix4 result;
		result.identity();

		double c = cos(theta*3.1415926/180.0);
		double s = sin(theta*3.1415926/180.0);

		result.matrix[0][0]=c;
		result.matrix[0][1]=s;
		result.matrix[1][0]=-s;
		result.matrix[1][1]=c;

		return result;

	}
	static Matrix4 RotationZInv(double theta)
	{
		// #TODO: Implement function to calculate inverse of rotation matrix - Z axis
		Matrix4 result;
		result.identity();

		result=result.RotationZ(theta);
		result=result.transpose();

		return result;
	}

	static Matrix4 Rotation(Vec3 theta)
	{
		// #TODO: Implement function to calculate rotation matrix for three axes
		// theta.x - rotate angle around X axis
		// theta.y - rotate angle around Y axis
		// theta.z - rotate angle around Z axis
		Matrix4 result;
		result.identity();

		Matrix4 m1=result.RotationX(theta.x);
		Matrix4 m2=result.RotationY(theta.y);
		Matrix4 m3=result.RotationZ(theta.z);

		result=m1*m2*m3;
		return result;

	}
	static Matrix4 RotationInv(Vec3 theta)
	{
		// #TODO: Implement function to calculate inverse of rotation matrix for three axes
		// theta.x - rotate angle around X axis
		// theta.y - rotate angle around Y axis
		// theta.z - rotate angle around Z axis
		Matrix4 result;
		result.identity();

		Matrix4 m1=result.RotationXInv(theta.x);
		Matrix4 m2=result.RotationYInv(theta.y);
		Matrix4 m3=result.RotationZInv(theta.z);

		result=m3*m2*m1;
		return result;

	}
};

std::ostream& operator<< (std::ostream& os, const Matrix4& m)
{
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			os << m.matrix[i][j] << "\t";
		}
		os << "\n";
	}
	return os;
}