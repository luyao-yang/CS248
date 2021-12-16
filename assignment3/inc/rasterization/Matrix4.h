#pragma once
#include "Vec3.h"
#include <iostream>
#include <cmath>

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
class Matrix4 {
public:
	double matrix[4][4];

	Matrix4() {
		identity();
	}
	Matrix4(double a) {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j)
			{
				matrix[i][j] = a;
			}
		}
	}
	void identity()
	{
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				if (i == j)
					matrix[i][j] = 1;
				else
					matrix[i][j] = 0;
			}
		}
	}
	Matrix4 operator *(Matrix4 a)
	{
		Matrix4 result;
		
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j)
			{
				double value = 0;
				for (int k = 0; k < 4; ++k)
				{
					value += matrix[i][k] * a.matrix[k][j];

				}
				result.matrix[i][j] = value;
			}
				
		}
		return result;
	}
	Vec4 operator *(Vec4 a)
	{
		Vec4 result;
		result.x = matrix[0][0] * a.x + matrix[0][1] * a.y + matrix[0][2] * a.z + matrix[0][3] * a.w;
		result.y = matrix[1][0] * a.x + matrix[1][1] * a.y + matrix[1][2] * a.z + matrix[1][3] * a.w;
		result.z = matrix[2][0] * a.x + matrix[2][1] * a.y + matrix[2][2] * a.z + matrix[2][3] * a.w;
		result.w = matrix[3][0] * a.x + matrix[3][1] * a.y + matrix[3][2] * a.z + matrix[3][3] * a.w;
		return result;
	}
	Matrix4 transpose()
	{
		Matrix4 result;
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				result.matrix[i][j] = matrix[j][i];
			}
		}

		return result;
	}
	static Matrix4 Translate(Vec3 t)
	{
		Matrix4 result;
		result.matrix[0][3] = t.x;
		result.matrix[1][3] = t.y;
		result.matrix[2][3] = t.z;
		return result;

	}
	static Matrix4 TranslateInv(Vec3 t)
	{
		Matrix4 result;
		result.matrix[0][3] = t.x * -1;
		result.matrix[1][3] = t.y * -1;
		result.matrix[2][3] = t.z * -1;
		return result;

	}
	static Matrix4 Scaling(Vec3 t)
	{
		Matrix4 result;
		if (t.x > 0)
			result.matrix[0][0] = t.x;
		if (t.y > 0)
			result.matrix[1][1] = t.y;
		if (t.z > 0)
			result.matrix[2][2] = t.z;

		return result;

	}
	static Matrix4 ScalingInv(Vec3 t)
	{
		Matrix4 result;
		if (t.x > 0)
			result.matrix[0][0] = 1.0 / t.x;
		if (t.y > 0)
			result.matrix[1][1] = 1.0 / t.y;
		if (t.z > 0)
			result.matrix[2][2] = 1.0 / t.z;
		return result;

	}
	static Matrix4 Rotation(Vec3 theta)
	{
		return (RotationX(theta.x) * RotationY(theta.y) * RotationZ(theta.z));

	}
	static Matrix4 RotationInv(Vec3 theta)
	{
		return (RotationZInv(theta.z) * RotationYInv(theta.y) * RotationXInv(theta.x));

	}
	static Matrix4 RotationX(double theta)
	{
		Matrix4 result;
		result.matrix[1][1] = cos(theta);
		result.matrix[1][2] = -1 * sin(theta);
		result.matrix[2][1] = sin(theta);
		result.matrix[2][2] = cos(theta);

		return result;

	}
	static Matrix4 RotationXInv(double theta)
	{
		Matrix4 result;
		result.matrix[1][1] = cos(theta);
		result.matrix[1][2] = -1 * sin(theta);
		result.matrix[2][1] = sin(theta);
		result.matrix[2][2] = cos(theta);

		return result.transpose();
	}
	static Matrix4 RotationY(double theta)
	{
		Matrix4 result;
		result.matrix[0][0] = cos(theta);
		result.matrix[2][0] = -1 * sin(theta);
		result.matrix[0][2] = sin(theta);
		result.matrix[2][2] = cos(theta);

		return result;

	}
	static Matrix4 RotationYInv(double theta)
	{
		Matrix4 result;
		result.matrix[0][0] = cos(theta);
		result.matrix[2][0] = -1 * sin(theta);
		result.matrix[0][2] = sin(theta);
		result.matrix[2][2] = cos(theta);

		return result.transpose();
	}
	static Matrix4 RotationZ(double theta)
	{
		Matrix4 result;
		result.matrix[0][0] = cos(theta);
		result.matrix[0][1] = -1 * sin(theta);
		result.matrix[1][0] = sin(theta);
		result.matrix[1][1] = cos(theta);

		return result;

	}
	static Matrix4 RotationZInv(double theta)
	{
		Matrix4 result;
		result.matrix[0][0] = cos(theta);
		result.matrix[0][1] = -1 * sin(theta);
		result.matrix[1][0] = sin(theta);
		result.matrix[1][1] = cos(theta);

		return result.transpose();
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