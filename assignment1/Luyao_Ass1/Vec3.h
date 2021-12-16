#pragma once
#include "Color.h"
#include <iostream>
#include <cmath>

class Vec3 {	
public:
	double x;
	double y;
	double z;
	Vec3(): x(0), y(0), z(0){}
	Vec3(double v) : x(v), y(v), z(v) {}
	Vec3(double _x, double _y, double _z): x(_x), y(_y), z(_z) {}
	Vec3(const Vec3 &v) : x(v.x), y(v.y), z(v.z) {}
	
	double vecLength(){
		// #TODO: Implement function to calculate length of vector
		double length;
		// cout<<length;
		length = sqrt(pow(x,2)+pow(y,2)+pow(z,2));
		return sqrt(pow(x,2)+pow(y,2)+pow(z,2));
	}

	double dotProduct(Vec3 b){
		// #TODO: Implement function to calculate dot product of two vectors: a.b
		return x*b.x + y*b.y + z*b.z;
	}

	Vec3 crossProduct(Vec3 b){
		// #TODO: Implement function to calculate cross product of two vectors: axb
		// x = y1z2-y2z1
		double xi = y*b.z - b.y*z;
		// y=z1x2-z2x1  
		double yi = z*b.x - b.z*x;
		// z=x1y2-x2y1
		double zi = x*b.y - b.x*y;
		Vec3 c(xi,yi,zi);

		return c;
	}

	Vec3 operator*(double b){
		// #TODO: Implement function to calculate vector a multiple with scalar b
		return Vec3(b*x, b*y, b*z);
	}
	Vec3 operator/(double b) {
		// #TODO: Implement function to calculate vector a divide by scalar b
		return Vec3(x/b, y/b, z/b);
	}
	
	Vec3 operator+(double b){
		// #TODO: Implement function to calculate element wise addition of vector a and scalar b
		return Vec3(x+b, y+b, z+b);
	}
	Vec3 operator-(double a){
		// #TODO: Implement function to calculate element wise subtraction of vector a and scalar b
		return Vec3(x-a, y-a, z-a);
	}
	Vec3 operator+(Vec3 b) {
		// #TODO: Implement function to calculate sum of two vectors: a + b
		return Vec3(x+b.x, y+b.y, z+b.z);
	}
	Vec3 operator-(Vec3 b){
		// #TODO: Implement function to calculate sum of two vectors: a - b
		return Vec3(x-b.x, y-b.y, z-b.z);
	}
	Vec3 normalize(){
		// #TODO: Implement function to normalize vector
		// double length;
		// length = vecLength();
		return Vec3(x/vecLength(), y/vecLength(), z/vecLength());
	}
	Vec3 negate() {
		// #TODO: Implement function to negate vector
		return Vec3(-x, -y, -z);
	}
};

std::ostream& operator<< (std::ostream& os, const Vec3& v)
{
	os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	return os;
}

