#pragma once
#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "GeometricObject.h"
#include<cmath>
#include<iostream>

using namespace std;

class Plane : public GeometricObject {
public:
	double A, B, C, D; // Plane equation: Ax + By + Cz + D = 0
	Vec3 Pn; // Normal vector of the plane
	// Vec3 viewPos(0.0f, 0.0f, 1.0f);
	double t = 0.0;
	double denom = 0.0;
	double dis=0.0;
	// denom = Pn.dotProduct(ray.direction);
	// #TODO: You can declare some additional variables for computation in here
	Vec3 p0;
	double x0,y0,z0;

	
	Plane(double _A, double _B, double _C, double _D, Color _color) : GeometricObject(_color), A(_A), B(_B), C(_C), D(_D)
	{
		Pn = Vec3(_A, _B, _C).normalize();
		x0=1.0;
		z0=1.0;
		// if(_C)
		double y0=(-_D-_A*x0-_C*z0)/_B;
		// else
		p0 = Vec3(x0,y0,z0);
	}
    
    Plane(Vec3 _Pn, double _D, Color _color) : GeometricObject(_color), A(_Pn.x), B(_Pn.y), C(_Pn.z), D(_D), Pn(_Pn.normalize()) {}

	double testIntersection(Ray ray)
	{
		// #TODO: Implement function to check ray intersects with sphere or not, return t.
		// Vec3 viewPos(0.0f, 0.0f, 1.0f);
		denom = Pn.dotProduct(ray.direction);
		
		if(fabs(denom)>0.0001f){
			double numerator =(p0-ray.origin).dotProduct(Pn);
			// cout<<numerator;
			dis = numerator/denom;
			if (dis>0.0001f){
				return 1;
			}else
				return 0;
		};
		
	}

	Vec3 computeIntersectionPoint(Ray ray, double t)
	{
		// #TODO: Implement function to find intesection point, return intersection point
		if(t){ 
			return ray.origin+ray.direction*dis;		
		}
		return Vec3(0.0);
	}

	Vec3 computeNormalIntersection(Ray ray, double t)
	{
		// #TODO: Implement function to find normal vector at intesection point, return normal vector
		if (testIntersection(ray)){
			if(denom>0)
				return Pn.negate().normalize();
			return Pn.normalize();
		}else
			return Vec3(0.0);
	}
};
