#pragma once
#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "GeometricObject.h"
#include<cmath>
#include<iostream>

using namespace std;
	
class Ellipsoid : public GeometricObject {
public:
	double aRadius; // radius along x axis
	double bRadius; // radius along y axis
	double cRadius; // radius along z axis
	Vec3 center; // center of ellipsoid
	double A,B,C,delta;

	// #TODO: You can declare some additional variables for computation in here
	
	Ellipsoid(Vec3 _center, double _aRadius, double _bRadius, double _cRadius, Color _color) : GeometricObject(_color), center(_center), aRadius(_aRadius), bRadius(_bRadius), cRadius(_cRadius) {}
	
	double testIntersection(Ray ray)
	{
		// #TODO: Implement function to check ray intersects with ellipsoid or not, return t
		A = ((pow(ray.direction.x,2))/(pow(aRadius,2)))+((pow(ray.direction.y,2))/(pow(bRadius,2)))+((pow(ray.direction.z,2))/(pow(cRadius,2)));
		B = 2*((ray.direction.x*(ray.origin.x-center.x)/(pow(aRadius,2)))+(ray.direction.y*(ray.origin.y-center.y)/(pow(bRadius,2)))+(ray.direction.z*(ray.origin.z-center.z)/(pow(cRadius,2))));
		C = (pow((ray.origin.x-center.x),2)/pow(aRadius,2)+pow((ray.origin.y-center.y),2)/pow(bRadius,2)+pow((ray.origin.z-center.z),2)/pow(cRadius,2))-1; 
		delta = pow(B,2) - 4*A*C;
		if (delta >= 0){
			return 1;
		}else 
			return 0;
	}

	Vec3 computeIntersectionPoint(Ray ray, double t)
	{
		// #TODO: Implement function to find intesection point, return intersection point
		if(t){ 
			double t1 = (-B + sqrt(delta))/(2*A);
			double t2 = (-B - sqrt(delta))/(2*A);
			double dis=min(t1,t2);	
			return ray.origin+ray.direction*dis;
		}
		return Vec3(0.0);

	}

	Vec3 computeNormalIntersection(Ray ray, double t)
	{
		// #TODO: Implement function to find normal vector at intesection point, return normal vector
		if(testIntersection(ray)){
			Vec3 p = computeIntersectionPoint(ray,t);
			double dx=2*(p.x-center.x)/(pow(aRadius,2));
			double dy=2*(p.y-center.y)/(pow(bRadius,2));
			double dz=2*(p.z-center.z)/(pow(cRadius,2));
			return Vec3(dx,dy,dz).normalize();
		}
		return Vec3(0.0);
	}
	
};

