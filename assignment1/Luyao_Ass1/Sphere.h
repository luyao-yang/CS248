#pragma once
#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "GeometricObject.h"
#include<cmath>
#include<iostream>

using namespace std;
	
class Sphere: public GeometricObject {
public:
	Vec3 center; // center point of the sphere
	double radius; // radius of the sphere
	double A, B, C, delta;
	double t=0.0;

	// #TODO: You can declare some additional variables for computation in here
	Sphere(Vec3 _center, double _radius, Color _color) : GeometricObject(_color), center(_center), radius(_radius){}

	double testIntersection(Ray ray)
	{
		// #TODO: Implement function to check ray intersects with sphere or not, return t
		// Vec3 viewPos(0.0f, 0.0f, 1.0f);

		// d . d
		A = ray.direction.dotProduct(ray.direction);
		// 2d.(e-c)
		B = ray.direction.dotProduct(ray.origin-center)*2.0;
		// (e-c)(e-c)-R2
		Vec3 dif = ray.origin-center;
		C = dif.dotProduct(dif) - pow(radius,2);
		delta = pow(B,2) - 4*A*C;
		// cout<<delta;
		if (delta >= 0){
			// double t1 = (-B + sqrt(delta))/(2*A);
			// double t2 = (-B - sqrt(delta))/(2*A);
			// cout<<"here";
			return 1;
		}else {
			return 0;
		}
	}

	Vec3 computeIntersectionPoint(Ray ray, double t)
	{
		// #TODO: Implement function to find intesection point, return intersection point
		// Vec3 viewPos(0.0f, 0.0f, 1.0f);
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
		if (testIntersection(ray)){
			Vec3 p = computeIntersectionPoint(ray,t);
			return ((p-center)/radius).normalize();
		}else{
			return Vec3(0.0);
	}}
};
