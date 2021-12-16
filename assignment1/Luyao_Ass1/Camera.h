#pragma once
#include "Vec3.h"
#include "Ray.h"
/* Base class for two types of camera*/
#pragma once
#include<iostream>
using namespace std;

class Camera{
public:
	Vec3 position;
	Vec3 direction;
	Vec3 up;

	Camera(Vec3 _position, Vec3 _direction, Vec3 _up): position(_position), direction(_direction), up(_up) {
		cout<<direction;
	}

	virtual void setCameraFrame() = 0;	
	virtual void getRay(Ray &outRay, double ui, double vj) = 0;

	/**
	 * Code for unit testing of cameras.
	 */
	void testGetRay(Ray correctRay, double ui, double vj) {
		Ray testRay;
		getRay(testRay, ui, vj);
		if (!raysEquivalent(testRay, correctRay)) {
			cout << "\ntest failed";
			cout << "\ntestRay: " << testRay.origin << " + t * " << testRay.direction;
			cout << "\ncorrectRay: " << correctRay.origin << " + t * " << correctRay.direction;
			return;
		}
		cout << "\nTest successful.";
	}

private:
	static bool raysEquivalent(Ray ray1, Ray ray2) {
		Vec3 dir1(ray1.direction);
		Vec3 dir2(ray2.direction);
		dir1.normalize();
		dir2.normalize();
		dir1 = dir1 - dir2;
		Vec3 dist = ray1.origin - ray2.origin;
		return dist.vecLength() < 1e-6 && dir1.vecLength() < 1e-6;
	}
};


