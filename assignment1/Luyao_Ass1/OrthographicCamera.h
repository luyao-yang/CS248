#pragma once
#include "Ray.h"
#include "Camera.h"
#include<iostream>

using namespace std;

class OrthographicCamera : public Camera{
public:
	Vec3 u, v, w;

	using Camera::Camera;
	
	void setCameraFrame() {
		// #TODO: Set up 3 basis vector for camera
		w = direction.negate().normalize();
		u = up.crossProduct(w).normalize();
		v = w.crossProduct(u).normalize();
	}

	void getRay(Ray& outRay, double ui, double vj){
		// #TODO: Get view ray from camera to the pixel	
		outRay.origin = Vec3(Camera::position + u*ui + v*vj);
		outRay.direction = w.negate().normalize();
	}

};

std::ostream& operator<< (std::ostream& os, const OrthographicCamera& camera)
{
	os << "Orthographic basis vectors: u = " << camera.u << "; v = " << camera.v << "; w = " << camera.w << "\n";
	return os;
}
