#pragma once
#include "Ray.h"
#include "Camera.h"
#include<iostream>
using namespace std;
class PerspectiveCamera : public Camera {
public:
	Vec3 w, u, v;
	double distance = 1.0f; 
	
	using Camera::Camera;
	// Need to comment?
	
	void setCameraFrame() {
		// #TODO: Set up 3 basis vector for camera

		// cout<<"test"<<Camera::direction<<"test"<<endl;
		w = Camera::direction.negate().normalize();
		u = Camera::up.crossProduct(w).normalize();
		v = w.crossProduct(u).normalize();
	}

	void getRay(Ray& outRay, double ui, double vj) {
		// #TODO: Get view ray from camera to the pixel
		// eyeview point
		outRay.direction = ((w*distance).negate() + u*ui + v*vj).normalize();
		// outRay.direction=outRay.direction.normalize();
		outRay.origin = Camera::position;
	}
};

std::ostream& operator<< (std::ostream& os, const PerspectiveCamera& camera)
{
	os << "Perspective basis vectors: u = " << camera.u << "; v = " << camera.v << "; w = " << camera.w << "\n";
	// os <<"Perspective basis vectors: u = " << camera.direction << "\n";
	
	return os;
}
