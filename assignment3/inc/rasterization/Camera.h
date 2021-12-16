/* Base class for two types of camera*/
#pragma once
#include "Vec3.h"
#include "Matrix4.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
using namespace std;

class Camera{
public:
	Vec3 eye_position;
	Vec3 gaze_direction;
	Vec3 view_up;
	Vec3 u, v, w;

	// #TODO: You can declare some additional variables for computation in here

	Camera(Vec3 _eye_position, Vec3 _gaze_direction, Vec3 _view_up): eye_position(_eye_position), gaze_direction(_gaze_direction), view_up(_view_up) {
	}

	// #TODO: Compute u, v, w for camera
	void setCameraFrame() {
		w = gaze_direction.normalize().negate();
		u = view_up.crossProduct(w).normalize();
		v = w.crossProduct(u).normalize();
	}

	// #TODO: Compute View matrix for Camera transformation: M_cam
	Matrix4 calculateViewMatrix() {
		setCameraFrame();
		Matrix4 m_cam;
		// ???????
		Matrix4 temp;
		temp.identity();
		temp.matrix[0][0]=u.x;
		temp.matrix[0][1]=u.y;
		temp.matrix[0][2]=u.z;
		temp.matrix[1][0]=v.x;
		temp.matrix[1][1]=v.y;
		temp.matrix[1][2]=v.z;
		temp.matrix[2][0]=w.x;
		temp.matrix[2][1]=w.y;
		temp.matrix[2][2]=w.z;

		// 教材上7.4部分的矩阵
		m_cam= temp*Matrix4::Translate(eye_position.negate());
		return m_cam;
	}

	// #TODO: Compute orthographic projection matrix for Projection transformation: M_ortho
	Matrix4 calculateOrthographicMatrix(double l, double r, double b, double t, double n, double f) {
		Matrix4 m_orth;
		// 教材上7.3矩阵
		m_orth.identity();
		// m_orth= Matrix4::Scaling(Vec3(2.0/(r-l),2.0/(t-b),2.0/(n-f)))*Matrix4::Translate(Vec3(-(r+l)/2.0,-(t+b)/2.0,-(n+f)/2.0));
		m_orth.matrix[0][3]=-(r+l)/(r-l);
		m_orth.matrix[1][3]=-(t+b)/(t-b);
		m_orth.matrix[2][3]=-(n+f)/(n-f);
		m_orth.matrix[0][0]=2.0/(r-l);
		m_orth.matrix[1][1]=2.0/(t-b);
		m_orth.matrix[2][2]=2.0/(n-f);

		return m_orth;
	}

	// #TODO: Compute perspective projection matrix for Projection transformation: M_per
	// Note: To make sure that there is no distortion of shape in the image, using Field-of-View to compute perspective projection matrix
	Matrix4 calculatePerspectiveMatrix(double fovy, double aspect, double n, double f) {
		
		double l=-2.0f;
		double r=2.0f;
		double t=1.5f;
		double b=-1.5f;

		Matrix4 m_per=calculateOrthographicMatrix(l,r,b,t,n,f);

		return m_per;
	}

	// #TODO: Compute perspective projection matrix for Projection transformation using left, right, bottom, top: M_per
	Matrix4 calculatePerspectiveMatrix(double l, double r, double b, double t, double n, double f) {
		Matrix4 m_per;
		Matrix4 temp;
		
		temp.identity();
		temp.matrix[0][0]=n;
		temp.matrix[1][1]=n;
		temp.matrix[2][2]=n+f;
		temp.matrix[2][3]=(-f)*n;
		temp.matrix[3][2]=1;
		temp.matrix[2][3]=0;

		m_per=calculateOrthographicMatrix(l,r,b,t,n,f)*temp;

		return m_per;
	}
};



