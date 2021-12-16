#pragma once
#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "GeometricObject.h"
#include "Plane.h"
#include<cmath>
#include<iostream>

using namespace std;
	
class Cylinder : public GeometricObject {
public:
    double radius;
    double height;
    Vec3 center;
	double A,B,C,delta,t1,t2,dis;
    // Plane p1(center.x,center.y-height/2,center.z,0.0,_color);
    // Plane p2(center.x,center.y+height/2,center.z,0.0,_color);
    Vec3 p;
    int type;
    double temp1=0.0;
    double temp2=0.0;
    
    // #TODO: You can declare some additional variables for computation in here

    Cylinder(Vec3 _center, double _radius, double _height, Color _color) : GeometricObject(_color), center(_center), radius(_radius), height(_height)
    {
    }

    double testIntersection(Ray ray)
    {
        // #TODO: Implement function to check ray intersects with cylinder or not, return t
        int type=0;
        // temp1=p1.testIntersection(ray);
        // temp2=p2.tesetIntersection(ray);
        // if(temp1){
        //     p = p1.computeIntersectionPoint(ray,temp1);
        //     double l_p1 = (p-center).dotProduct(p-center);
        //     if(l_p1<radius*radius){
        //         type=1;
        //         return 1;
        //     }
        // }

        // if(temp2){
        //     p = p2.computeIntersectionPoint(ray,t);
        //     double l_p2 = (p-center).dotProduct(p-center);
        //     if(l_p2<radius*radius){
        //         type=1;
        //         return 1;
        //     }
        // }
		A = pow(ray.direction.x,2)+pow(ray.direction.z,2);
		B = (ray.origin.x*ray.direction.x+ray.direction.z*ray.origin.z)*2.0;
		C = pow(ray.origin.x,2) + pow(ray.origin.z,2) - pow(radius,2);
		delta = pow(B,2) - 4*A*C;
		// cout<<delta;
		t1 = (-B + sqrt(delta))/(2*A);
		t2 = (-B - sqrt(delta))/(2*A);
		dis=min(t1,t2);	
		if (delta >= 0){
			double r = ray.origin.y + ray.direction.y*dis;
			// return 1;
			// cout<<r;
			if ((r <= center.y+height/2) and (r >= center.y-height/2)){
				// cout<<"here";
				return 1;
			}else
				return 0;
		}else 
			return 0;

    }

    Vec3 computeIntersectionPoint(Ray ray, double t)
    {
		t=testIntersection(ray);
        if(t){ 	
			// if(type==1){
            //     // Plane p_bottom(pn,0.0,Color(0.0f, 0.0f, 1.0f));
            //     if(temp1)
            //         return p1.computeIntersectionPoint(ray,t);
            //     else
            //         return p2.computeIntersectionPoint(ray,t);
            // }else
            return ray.origin+ray.direction*dis;
		}
		return Vec3(0.0);
    }

    Vec3 computeNormalIntersection(Ray ray, double t)
    {
        // #TODO: Implement function to find normal vector at intesection point, return normal vector
		if (testIntersection(ray)){
            // if(type==1){
            //     if(temp1){
            //     // Plane p_bottom(pn,0.0,Color(0.0f, 0.0f, 1.0f));
            //     return p1.computeNormalIntersection(ray,t).normalize();
            //     }else
            //         return p2.computeNormalIntersection(ray,t).normalize();
            // }else{
            Vec3 p = computeIntersectionPoint(ray,t);
            Vec3 n = Vec3(p.x-center.x,0,p.z-center.z).normalize();
            return n;
            }else
			    return Vec3(0.0);
	}

};

