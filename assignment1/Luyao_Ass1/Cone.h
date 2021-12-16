#pragma once
#include "Vec3.h"
#include "Color.h"
#include "Ray.h"
#include "GeometricObject.h"
#include "Plane.h"
#include<cmath>
#include<iostream>


using namespace std;
	
class Cone : public GeometricObject {
public:
	double height;
	double rOverh2;
	Vec3 apex; // tip or vertex or apex of cone
	double A,B,C,radius;
	double m,delta;
	double t,t1,t2,dis;
	Vec3 center;
	Vec3 hi,p;
    int type;
    Vec3 pn=(center-apex).normalize();


	Cone(Vec3 _apex, double _radius, double _height, Color _color) : GeometricObject(_color), apex(_apex), radius(_radius), height(_height)
	{}

	double testIntersection(Ray ray)
	{
        type=0;
        center = (apex.x,apex.y,apex.z-height);

        Vec3 pn=(apex-center).normalize();
        Plane p_bottom(pn,0.0,Color(0.0f, 0.0f, 1.0f));
        t=p_bottom.testIntersection(ray);
        if (t){
            p = p_bottom.computeIntersectionPoint(ray,t);
            double l_bottom = (p-center).dotProduct(p-center);
            if(l_bottom<radius*radius){
                type=1;
                return 1;
            }
        }

		m=(radius*radius)/(height*height);
		Vec3 w=ray.origin-apex;
		Vec3 p;

		hi = (center-apex).normalize();

		A = ray.direction.dotProduct(ray.direction)-(m+1)*(pow(ray.direction.dotProduct(hi),2));
		B = 2*(ray.direction.dotProduct(w)-m*(ray.direction.dotProduct(hi)*w.dotProduct(hi))-ray.direction.dotProduct(hi)*w.dotProduct(hi));
		C = w.dotProduct(w)-m*(pow(w.dotProduct(hi),2))-pow(w.dotProduct(hi),2);
		double cos=hi.vecLength()/sqrt(pow(hi.vecLength(),2)+pow(radius,2));
		
		delta = pow(B,2)-4*A*C;
		if (delta>0){
			t1 = (-B + sqrt(delta))/(2*A);
			t2 = (-B - sqrt(delta))/(2*A);
			dis=min(t1,t2);	
			p=ray.origin+ray.direction*dis;
			double temp_height=p.dotProduct(hi);
			if(temp_height<=hi.vecLength() && temp_height>=0){
				return 1;
			}else
				return 0;		
		}else if (delta==0){
			double temp=ray.direction.dotProduct(hi);
			if(cos==temp){
				return 0;
			}else{
				dis=-B/(2*A);
				p=ray.origin+ray.direction*dis;
				double temp_height=p.dotProduct(hi);
				if(temp_height<=hi.vecLength() && temp_height>=0){
					return 1;	
				}else
					return 0;
			}
		}else
			return 0;
	}

	Vec3 computeIntersectionPoint(Ray ray, double t)
	{
		// #TODO: Implement function to find intesection point, return intersection point
		if(t){
             if(type==1){
                Plane p_bottom(pn,0.0,Color(0.0f, 0.0f, 1.0f));
                return p_bottom.computeIntersectionPoint(ray,t);
            }else if(type==0){
                return ray.origin+ray.direction*dis;
            }
		}else
			return Vec3(0.0);
	}

	Vec3 computeNormalIntersection(Ray ray, double t)
	{
		// #TODO: Implement function to find normal vector at intesection point, return normal vector
		if(t){
            if(type==1){
                Plane p_bottom(pn,0.0,Color(0.0f, 0.0f, 1.0f));
                return p_bottom.computeNormalIntersection(ray,t).normalize();
            }else{
                Vec3 p=computeIntersectionPoint(ray,t);
			    Vec3 n = Vec3(p.x-center.x,0,p.z-center.z);
    		    return n.normalize();
            }
		}else
			return (0.0);

	}
};

