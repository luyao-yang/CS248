#pragma once
#include <vector>
using std::vector;
#include "Vec3.h"
#include "Matrix4.h"
#include "Color.h"

#include <string>
using std::string;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <fstream>
using std::ifstream;
#include <sstream>
using std::istringstream;

// Material for object
class Material
{
public:
    Color Ka; // Ambient coefficient
    Color Kd; // Diffuse coefficient
    Color Ks; // Specular coefficient
    double shininess; // Phong exponent

    Material(Color _ka, Color _kd, Color _ks, double _shininess):
        Ka(_ka), Kd(_kd), Ks(_ks), shininess(_shininess){}  
};

// Each object is a set of triangle
class Triangle
{
public:
    Vec3 v0, v1, v2;                    // Three vertices of triangle
    int v0_index, v1_index, v2_index;
    Vec3 faceNormal;                    // Face normal of triangle
    Vec3 v0Normal, v1Normal, v2Normal;  // Three normals for vertices of triangle
    Color c0, c1, c2;                   // Three color for vertices
    int boudingBox[4];                  //Bouding box of triangle: xmin, ymin, xmax, ymax


    Triangle() {};
    Triangle(Vec3 _v0, Vec3 _v1, Vec3 _v2, Vec3 _faceNormal, Vec3 _v0Normal, Vec3 _v1Normal, Vec3 _v2Normal)
        : v0(_v0), v1(_v1), v2(_v2), faceNormal(_faceNormal), v0Normal(_v0Normal), v1Normal(_v1Normal), v2Normal(_v2Normal)
    {
        boudingBox[0] = 0;
        boudingBox[1] = 0;
        boudingBox[2] = 0;
        boudingBox[3] = 0;
        c0 = Color(1.0f, 0.0f, 0.0f);
        c1 = Color(0.0f, 1.0f, 0.0f);
        c2 = Color(0.0f, 0.0f, 1.0f);
    }

    float min3(const double& a, const double& b, const double& c)
    {
        return std::min(a, std::min(b, c));
    }

    float max3(const double& a, const double& b, const double& c)
    {
        return std::max(a, std::max(b, c));
    } 

    // Use this function for compute bounding box of the triangle
    void calculateBoundingBox()
    {
        boudingBox[0] = std::floor(min3(v0.x, v1.x, v2.x));
        boudingBox[1] = std::floor(min3(v0.y, v1.y, v2.y));
        boudingBox[2] = std::ceil(max3(v0.x, v1.x, v2.x));
        boudingBox[3] = std::ceil(max3(v0.y, v1.y, v2.y));
    }
};

// Each object is represented by a mesh
class Mesh
{
public:
    // Each mesh has these attributes:
    vector<Vec3> points;        // list of vertices
    vector<Vec3> normals;       // list of normal for each vertex
    vector<int> faces;          // list face index
    vector<Vec3> faceNormals;   // list of face normal for each face
    vector<Triangle> triangles; // list of triangles
   
    // 每一个点对应点面的序号
    vector<vector<int> > vertexlink;
    // 每一个面对应的点的序号Vec3<1,2,3>
    vector<Vec3> pointlink;
    bool isteapot=true;
    // bool issphere=false;


    // #TODO: Implement function for calculate vertex normal if the mesh does not contain normal for each vertex
    // Reference:  http://web.missouri.edu/~duanye/course/cs4610-spring-2017/assignment/ComputeVertexNormal.pdf
    void generateVertexNormals(
        const vector<Vec3>& points,
        vector<Vec3>& normals,
        const vector<int>& faces)
    {
        int c=0;
        if(isteapot){
         
            Vec3 Fnorm=Vec3(0);
            vector<int> faceids;
            for(int i=0; i<points.size();i++){
                // c=c+1;
                // if(c==3){
                //     break;
                // }
                faceids=vertexlink[i];
                // cout<<faceids[0]<<endl;
                // if(faceids[0]==0){
                //     cout<<"here"<<endl;
                // }
                for(int j=0;j<faceids.size();j++){
                    Fnorm=Fnorm+faceNormals[faceids[j]];
                }
                Fnorm=Fnorm.normalize();
                normals.push_back(Fnorm);
            
            }
            for(int j=0; j<triangles.size();j++){
                int v0_index=triangles[j].v0_index;
                int v1_index=triangles[j].v1_index;
                int v2_index=triangles[j].v2_index;
                // if(v0_index==-1){
                //     cout<<"here"<<endl;
                // }

                triangles[j].v0Normal=normals[v0_index];
                triangles[j].v1Normal=normals[v1_index];
                triangles[j].v2Normal=normals[v2_index];
                // cout<<triangles[j].v0Normal<<endl;
            }
   
            }

        

    }

    // #TODO: Implement function for calculate face normal for each face (triangle) of the mesh
    void generateFaceNormals(
        const vector<Vec3>& points,
        const vector<Vec3>& pointlink,
        // const vector<int>& faces,
        vector<Vec3>& faceNormals)
    {
        Vec3 index;
        Vec3 A,B,C;
        for(int i=0; i<pointlink.size(); i++){
            index=pointlink[i];
        
            A=points[index.x];
            B=points[index.y];
            C=points[index.z];

            Vec3 l1=B-A;
            Vec3 l2=C-A;
            triangles[i].faceNormal=l1.crossProduct(l2).normalize();
            faceNormals.push_back(triangles[i].faceNormal);
            }
       

    }

    Mesh()
    {
    }
    Mesh(const char* fileName)
    {
        loadOBJ(fileName);
        
    }

    // #TODO: Implement function to read the data for the mesh from an OBJ file
    void loadOBJ(const char* fileName)
    {
        // bool isteapot=true;
        // int c=0;
        //记录face的id
        int face_idx=0;

        FILE * file = fopen(fileName, "r");
        if( file == NULL ){
            printf("Impossible to open the file !n");
            }
        while(1){
            char lineHeader[128];
            
            // read the first word of the line
            int res = fscanf(file, "%s", lineHeader);
            if (res == EOF)
                break;
            if ( strcmp( lineHeader, "v" ) == 0 ){
                Vec3 vertex;
                fscanf(file, "%lf %lf %lfn", &vertex.x, &vertex.y, &vertex.z );
                points.push_back(vertex);
                // 初始化这个二重数组
                vector<int> empty;
                vertexlink.push_back(empty);

            }else if ( strcmp( lineHeader, "vn" ) == 0 ){
                isteapot=false;
                Vec3 normal;
                fscanf(file, "%lf %lf %lfn", &normal.x, &normal.y, &normal.z );
                normals.push_back(normal);
            }else if ( strcmp( lineHeader, "f" ) == 0 ){
                // c=c+1;
                // if(c==20)
                //     return;
                faces.push_back(face_idx);
                int v0,v1,v2; 
                unsigned int vertexIndex[3], normalIndex[3];

                if(isteapot)
                    fscanf(file, "%d %d %dn", &vertexIndex[0], &vertexIndex[1],  &vertexIndex[2] );
                else
                    fscanf(file, "%d//%d %d//%d %d//%dn", &vertexIndex[0], &normalIndex[0], \
                    &vertexIndex[1], &normalIndex[1], &vertexIndex[2], &normalIndex[2]);
                
                v0=vertexIndex[0]-1;
                v1=vertexIndex[1]-1;
                v2=vertexIndex[2]-1; 

                // 记录每一个点序号对应点面序号
                pointlink.push_back(Vec3(v0,v1,v2));
                
                vertexlink[v0].push_back(face_idx);
                vertexlink[v1].push_back(face_idx);
                vertexlink[v2].push_back(face_idx);

                Triangle triangle;
                triangle.v0 = points[v0];
                triangle.v1 = points[v1];
                triangle.v2 = points[v2];
                triangle.v0_index=v0;
                triangle.v1_index=v1;
                triangle.v2_index=v2;



                if(!isteapot){
                    triangle.v0Normal = normals[v0];
                    triangle.v1Normal = normals[v1];
                    triangle.v2Normal = normals[v2];
                }
                
                triangles.push_back(triangle);

                face_idx+=1;
            }
        }
        
        // for(int i=0;i<triangles.size();i++){
        //     printf("111111");
        //     cout<<triangles[i].v0<<endl;
        //     cout<<triangles[i].v1<<endl;
        //     cout<<triangles[i].v2<<endl;

        // }
    }
};

