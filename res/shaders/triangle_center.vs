#version 430 core
layout(location = 0) in int vertex_id;

//3D transformation matrices
uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;

uniform samplerBuffer latVertex;
uniform samplerBuffer lonVertex;

uniform float GLOBAL_RADIUS;

void main() {
	float lat = texelFetch(latVertex, vertex_id-1).x;
	float lon = texelFetch(lonVertex, vertex_id-1).x;
	vec4 xyzw = vec4(
						GLOBAL_RADIUS * cos(lat) * cos(lon),
						GLOBAL_RADIUS * cos(lat) * sin(lon),
						GLOBAL_RADIUS * sin(lat),
						1.0);
	gl_Position = uPMatrix * uMVMatrix * xyzw;
}