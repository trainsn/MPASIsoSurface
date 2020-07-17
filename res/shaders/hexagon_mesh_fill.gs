#version 430 core

layout (points) in;
layout (triangle_strip, max_vertices = 64) out;

in V2G{
	flat int cell_id;
}v2g[];

out G2F{
	vec3 normal;
	vec3 o_pos;
	vec4 e_pos;
}g2f;

//3D transformation matrices
uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat3 uNMatrix;

//connectivity
uniform samplerBuffer latVertex;
uniform samplerBuffer lonVertex;
uniform isamplerBuffer verticesOnCell;
uniform isamplerBuffer nEdgesOnCell;

uniform int maxEdges;
uniform float GLOBAL_RADIUS;

void main(){
	//only consider the outter most surface mesh.
	int cellId = v2g[0].cell_id;

	// query maxEdges neighboring vertexId
	int vertexIds[7];
	int nEdges = texelFetch(nEdgesOnCell, cellId  - 1).x;
	for (int i = 0; i < nEdges; i++)
		vertexIds[i] = texelFetch(verticesOnCell, (cellId-1) * maxEdges + i).x;

	// generate trianle strip
	// first corner
	int vertex_id = vertexIds[0];
	float lat = texelFetch(latVertex, vertex_id - 1).r;
	float lon = texelFetch(lonVertex, vertex_id - 1).r;
	vec4 xyzw0 = vec4(
							GLOBAL_RADIUS * cos(lat) * cos(lon),
							GLOBAL_RADIUS * cos(lat) * sin(lon),
							GLOBAL_RADIUS * sin(lat),
							1.0);

	// second corner
	vertex_id = vertexIds[1];
	lat = texelFetch(latVertex, vertex_id - 1).r;
	lon = texelFetch(lonVertex, vertex_id - 1).r;
	vec4 xyzw_prev = vec4(
							GLOBAL_RADIUS * cos(lat) * cos(lon),
							GLOBAL_RADIUS * cos(lat) * sin(lon),
							GLOBAL_RADIUS * sin(lat),
							1.0);
	
	for (int i = 2; i < nEdges; i++){
		// create triangle fans that form current hexagon cell.
		vertex_id = vertexIds[i];

		lat = texelFetch(latVertex, vertex_id - 1).r;
		lon = texelFetch(lonVertex, vertex_id - 1).r;

		vec4 xyzw_new = vec4(
							GLOBAL_RADIUS * cos(lat) * cos(lon),
							GLOBAL_RADIUS * cos(lat) * sin(lon),
							GLOBAL_RADIUS * sin(lat),
							1.0);

		gl_Position = uPMatrix * uMVMatrix * xyzw0;
		g2f.normal = uNMatrix * normalize(xyzw0.xyz);
		g2f.o_pos = xyzw0.xyz;
		g2f.e_pos = uMVMatrix * xyzw0;
		EmitVertex();

		gl_Position = uPMatrix * uMVMatrix * xyzw_prev;
		g2f.normal = uNMatrix * normalize(xyzw_prev.xyz);
		g2f.o_pos = xyzw_prev.xyz;
		g2f.e_pos = uMVMatrix * xyzw_prev;
		EmitVertex();

		gl_Position = uPMatrix * uMVMatrix * xyzw_new;
		g2f.normal = uNMatrix * normalize(xyzw_new.xyz);
		g2f.o_pos = xyzw_new.xyz;
		g2f.e_pos = uMVMatrix * xyzw_new;
		EmitVertex();

		EndPrimitive();
		xyzw_prev = xyzw_new;
	}
}