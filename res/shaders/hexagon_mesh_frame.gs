#version 430 core

layout (points) in;
layout (line_strip, max_vertices = 7) out;

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
	//only consider outter most surface mesh
	int cellId = v2g[0].cell_id;

	// query maxEdges neighboring vertexId
	int vertexIds[7];
	int nEdges = texelFetch(nEdgesOnCell, cellId  - 1).x;
	for (int i = 0; i < nEdges; i++)
		vertexIds[i] = texelFetch(verticesOnCell, (cellId-1) * maxEdges + i).x;

	// generate line strip
	for (int i = 0; i < nEdges; i++){
		int vertex_id = vertexIds[i];
		float lat = texelFetch(latVertex, vertex_id - 1).r;
		float lon = texelFetch(lonVertex, vertex_id - 1).r;

		vec4 xyzw = vec4(
							GLOBAL_RADIUS * cos(lat) * cos(lon),
							GLOBAL_RADIUS * cos(lat) * sin(lon),
							GLOBAL_RADIUS * sin(lat),
							1.0);
		gl_Position = uPMatrix * uMVMatrix * xyzw;
		g2f.normal = uNMatrix * normalize(xyzw.xyz);
		g2f.o_pos = xyzw.xyz;
		g2f.e_pos = uMVMatrix * xyzw;
		EmitVertex();
	}
	EndPrimitive();
}