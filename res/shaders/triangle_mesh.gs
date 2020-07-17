#version 430 core 
layout (points) in;
layout (triangle_strip, max_vertices = 3) out;
//layout (points, max_vertices = 3) out;

in V2G{
	flat int vertex_id;
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
uniform samplerBuffer latCell;
uniform samplerBuffer lonCell;
uniform isamplerBuffer cellsOnVertex;

uniform float GLOBAL_RADIUS;

void main(){
	// only consider outter most surface mesh 
	int vertexId = v2g[0].vertex_id;
	// query three nerighboring cell id.
	ivec3 cellId3 = texelFetch(cellsOnVertex, vertexId - 1).xyz;
	if (cellId3.x == 0 || cellId3.y == 0 || cellId3.z == 0){ // on boundary
	}
	else {
		int cellIds[3];
	
		cellIds[0] = cellId3.x;
		cellIds[1] = cellId3.y;
		cellIds[2] = cellId3.z;

		for (int i = 0; i < 3; i++){
			int cell_id = cellIds[i];
		
			float lat = texelFetch(latCell, cell_id - 1).x;
			float lon = texelFetch(lonCell, cell_id - 1).x; 
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
}