#version 430 core
layout (location = 0) in int in_vertex_id;

out V2G{
	flat int vertex_id;
}v2g;

void main(){
	v2g.vertex_id = in_vertex_id;
}
