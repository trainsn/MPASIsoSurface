#version 430 core
layout (location = 0) in int in_vertex_id;
layout (location = 1) in int in_layer_id;

out V2G{
	flat int vertex_id;
	flat int layer_id;
}v2g;

void main(){
	v2g.vertex_id = in_vertex_id;
	v2g.layer_id = in_layer_id;
}