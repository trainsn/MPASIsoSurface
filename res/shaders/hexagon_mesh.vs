#version 430 core

layout (location = 0) in int in_cell_id;

out V2G{
	flat int cell_id;
}v2g;

void main(){
	v2g.cell_id = in_cell_id;
}