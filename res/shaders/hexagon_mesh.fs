#version 430 core 
in G2F{
	vec3 normal;
	vec3 o_pos;
	vec3 e_pos;
}g2f;

out vec4 out_Color;

void main(){
	vec3 vNormal = normalize(g2f.normal);
	if (vNormal.z < -0.00){
		discard;	//only draw visble surface.
	}
	out_Color = vec4(g2f.o_pos, 1.0);
}