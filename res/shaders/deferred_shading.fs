#version 430 core

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gDiffuseColor;
uniform sampler2D gMask;
uniform sampler2D gDepth;

void main()
{
	//FragColor = vec4( texture(gPosition, TexCoords).rgb, 1.0);
	FragColor = vec4( normalize(texture(gNormal, TexCoords).rgb), 1.0 );
	//FragColor = vec4(texture(gMask, TexCoords).r, vec3(1.0));
}