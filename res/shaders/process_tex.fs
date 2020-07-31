#version 430 core
in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gDiffuseColor;
uniform sampler2D gMask;
uniform sampler2D gDepth;

layout (location = 0) out vec3 gPositionPost;
layout (location = 1) out vec3 gNormalPost;
layout (location = 2) out vec4 gDiffuseColorPost;
layout (location = 3) out float gMaskPost;
layout (location = 4) out float gDepthPost;

const float offset = 1.0 / 256;
const float eps = 1e-6;
const float maxGap = 0.1;

 vec2 offsets[8] = vec2[](
        vec2(-offset,  offset), // up left
        vec2( 0.0f,    offset), // up 
        vec2( offset,  offset), // up right
        vec2(-offset,  0.0f),   // left
        vec2( offset,  0.0f),   // right
        vec2(-offset, -offset), // bottom left
        vec2( 0.0f,   -offset), // bottom
        vec2( offset, -offset)  // bottom right
    );
    
void main(){
    gPositionPost = texture(gPosition, TexCoords).rgb;
    gNormalPost = texture(gNormal, TexCoords).rgb;
    gDiffuseColorPost = texture(gDiffuseColor, TexCoords).rgba;
    gMaskPost = texture(gMask, TexCoords).r;
    gDepthPost = texture(gDepth, TexCoords).r;
}