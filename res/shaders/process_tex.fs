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
    
    if (gMaskPost < eps){
        int near_valid = 0;
		for (int i = 0; i < 8; i++){
			if (abs(texture(gMask, TexCoords + offsets[i]).r - 1) < eps)
				near_valid++;
		}
		if (near_valid >= 6){
			gMaskPost = 1;
			for (int i = 0; i < 8; i++){
			    if (abs(texture(gMask, TexCoords + offsets[i]).r - 1) < eps 
			    && !isnan(texture(gNormal, TexCoords + offsets[i]).r)){	// valid point 
				    gDepthPost += texture(gDepth, TexCoords + offsets[i]).r / near_valid;
				    gPositionPost += texture(gPosition, TexCoords + offsets[i]).rgb / near_valid;
				    gNormalPost += texture(gNormal, TexCoords + offsets[i]).rgb;
				    gDiffuseColorPost += texture(gDiffuseColor, TexCoords).rgba / near_valid; 
				}
			}
			gNormalPost = normalize(gNormalPost);
		}
    }
    
    if (gMaskPost > 1 - eps){
        int near_valid = 0;
		for (int i = 0; i < 8; i++){
		    if (texture(gMask, TexCoords + offsets[i]).r > 1 - eps && 
		    gDepthPost - texture(gDepth, TexCoords + offsets[i]).r > maxGap){
		        near_valid++;
		    }
		}
		if (near_valid >= 6){
		    gDepthPost = 0.0;
		    gPositionPost = vec3(0.0);
		    gNormalPost = vec3(0.0);
		    gDiffuseColorPost = vec4(0.0);
		    for (int i = 0; i < 8; i++){
		        if (texture(gMask, TexCoords + offsets[i]).r > 1 - eps && 
		        texture(gDepth, TexCoords).r - texture(gDepth, TexCoords + offsets[i]).r > maxGap && 
		        !isnan(texture(gNormal, TexCoords + offsets[i]).r)){
		            gDepthPost += texture(gDepth, TexCoords + offsets[i]).r / near_valid;
				    gPositionPost += texture(gPosition, TexCoords + offsets[i]).rgb / near_valid;
				    gNormalPost += texture(gNormal, TexCoords + offsets[i]).rgb;
				    gDiffuseColorPost += texture(gDiffuseColor, TexCoords).rgba / near_valid; 
		        }
		    }
		    gNormalPost = normalize(gNormalPost);
		}
    }
    
    if (isnan(gNormalPost.r)){
        gPositionPost = vec3(0.0f);
        gNormalPost = vec3(0.0f);
        gDiffuseColorPost = vec4(0.0f);
        gMaskPost = 0.0f;
        gDepthPost = 0.0f;
    }
    
}