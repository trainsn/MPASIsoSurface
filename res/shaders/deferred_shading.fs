#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gDiffuseColor;
uniform sampler2D gMask;
uniform sampler2D gDepth;

uniform int uUseLighting;
uniform int uShowDepth;
uniform int uShowNormals;
uniform int uShowPosition;
uniform int uPerspectiveProjection;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat3 uNMatrix;

uniform vec3 uAmbientColor;
uniform vec3 uPointLightingLocation;
uniform vec3 uPointLightingColor;
uniform vec3 uPointLightingLocation1;
uniform vec3 uPointLightingColor1;

void main()
{
	// backgroundColor
    vec4 bgColor = vec4(1.0, 1.0, 1.0, 0.0);

	// retrive data from gbuffer
	vec3 vPosition = (uMVMatrix * vec4( texture(gPosition, TexCoords).rgb, 1.0 )).xyz;
	vec3 vTransformedNormal = texture(gNormal, TexCoords).rgb;
	vec4 vDiffuseColor = texture(gDiffuseColor, TexCoords).rgba;

	// calculate lighting as usual 
	vec3 ambient = vDiffuseColor.rgb * uAmbientColor;
	vec3 color = ambient;

	if (uUseLighting == 1){
		vec3 light_direction = normalize(uPointLightingLocation - vPosition.xyz);
		vec3 light_direction1 = normalize(uPointLightingLocation1 - vPosition.xyz);
		vec3 eye_direction = normalize(-vPosition.xyz);
		vec3 surface_normal = normalize(vTransformedNormal);

		vec3 diffuse = vDiffuseColor.xyz * uPointLightingColor * max(dot(surface_normal, light_direction), 0.0);
		vec3 diffuse1 = vDiffuseColor.xyz * uPointLightingColor1 * max(dot(surface_normal, light_direction1), 0.0);
		vec3 specular  = uPointLightingColor  * pow( max( dot( reflect( -light_direction,  surface_normal ), eye_direction ), 0.0 ), 100.0 );
		vec3 specular1 = uPointLightingColor1 * pow( max( dot( reflect( -light_direction1, surface_normal ), eye_direction ), 0.0 ), 100.0 );
		float light_outer_radius  = 20.0;
		float light_inner_radius  = 0.0;
		float light_outer_radius1 = 10.0;
		float light_inner_radius1 = 0.0;
		float light_distance  = length( vPosition.xyz - uPointLightingLocation  );
		float light_distance1 = length( vPosition.xyz - uPointLightingLocation1 );
		float attenuation     = 1.0 - smoothstep( light_inner_radius,  light_outer_radius,  light_distance  );
		float attenuation1    = 1.0 - smoothstep( light_inner_radius1, light_outer_radius1, light_distance1 );
		diffuse   = attenuation  * diffuse;
		diffuse1  = attenuation1 * diffuse1;
		specular  = attenuation  * specular;
		specular1 = attenuation1 * specular1;
		color = ambient + diffuse + diffuse1 + specular + specular1;
	}
	vec4 final_color = vec4(color, 1.0);
	float mask = texture(gMask, TexCoords).r;

	FragColor = mask * vec4(color, vDiffuseColor.a) + (1 - mask) * bgColor;

	if (uShowDepth == 1) {
		// FragColor = mix( vec4( 1.0 ), vec4( vec3( 0.0 ), 1.0 ), smoothstep( 0.1, 1.0, fog_coord ) );
		//FragColor = vDiffuseColor;
		//float depth = texture(gDepth, TexCoords).r;
		//FragColor = vec4( vec3(depth), 1.0 );
		float mask = texture(gMask, TexCoords).r;
		FragColor = vec4( vec3(mask), 1.0 );
		
	}
	if (uShowNormals == 1) {
		vec3 nTN      = normalize(vTransformedNormal);
		FragColor  = vec4( nTN, 1.0 );
	}
	if (uShowPosition == 1) {
		vec3 nP       = vPosition.xyz;
		FragColor  = vec4( nP.r, nP.g, nP.b, 1.0 );
	}
}
