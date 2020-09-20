#version 430 core

layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out float gSalinity;
layout (location = 3) out float gMask;
layout (location = 4) out float gDepth;

uniform int maxEdges;
uniform float threshold;

in G2F{
	flat int triangle_id;
	flat int layer_id;
	flat int hitFaceid;
	smooth vec3 o_pos;
}g2f;

uniform mat4 uInvMVMatrix;
uniform float stepsize;
//3D transformation matrices
uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat3 uNMatrix;

//connectivity
uniform samplerBuffer latCell;
uniform samplerBuffer lonCell;
uniform isamplerBuffer cellsOnVertex;
uniform isamplerBuffer edgesOnVertex;
uniform isamplerBuffer cellsOnEdge;
uniform isamplerBuffer verticesOnEdge;
uniform isamplerBuffer verticesOnCell;
uniform isamplerBuffer nEdgesOnCell;
uniform isamplerBuffer maxLevelCell;
uniform samplerBuffer temperature;
uniform samplerBuffer salinity;
//symbolic definition of dual triangle mesh
#define TRIANGLE_TO_EDGES_VAR edgesOnVertex
#define EDGE_CORNERS_VAR cellsOnEdge
#define CORNERS_LAT_VAR latCell
#define CORNERS_LON_VAR lonCell
#define FACEID_TO_EDGEID faceId_to_edgeId
#define EDGE_TO_TRIANGLES_VAR verticesOnEdge
#define CORNER_TO_TRIANGLES_VAR verticesOnCell
#define CORNER_TO_TRIANGLES_DIMSIZES nEdgesOnCell

#define FLOAT_MAX  3.402823466e+38F
#define FLOAT_MIN -2.402823466e+36F
uniform float GLOBAL_RADIUS;
uniform float THICKNESS;
uniform int TOTAL_LAYERS;

int	d_mpas_faceCorners[24] = {
    0, 1, 2,  3, 4, 5,//top 0 and bottom 1
    4, 2, 1,  4, 5, 2,//front 2,3
    5, 0, 2,  5, 3, 0,//right 4,5
    0, 3, 1,  3, 4, 1//left 6,7
};

struct Ray{
	dvec3 o;
	dvec3 d;
};

struct HitRec{
	double t;	//  t value along the hitted face 
	int hitFaceid;	// hitted face id 
	int nextlayerId;
};

struct MPASPrism {
	uint m_prismId;
	int m_iLayer;
	dvec3 vtxCoordTop[3]; // 3 top vertex coordinates
	dvec3 vtxCoordBottom[3]; // 3 botton vertex coordinates 
	int m_idxEdge[3];
	int idxVtx[3]; // triangle vertex index is equvalent to hexagon cell index 
};

#define DOUBLE_ERROR 1.0e-8
bool rayIntersectsTriangleDouble(dvec3 p, dvec3 d,
    dvec3 v0, dvec3 v1, dvec3 v2, inout double t)
{
    dvec3 e1, e2, h, s, q;
    double a, f, u, v;
    //float error = 1.0e-4;//0.005f;
    e1 = v1 - v0;
    e2 = v2 - v0;
    //crossProduct(h, d, e2);
    h = cross(d, e2);
    a = dot(e1, h);//innerProduct(e1, h);

    if (a > -DOUBLE_ERROR && a < DOUBLE_ERROR)
        return(false);

    f = 1.0 / a;
    s = p - v0;//_vector3d(s, p, v0);
    u = f * dot(s, h);//(innerProduct(s, h));

    if (u < -DOUBLE_ERROR || u >(1.0 + DOUBLE_ERROR))
        return(false);

    q = cross(s, e1);//crossProduct(q, s, e1);
    v = f * dot(d, q);//innerProduct(d, q);

    if (v < -DOUBLE_ERROR || u + v >(1.0 + DOUBLE_ERROR))
        return(false);

    // at this stage we can compute t to find out where
    // the intersection point is on the line
    t = f * dot(e2, q);//innerProduct(e2, q);

    if (t > DOUBLE_ERROR)//ray intersection
        return(true);
    else // this means that there is a line intersection
        // but not a ray intersection
        return (false);
}

bool ReloadVtxInfo(in int triangle_id, in int iLayer, inout MPASPrism prism) {
	prism.m_prismId = triangle_id;
	prism.m_iLayer = iLayer;

	prism.idxVtx[0] = -1;
	prism.idxVtx[1] = -1;
	prism.idxVtx[2] = -1;

	// load first edge 
	// index of first edge of triangle
	ivec3 idxEdges = texelFetch(TRIANGLE_TO_EDGES_VAR, triangle_id - 1).xyz;
	int idxEdge = idxEdges.x; // TRIANGLE_TO_EDGES_VAR[m_prismId * 3 + 0];
	prism.m_idxEdge[0] = idxEdge;
	// index of start corner of this edge
	ivec2 cornerIdxs = texelFetch(EDGE_CORNERS_VAR, idxEdge - 1).xy;
	int iS = cornerIdxs.x; //EDGE_CORNERS_VAR[idxEdge * 2];	
	// index of end corner of this edge 
	int iE = cornerIdxs.y; //EDGE_CORNERS_VAR[idxEdge * 2 + 1];
	prism.idxVtx[0] = iS;
	prism.idxVtx[1] = iE;
	int edge1E = iE;

	// load second edge 
	idxEdge = idxEdges.y; // TRIANGLE_TO_EDGES_VAR[m_prismId * 3 + 1];
	prism.m_idxEdge[1] = idxEdge;
	// index of start corner of second edge
	cornerIdxs = texelFetch(EDGE_CORNERS_VAR, idxEdge - 1).xy;
	iS = cornerIdxs.x; //EDGE_CORNERS_VAR[idxEdge * 2];	
	// index of end corner of this edge 
	iE = cornerIdxs.y; //EDGE_CORNERS_VAR[idxEdge * 2 + 1];

	bool normalCase = (edge1E != iS) && (edge1E != iE); // the second edge connects corner 0 and corner 2
	if (iS != prism.idxVtx[0] && iS != prism.idxVtx[1]) {	// find the index of the third corner.
		prism.idxVtx[2] = iS;
	}
	else {
		prism.idxVtx[2] = iE;
	}

	// index of third edge.
	idxEdge = idxEdges.z; // TRIANGLE_TO_EDGES_VAR[m_prismId * 3 + 1];
	prism.m_idxEdge[2] = idxEdge;
	if (!normalCase) {
		// swap m_idxEdge[1] and m_idxEdge[2]
		prism.m_idxEdge[1] ^= prism.m_idxEdge[2];
		prism.m_idxEdge[2] ^= prism.m_idxEdge[1];
		prism.m_idxEdge[1] ^= prism.m_idxEdge[2];
	}

	// load vertex info based on edge's info
	float lon[3], lat[3]; // longtitude and latitude of three corners
	float maxR = GLOBAL_RADIUS - THICKNESS * iLayer; // Radius of top triangle in current cell
	float minR = maxR - THICKNESS; // Radius of bottom triangle in current cell

	for (int i = 0; i < 3; i++) {	// for each corner index of the triangle (specified by prismId)
		int idxCorner = prism.idxVtx[i];	//TRIANGLE_TO_CORNERS_VAR[m_prismId*3+i];
		lat[i] = texelFetch(CORNERS_LAT_VAR, idxCorner - 1).r;	// CORNERS_LAT_VAR[idxCorner]
		lon[i] = texelFetch(CORNERS_LON_VAR, idxCorner - 1).r;	// CORNERS_LON_VAR[idxCorner]
		prism.vtxCoordTop[i] = dvec3(maxR*cos(lat[i])*cos(lon[i]), maxR*cos(lat[i])*sin(lon[i]), maxR*sin(lat[i]));
		prism.vtxCoordBottom[i] = dvec3(minR*cos(lat[i])*cos(lon[i]), minR*cos(lat[i])*sin(lon[i]), minR*sin(lat[i]));
	}

	return true; 
}

// Return adjacent mesh cell (which is triangle in the remeshed MPAS mesh) id 
// which shares with current triangle (specified by curTriangleId) the edge belongs to the face (denoted by faceId)
int getAdjacentCellId(inout MPASPrism prism, int faceId) {
	if (prism.m_iLayer == TOTAL_LAYERS - 2 && faceId == 1) {
		return -1; // we reached the deepest layer, no more layers beyond current layer
	} 
	if (prism.m_iLayer == 0 && faceId == 0){
		return -1;	// we reached the most top layer
	}
	if (faceId == 0 || faceId == 1) {
		return int(prism.m_prismId);	// currentTriangleId
	}

	const int faceId_to_edgeId[8] = { -1, -1, 2, 2, 1, 1, 0, 0 };
	int edgeId = FACEID_TO_EDGEID[faceId];
	int idxEdge = prism.m_idxEdge[edgeId];
	ivec2 nextTriangleIds = texelFetch(EDGE_TO_TRIANGLES_VAR, idxEdge - 1).xy;	// EDGE_TO_TRIANGLES_VAR[idxEdge * 2];
	if (nextTriangleIds.x == prism.m_prismId){ // curTriangleId
		ivec3 cellId3 = texelFetch(cellsOnVertex, nextTriangleIds.y - 1).xyz;
		if (cellId3.x == 0 || cellId3.y == 0 || cellId3.z == 0){ // on boundary
			return -1;
		}
		return nextTriangleIds.y;
	}
	ivec3 cellId3 = texelFetch(cellsOnVertex, nextTriangleIds.x - 1).xyz;
	if (cellId3.x == 0 || cellId3.y == 0 || cellId3.z == 0){ // on boundary
		return -1;
	}
	return nextTriangleIds.x;
}

int rayPrismIntersection(inout MPASPrism prism, in Ray r, inout HitRec tInRec,
	inout HitRec tOutRec, inout int nextCellId) {
	nextCellId = -1;	// assume no next prism to shot into
	int nHit = 0;
	int nFaces = 8;
	tOutRec.hitFaceid = -1; // initialize to tOutRec
	tOutRec.t = -1.0f;
	double min_t = FLOAT_MAX, max_t = -1.0f;
	dvec3 vtxCoord[6];
	vtxCoord[0] = prism.vtxCoordTop[0];
	vtxCoord[1] = prism.vtxCoordTop[1];
	vtxCoord[2] = prism.vtxCoordTop[2];
	vtxCoord[3] = prism.vtxCoordBottom[0];
	vtxCoord[4] = prism.vtxCoordBottom[1];
	vtxCoord[5] = prism.vtxCoordBottom[2];

	for (int idxFace = 0; idxFace < nFaces; idxFace++) {	// 8 faces
		dvec3 v0 = vtxCoord[d_mpas_faceCorners[idxFace * 3]];
		dvec3 v1 = vtxCoord[d_mpas_faceCorners[idxFace * 3 + 1]];
		dvec3 v2 = vtxCoord[d_mpas_faceCorners[idxFace * 3 + 2]];

		double t = 0.0;
		dvec3 rayO = dvec3(r.o);
		dvec3 rayD = dvec3(r.d);
		dvec3 vtxTB0 = dvec3(v0);
        dvec3 vtxTB1 = dvec3(v1);
        dvec3 vtxTB2 = dvec3(v2);
		bool bhit = rayIntersectsTriangleDouble(rayO, rayD,
                    vtxTB0, vtxTB1, vtxTB2, t);

		if (bhit) {
			nHit++;
			
			if (min_t > t) {
				min_t = t;
				tInRec.t = t;
				tInRec.hitFaceid = idxFace; 
			}
			if (max_t < t) {
				max_t = t;
				tOutRec.t = t;
				tOutRec.hitFaceid = idxFace;
				if (idxFace == 1) {
					tOutRec.nextlayerId = prism.m_iLayer + 1;	// the next prism to be traversed is in the lower layer 
				}
				else if (idxFace == 0) {
					tOutRec.nextlayerId = prism.m_iLayer - 1;	// the next prism to be traversed is in the upper layer
				} 
				else {
					tOutRec.nextlayerId = prism.m_iLayer;
				}
			}
		}
	}

	if (nHit == 2) {
		nextCellId = getAdjacentCellId(prism, tOutRec.hitFaceid);
	}
	else {	// specical case when ray hit on the edge 
		nextCellId = -1;
	}

	return nHit;
}

void GetAs0(inout MPASPrism prism, inout double A[12])
{
	double x1 = (prism.vtxCoordTop[0].x);
	double y1 = (prism.vtxCoordTop[0].y);
	double z1 = (prism.vtxCoordTop[0].z);

	double x2 = (prism.vtxCoordTop[1].x);
	double y2 = (prism.vtxCoordTop[1].y);
	double z2 = (prism.vtxCoordTop[1].z);

	double x3 = (prism.vtxCoordTop[2].x);
	double y3 = (prism.vtxCoordTop[2].y);
	double z3 = (prism.vtxCoordTop[2].z);

	A[1] = (+x3 * y1 - x1 * y3);
	A[2] = (+x3 * z1 - x1 * z3);
	A[3] = (+y3 * z1 - y1 * z3);
	A[4] = (-x2 * y1 + x1 * y2);
	A[5] = (-x2 * z1 + x1 * z2);
	A[6] = (-y2 * z1 + y1 * z2);
	A[7] = (-x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3);//x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3);//(-x2*y1 + x3*y1 + x1*y2 - x3*y2 - x1*y3 + x2*y3);
	A[8] = (-x2 * z1 + x3 * z1 + x1 * z2 - x3 * z2 - x1 * z3 + x2 * z3);//x3 * (z1 - z2) + x1 * (z2 - z3) + x2 * (-z1 + z3);//(-x2*z1 + x3*z1 + x1*z2 - x3*z2 - x1*z3 + x2*z3);
	A[9] = (-y2 * z1 + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3);//y3 * (z1 - z2) + y1 * (z2 - z3) + y2 * (-z1 + z3);//(-y2*z1 + y3*z1 + y1*z2 - y3*z2 - y1*z3 + y2*z3);
	A[10] = (x3*y2*z1);
	A[11] = (-x2 * y3*z1 - x3 * y1*z2 + x1 * y3*z2 + x2 * y1*z3 - x1 * y2*z3);
}

void GetBs(
    inout MPASPrism prism,
    inout double A[12], inout dvec3 fTB[2], inout double B[8],
    inout double OP1, inout double OP4)
{

    OP1 = length(prism.vtxCoordTop[0]);
    OP4 = length(prism.vtxCoordBottom[0]);
    B[0] = 1.0f / ((A[10] + A[11])*(OP1 - OP4));//BUG-->//1.0f/((A[1]+A[11])*OP1 - OP4);//
    B[1] = (A[3] + A[6] - A[9]) * fTB[0].x - A[3] * fTB[0].y - A[6] * fTB[0].z + (-A[3] - A[6] + A[9])* fTB[1].x + A[3] * fTB[1].y + A[6] * fTB[1].z;
    //(A3 + A6 - A9) V1			- A3 V2			- A6 V3			 + (-A3 - A6 + A9) V4			+ A3 V5			+ A6 V6)
    B[2] = (A[1] + A[4] - A[7]) * fTB[0].x - A[1] * fTB[0].y - A[4] * fTB[0].z + (-A[1] - A[4] + A[7])* fTB[1].x + A[1] * fTB[1].y + A[4] * fTB[1].z;
    // (A1 + A4 - A7) V1				- A1 V2			- A4 V3			 + (-A1 - A4 + A7) V4			+ A1 V5			+ A4 V6)
    B[3] = (-A[2] - A[5] + A[8])* fTB[0].x + A[2] * fTB[0].y + A[5] * fTB[0].z + (A[2] + A[5] - A[8])* fTB[1].x - A[2] * fTB[1].y - A[5] * fTB[1].z;
    // (-A2 - A5 + A8) V1			+ A2 V2			+ A5 V3			 + (A2 + A5 - A8) V4				- A2 V5			- A5 V6
    B[4] = fTB[0].x - fTB[0].y;//V1-V2
    B[5] = fTB[0].x - fTB[0].z;//V1-V3
    B[6] = fTB[1].x - fTB[1].y;//V4-V5
    B[7] = fTB[1].x - fTB[1].z;//V4-V6
}

void GetScalarValue(inout MPASPrism prism, samplerBuffer CLIMATE_VALS_VAR, inout dvec3 scalars[2]) {
	for (int iFace = 0; iFace < 2; iFace++) {
		int layerId = prism.m_iLayer + iFace;
		scalars[iFace].x = texelFetch(CLIMATE_VALS_VAR, (prism.idxVtx[0]-1) * TOTAL_LAYERS + layerId).r; //CLIMATE_VALS_VAR[idxVtx[0] * TOTAL_LAYERS + (m_iLayer + iFace)];
		scalars[iFace].y = texelFetch(CLIMATE_VALS_VAR, (prism.idxVtx[1]-1) * TOTAL_LAYERS + layerId).r; //CLIMATE_VALS_VAR[idxVtx[0] * TOTAL_LAYERS + (m_iLayer + iFace)];
		scalars[iFace].z = texelFetch(CLIMATE_VALS_VAR, (prism.idxVtx[2]-1) * TOTAL_LAYERS + layerId).r; //CLIMATE_VALS_VAR[idxVtx[0] * TOTAL_LAYERS + (m_iLayer + iFace)];
	}
}

void GetMaxLevelCell(inout MPASPrism prism, inout ivec3 maxLevel){
	maxLevel.x = texelFetch(maxLevelCell, prism.idxVtx[0]-1).r;
	maxLevel.y = texelFetch(maxLevelCell, prism.idxVtx[1]-1).r;
	maxLevel.z = texelFetch(maxLevelCell, prism.idxVtx[2]-1).r;
}

void GetUV(in dvec3 O, in dvec3 Q, inout double A[12],
	inout double u, inout double v) {
	dvec3 QO = (Q - O);//*Factor;
	double denominator = (A[9] * QO.x - A[8] * QO.y + A[7] * QO.z);
	u = (A[3] * QO.x - A[2] * QO.y + A[1] * QO.z) / denominator;
	v = (A[6] * QO.x - A[5] * QO.y + A[4] * QO.z) / denominator;
}

double GetInterpolateValue2(in MPASPrism prism, in const double u, in const double v,
	in const dvec3 Q, in dvec3 fT, in dvec3 fB) {
	dvec3 baryCoord = vec3(1.0 - u - v, u, v);
	dvec3 m1 = baryCoord.x * prism.vtxCoordTop[0] + baryCoord.y * prism.vtxCoordTop[1] + baryCoord.z * prism.vtxCoordTop[2];
	dvec3 m2 = baryCoord.x * prism.vtxCoordBottom[0] + baryCoord.y * prism.vtxCoordBottom[1] + baryCoord.z * prism.vtxCoordBottom[2];

	double scalar_m1 = dot(baryCoord, fT);
	double scalar_m2 = dot(baryCoord, fB);
	double t3 = length(Q - m2) / length(m1 - m2);
	double lerpedVal = mix(scalar_m2, scalar_m1, t3);	//lerp()
	return lerpedVal;
}

dvec3 GetInterpolateNormal2(in MPASPrism prism, in const double u, in const double v,
	in const dvec3 Q, in dvec3 nT0, in dvec3 nT1, in dvec3 nT2, 
	in dvec3 nB0, in dvec3 nB1, in dvec3 nB2){
	dvec3 baryCoord = vec3(1.0 - u - v, u, v);
	dvec3 m1 = baryCoord.x * prism.vtxCoordTop[0] + baryCoord.y * prism.vtxCoordTop[1] + baryCoord.z * prism.vtxCoordTop[2];
	dvec3 m2 = baryCoord.x * prism.vtxCoordBottom[0] + baryCoord.y * prism.vtxCoordBottom[1] + baryCoord.z * prism.vtxCoordBottom[2];
	
	dvec3 normal_m1 = baryCoord.x * nT0 + baryCoord.y * nT1 + baryCoord.z * nT2;
	dvec3 normal_m2 = baryCoord.x * nB0 + baryCoord.y * nB1 + baryCoord.z * nB2;
	double t3 = length(Q - m2) / length(m1 - m2);
	dvec3 lerpedNormal = mix(normal_m2, normal_m1, t3);
	return lerpedNormal;
}

dvec3 GetNormal(const dvec3 position, const double A[12],
	const double B[8], const double OP1, const double OP4) {
	double delx = 0.0f, dely = 0.0f, delz = 0.0f;//partial derivative of scalar value at sampling point.
	double inv_denom = 1.0f / (A[9] * position.x - A[8] * position.y + A[7] * position.z);
	double C0 = B[0] * OP1*(A[9] * position.x - A[8] * position.y + A[7] * position.z)*(B[1] * position.x + B[3] * position.y + B[2] * position.z);
	//B10 OP1 (A9 qx			- A8 qy				+ A7 qz)		 (B11 qx			+ B13 qy		 + B12 qz)
	double temp = (A[10] + A[11])*OP4 + OP1 * (A[9] * position.x - A[8] * position.y + A[7] * position.z);
	//(A10 + A11) OP4	 + OP1 (A9 qx			- A8 qy				+ A7 qz)
	double C1 = B[6] - B[0] * (B[4] - B[6])*temp;
	//B16 - B10 (B14 - B16)
	double C2 = B[7] - B[0] * (B[5] - B[7])*temp;
	//B17 - B10 (B15 - B17)

	delx = (inv_denom*inv_denom)*
		(A[9] * C0 +
			C1 * ((A[3] * A[8] - A[2] * A[9])*position.y + (-A[3] * A[7] + A[1] * A[9])*position.z) +
			C2 * ((A[6] * A[8] - A[5] * A[9])*position.y + (-A[6] * A[7] + A[4] * A[9])*position.z)
			);
	//A9 C0 +
	//		C1 ((A3 A8		- A2 A9) qy				+ (-A3 A7		+ A1 A9) qz) +
	//		C2 ((A6 A8		- A5 A9) qy				+ (-A6 A7		+ A4 A9) qz)
	dely = (inv_denom*inv_denom)*
		(-A[8] * C0 +
			C1 * ((-A[3] * A[8] + A[2] * A[9])*position.x + (A[2] * A[7] - A[1] * A[8])*position.z) +
			C2 * ((-A[6] * A[8] + A[5] * A[9])*position.x + (A[5] * A[7] - A[4] * A[8])*position.z)
			);
	//-A8 C0 +
	//		C1 ((-A3 A8			+ A2 A9) qx			+ (A2 A7		- A1 A8) qz) +
	//		C2 ((-A6 A8			+ A5 A9) qx			+ (A5 A7		- A4 A8) qz)
	delz = (inv_denom*inv_denom)*
		(A[7] * C0 +
			C1 * ((A[3] * A[7] - A[1] * A[9])*position.x + (-A[2] * A[7] + A[1] * A[8])*position.y) +
			C2 * ((A[6] * A[7] - A[4] * A[9])*position.x + (-A[5] * A[7] + A[4] * A[8])*position.y)
			);
	//A7 C0 +
	// C1 ((A3 A7		- A1 A9) qx				+ (-A2 A7		+ A1 A8) qy) +
	// C2 ((A6 A7		- A4 A9) qx				+ (-A5 A7		+ A4 A8) qy)
	return normalize(dvec3(delx, dely, delz));
}

void ComputeVerticesNormalTop(in MPASPrism prism, inout dvec3 vtxNormals[3]) {	
	//out_Color = vec4(vec3(prism.m_prismId)/15459.0, 1.0);
	// for each one of the three corners of top face of current prism
	// compute their shared normals respectively
	for (int iCorner = 0; iCorner < 3; iCorner++) {
		int idxCorner = prism.idxVtx[iCorner];	//TRIANGLE_TO_CORNERS_VAR[m_prismId*3+iCorner]
		// for each nNeighbors prisms (including current prism and nNeighbors-1 neighbor prisms)
		int nNeighbors = texelFetch(CORNER_TO_TRIANGLES_DIMSIZES, idxCorner - 1).x;
		dvec3 avgNormal = vec3(0.0f);
		int count = 1;
		for (int iPrism = 0; iPrism < nNeighbors; iPrism++) {	//weighted normal 
			int id = texelFetch(CORNER_TO_TRIANGLES_VAR, (idxCorner-1) * maxEdges + iPrism).x;	// CORNER_TO_TRIANGLES_VAR[idxCorner * nNeighbors + iPrism]

			MPASPrism curPrismHitted;	// (id, m_iLayer)
			ReloadVtxInfo(id, prism.m_iLayer, curPrismHitted);
			// find the vertex in curPrism whose global index == idxCorner
			int i = 0;
			for (; i < 3; i++) {
				if (curPrismHitted.idxVtx[i] == idxCorner)
					break;
			}
			double A[12];
			GetAs0(curPrismHitted, A);
			dvec3 fTB[2];
			GetScalarValue(curPrismHitted, temperature, fTB);
			double OP1, OP4, B[8];
			GetBs(curPrismHitted, A, fTB, B, OP1, OP4);
			avgNormal += GetNormal(curPrismHitted.vtxCoordTop[i], A, B, OP1, OP4);

			if (prism.m_iLayer > 0) {
				MPASPrism curPrismHitted1;	// (id, m_iLayer)
				ReloadVtxInfo(id, prism.m_iLayer - 1, curPrismHitted1); 
				double A[12];
				GetAs0(curPrismHitted1, A);
				dvec3 fTB[2];
				GetScalarValue(curPrismHitted1, temperature, fTB);
				double OP1, OP4, B[8];
				GetBs(curPrismHitted1, A, fTB, B, OP1, OP4);
				avgNormal += GetNormal(curPrismHitted1.vtxCoordBottom[i], A, B, OP1, OP4);
				count = 2; 
			}
		}
		vtxNormals[iCorner] = avgNormal / double(nNeighbors * count);
	}
}

void ComputeVerticesNormalBottom(in MPASPrism prism, inout dvec3 vtxNormals[3]) {  
	// for each one of the three corners of top face of current prism
	// compute their shared normals respectively
	for (int iCorner = 0; iCorner < 3; iCorner++) {
		int idxCorner = prism.idxVtx[iCorner];	//TRIANGLE_TO_CORNERS_VAR[m_prismtId*3+iCorner]
		// for each nNeighbors prisms (including current prism and nNeighbors-1 neighbor prisms)
		int nNeighbors = texelFetch(CORNER_TO_TRIANGLES_DIMSIZES, idxCorner - 1).x;
		dvec3 avgNormal = vec3(0.0f);
		int count = 1;
		for (int iPrism = 0; iPrism < nNeighbors; iPrism++) {	//weighted normal 
			int id = texelFetch(CORNER_TO_TRIANGLES_VAR, (idxCorner-1) * maxEdges + iPrism).x;	// CORNER_TO_TRIANGLES_VAR[idxCorner * nNeighbors + iPrism]
			MPASPrism curPrismHitted;	// (id, m_iLayer)
			ReloadVtxInfo(id, prism.m_iLayer, curPrismHitted);
			// find the vertex in curPrism whose global index == idxCorner
			int i = 0;
			for (; i < 3; i++) {
				if (curPrismHitted.idxVtx[i] == idxCorner)
					break;
			}
			double A[12];
			GetAs0(curPrismHitted, A);
			dvec3 fTB[2];
			GetScalarValue(curPrismHitted, temperature, fTB);
			double OP1, OP4, B[8];
			GetBs(curPrismHitted, A, fTB, B, OP1, OP4);
			avgNormal += GetNormal(curPrismHitted.vtxCoordBottom[i], A, B, OP1, OP4);

			ivec3 maxLevel;
			GetMaxLevelCell(prism, maxLevel);
			if (prism.m_iLayer < maxLevel.x - 2 && 
			prism.m_iLayer < maxLevel.y - 2 && 
			prism.m_iLayer < maxLevel.z - 2){
				MPASPrism curPrismHitted1;	// (id, m_iLayer)
				ReloadVtxInfo(id, prism.m_iLayer + 1, curPrismHitted1); 
				double A[12];
				GetAs0(curPrismHitted1, A);
				dvec3 fTB[2];
				GetScalarValue(curPrismHitted1, temperature, fTB);
				double OP1, OP4, B[8];
				GetBs(curPrismHitted1, A, fTB, B, OP1, OP4);
				avgNormal += GetNormal(curPrismHitted1.vtxCoordTop[i], A, B, OP1, OP4);
				count = 2;
			}		
		}
		vtxNormals[iCorner] = avgNormal / double(nNeighbors * count);
	}
}

void main(){
	int triangle_id = g2f.triangle_id;
	dvec3 o_eye = dvec3((uInvMVMatrix * vec4(0, 0, 0, 1.0)).xyz);
	Ray ray;
	ray.o = o_eye;
	ray.d = normalize(g2f.o_pos - o_eye);

	HitRec tInHitRecord, tOutHitRecord, tmpInRec, tmpOutRec;
	tInHitRecord.hitFaceid = -1;
	tInHitRecord.t = FLOAT_MAX;
	tInHitRecord.nextlayerId = -1;

	tOutHitRecord.hitFaceid = -1;
	tOutHitRecord.t = FLOAT_MIN;
	tOutHitRecord.nextlayerId = -1;
	
	tmpInRec.t = FLOAT_MAX;
	tmpOutRec.t = FLOAT_MIN;
	tmpInRec.nextlayerId = g2f.layer_id;
	tmpOutRec.nextlayerId = -1;
	
	int nHit = 0;
	int nextCellId = -1;
	int tmpNextCellId = -1;
	uint curPrismHittedId = triangle_id;
	
	MPASPrism curPrismHitted;
	curPrismHitted.idxVtx[0] = -1;
	curPrismHitted.idxVtx[1] = -1;
	curPrismHitted.idxVtx[2] = -1;
	ReloadVtxInfo(int(curPrismHittedId), tmpInRec.nextlayerId, curPrismHitted);
	
	int tmpNHit = rayPrismIntersection(curPrismHitted, ray, tmpInRec, tmpOutRec, tmpNextCellId);
	//gPosition = vec3(tmpOutRec.nextlayerId);

	if (tmpNHit > 0) {
		nHit = tmpNHit;
		nextCellId = tmpNextCellId;
		curPrismHittedId = triangle_id;
		tInHitRecord = (tInHitRecord.t > tmpInRec.t) ? tmpInRec : tInHitRecord; 
		tOutHitRecord = (tOutHitRecord.t < tmpOutRec.t) ? tmpOutRec : tOutHitRecord;
	}	

	dvec3 position = vec3(GLOBAL_RADIUS + GLOBAL_RADIUS);
	bool hasIsosurface = false;
	//if (g2f.hitFaceid == tInHitRecord.hitFaceid)
	{
		ivec3 maxLevel;
		GetMaxLevelCell(curPrismHitted, maxLevel);
		if (curPrismHitted.m_iLayer < maxLevel.x - 1 && 
		curPrismHitted.m_iLayer < maxLevel.y - 1 && 
		curPrismHitted.m_iLayer < maxLevel.z - 1) {
			// loop through 3d grid 
			double A[12];
			GetAs0(curPrismHitted, A);
			dvec3 fTB[2];
			GetScalarValue(curPrismHitted, temperature, fTB);
			double OP1, OP4;
            double B[8];
            GetBs(curPrismHitted, A, fTB, B, OP1, OP4);
			
			if (tInHitRecord.t < 0.0f)
				tInHitRecord.t = 0.0f;
			double t = tInHitRecord.t;	
			position = ray.o + ray.d * t;
			double u, v;
			GetUV(vec3(0.0f), position, A, u, v);
			double scalar_last = GetInterpolateValue2(curPrismHitted, u, v, position, fTB[0], fTB[1]);
		
			t = tOutHitRecord.t; 
			position = ray.o + ray.d * t;
			GetUV(vec3(0.0f), position, A, u, v);
			double scalar = GetInterpolateValue2(curPrismHitted, u, v, position, fTB[0], fTB[1]);
			// out_Color = vec4(vec3(abs(scalar_last - scalar) * 1000.0) , 1.0);
			
			if ((scalar_last - double(threshold)) * (scalar - double(threshold)) < 0) 
			//if (scalar_last - double(threshold) > 0 && scalar - double(threshold) < 0)
			{
				// compute normal on the six vertices of current prism
				dvec3 vtxFNormals[3];
				ComputeVerticesNormalTop(curPrismHitted, vtxFNormals);
				dvec3 vtxBNormals[3];
				ComputeVerticesNormalBottom(curPrismHitted, vtxBNormals);

				double offset = (scalar - double(threshold)) / (scalar - scalar_last);
				t = tOutHitRecord.t - offset * (tOutHitRecord.t - tInHitRecord.t);
				position = ray.o + ray.d * t;
				gPosition = vec3(position);
				GetUV(vec3(0.0f), position, A, u, v);
				gNormal = vec3(normalize(dmat3(uNMatrix) * GetInterpolateNormal2(curPrismHitted, u, v, position, 
												vtxFNormals[0], vtxFNormals[1], vtxFNormals[2],
												vtxBNormals[0], vtxBNormals[1], vtxBNormals[2])));
				if ((scalar_last - threshold) < 0)
					gNormal = -gNormal;
				//gNormal = vec3(vtxFNormals[0]);
				gSalinity = 1.0;
				gMask = 1.0;
				vec4 posProjSpace = uPMatrix * uMVMatrix * vec4(position, 1.0);
				gDepth = posProjSpace.z /posProjSpace.w;
				gDepth = gDepth * 0.5 + 0.5;
				//gDepth = LinearizeDepth(gDepth) / far;
				
				hasIsosurface = true;
			}
		}
	}
	if (!hasIsosurface)
		discard;
}