#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define _USE_MATH_DEFINES

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform2.hpp>

#include <stdlib.h>
#include <cfloat>
#include <hdf5.h>
#include <netcdf.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <limits>
#include "def.h"
#include <assert.h>

#include "shader.h"
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void renderQuad();

const bool dump_buffer = false;
const int nSample = 50;
float isoValue;
const float view_params[nSample * 3] = {
20.314484,187.908897,171.774259,
22.705146,353.781572,128.653411,
21.823998,75.905606,75.036965,
19.282300,293.549511,50.002891,
16.090334,133.269339,123.457286,
15.315076,103.901258,9.258459,
15.245793,117.139243,43.174920,
23.406247,108.092297,0.440163,
22.155665,323.578984,12.691946,
16.941582,128.449090,69.940793,
17.346898,269.210309,11.212497,
23.474697,223.899455,6.632119,
22.838593,283.194386,114.214793,
19.875788,174.193752,48.220350,
21.449050,15.732136,94.229160,
18.716461,185.697975,122.360962,
22.719250,27.991906,56.133349,
18.471506,356.730204,31.184229,
16.325181,308.757227,73.947490,
21.390395,345.603730,141.117501,
23.298309,234.760645,156.420186,
23.650707,331.293966,150.831893,
20.779962,178.143805,149.994288,
24.376396,184.379013,83.329037,
20.083013,57.991362,29.889161,
22.448610,307.089471,105.756724,
19.237054,145.588793,40.063822,
16.410821,315.447263,25.732794,
21.235742,63.850396,82.833046,
23.021740,167.563972,38.540286,
17.658054,87.398911,129.835562,
16.865753,36.670710,124.564688,
16.500053,59.641168,174.642346,
15.370439,297.939758,110.793316,
17.465472,32.291522,12.482814,
24.350887,82.395558,172.405820,
21.936678,19.688731,50.690288,
18.120732,157.173595,96.707968,
15.964961,94.343080,80.454207,
23.322053,44.478843,58.497118,
18.023226,357.442690,106.699932,
20.564321,105.166395,155.868415,
20.185707,40.218361,76.216746,
19.659452,207.369895,91.693415,
15.296466,239.860163,49.150644,
16.223836,117.075535,90.694643,
23.770049,300.490202,116.574573,
18.522119,60.556431,96.261180,
15.141651,332.397868,117.298147,
24.260451,310.795274,34.725567
};

// settings
const unsigned int SCR_WIDTH = 512;
const unsigned int SCR_HEIGHT = 512;

size_t nCells, nEdges, nVertices, nVertLevels, maxEdges, vertexDegree, Time;
vector<double> latVertex, lonVertex, xVertex, yVertex, zVertex;
vector<double> xyzCell, latCell, lonCell; 
vector<int> indexToVertexID, indexToCellID, indexToEdgeID;
vector<int> verticesOnEdge, cellsOnEdge, 
			cellsOnVertex, edgesOnVertex, 
			verticesOnCell, nEdgesOnCell, maxLevelCell; 
vector<double> temperature, salinity;

map<int, int> vertexIndex, cellIndex;

const double max_rho = 6371229.0;
const double layerThickness = 20000.0 ;
const double eps = 1e-5;

// Base color used for the fog, and clear-to colors.
glm::vec3 base_color(0.0f / 255.0, 0.0f / 255.0, 0.0f / 255.0);

// Used for time based animation.
float time_last = 0;

// Used to rotate the isosurface.
float rotation_radians = 0.0;
float rotation_radians_step = 0.3 * 180 / M_PI;

glm::mat4 pMatrix;
glm::mat4 mvMatrix;
glm::mat4 inv_pMatrix;
glm::mat4 inv_mvMatrix;
glm::mat3 normalMatrix;

// Perspective or orthographic projection?
bool perspective_projection = true;

// Information related to the camera 
//float dist = 0.6;
float dist = 2.3;
float theta, phi;
glm::vec3 direction;
glm::vec3 up;
glm::vec3 center;
glm::vec3 eye;

glm::mat4 view;
glm::mat4 model;

// Lighting power.
float lighting_power = 0.5;

// lighting parameters 
// Use lighting?
int use_lighting = 1;

// Render different buffers.
int show_depth = 0;
int show_normals = 0;
int show_position = 0;

// Used to orbit the point lights.
float point_light_theta = M_PI / 2;
float point_light_phi = M_PI / 2;

float point_light_theta1 = 1.57;
float point_light_phi1 = 1.57;

unsigned int vao[1];
unsigned int gbo[2];

unsigned int latCellBuf, latCellTex, lonCellBuf, lonCellTex;
unsigned int latVertexBuf, latVertexTex, lonVertexBuf, lonVertexTex;
unsigned int cellsOnVertexBuf, cellsOnVertexTex;
unsigned int edgesOnVertexBuf, edgesOnVertexTex;
unsigned int nEdgesOnCellBuf, nEdgesOnCellTex;
unsigned int verticesOnCellBuf, verticesOnCellTex;
unsigned int cellsOnEdgeBuf, cellsOnEdgeTex;
unsigned int verticesOnEdgeBuf, verticesOnEdgeTex;
unsigned int maxLevelCellBuf, maxLevelCellTex;
unsigned int temperatureBuf, temperatureTex;
unsigned int salinityBuf, salinityTex;

void loadMeshFromNetCDF(const string& filename) {
	int ncid;
	int dimid_cells, dimid_edges, dimid_vertices, dimid_vertLevels, dimid_maxEdges,
		dimid_vertexDegree, dimid_Time;
	int varid_latVertex, varid_lonVertex, varid_xVertex, varid_yVertex, varid_zVertex,
		varid_latCell, varid_lonCell, varid_xCell, varid_yCell, varid_zCell,
		varid_edgesOnVertex, varid_cellsOnVertex,
		varid_indexToVertexID, varid_indexToCellID, varid_indexToEdgeID,
		varid_nEdgesOnCell, varid_cellsOncell, varid_verticesOnCell, varid_maxLevelCell,
		varid_verticesOnEdge, varid_cellsOnEdge,
		varid_temperature, varid_salinity;

	NC_SAFE_CALL(nc_open(filename.c_str(), NC_NOWRITE, &ncid));

	NC_SAFE_CALL(nc_inq_dimid(ncid, "nCells", &dimid_cells));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "nEdges", &dimid_edges));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "nVertices", &dimid_vertices));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "nVertLevels", &dimid_vertLevels));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "maxEdges", &dimid_maxEdges));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "vertexDegree", &dimid_vertexDegree));
	NC_SAFE_CALL(nc_inq_dimid(ncid, "Time", &dimid_Time));

	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_cells, &nCells));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_edges, &nEdges));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_vertices, &nVertices));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_vertLevels, &nVertLevels));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_maxEdges, &maxEdges));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_vertexDegree, &vertexDegree));
	NC_SAFE_CALL(nc_inq_dimlen(ncid, dimid_Time, &Time));

	NC_SAFE_CALL(nc_inq_varid(ncid, "indexToVertexID", &varid_indexToVertexID));
	NC_SAFE_CALL(nc_inq_varid(ncid, "indexToCellID", &varid_indexToCellID));
	NC_SAFE_CALL(nc_inq_varid(ncid, "indexToEdgeID", &varid_indexToEdgeID));
	NC_SAFE_CALL(nc_inq_varid(ncid, "latCell", &varid_latCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "lonCell", &varid_lonCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "xCell", &varid_xCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "yCell", &varid_yCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "zCell", &varid_zCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "nEdgesOnCell", &varid_nEdgesOnCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "maxLevelCell", &varid_maxLevelCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "latVertex", &varid_latVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "lonVertex", &varid_lonVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "xVertex", &varid_xVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "yVertex", &varid_yVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "zVertex", &varid_zVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "edgesOnVertex", &varid_edgesOnVertex));
	NC_SAFE_CALL(nc_inq_varid(ncid, "cellsOnVertex", &varid_cellsOnVertex));	
	NC_SAFE_CALL(nc_inq_varid(ncid, "cellsOnCell", &varid_cellsOncell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "verticesOnCell", &varid_verticesOnCell));
	NC_SAFE_CALL(nc_inq_varid(ncid, "verticesOnEdge", &varid_verticesOnEdge));
	NC_SAFE_CALL(nc_inq_varid(ncid, "cellsOnEdge", &varid_cellsOnEdge));
	NC_SAFE_CALL(nc_inq_varid(ncid, "temperature", &varid_temperature)); 
	NC_SAFE_CALL(nc_inq_varid(ncid, "salinity", &varid_salinity));

	const size_t start_cells[1] = { 0 }, size_cells[1] = { nCells };

	latCell.resize(nCells);
	lonCell.resize(nCells);
	indexToCellID.resize(nCells);
	nEdgesOnCell.resize(nCells);
	maxLevelCell.resize(nCells);

	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_latCell, start_cells, size_cells, &latCell[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_lonCell, start_cells, size_cells, &lonCell[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_indexToCellID, start_cells, size_cells, &indexToCellID[0]));
	for (int i = 0; i < nCells; i++) {
		cellIndex[indexToCellID[i]] = i;
		// fprintf(stderr, "%d, %d\n", i, indexToCellID[i]);
	}
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_nEdgesOnCell, start_cells, size_cells, &nEdgesOnCell[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_maxLevelCell, start_cells, size_cells, &maxLevelCell[0]));

	std::vector<double> coord_cells;
	coord_cells.resize(nCells);
	xyzCell.resize(nCells * 3);
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_xCell, start_cells, size_cells, &coord_cells[0]));
	for (int i = 0; i < nCells; i++)
		xyzCell[i * 3] = coord_cells[i];
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_yCell, start_cells, size_cells, &coord_cells[0]));
	for (int i = 0; i < nCells; i++)
		xyzCell[i * 3 + 1] = coord_cells[i];
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_zCell, start_cells, size_cells, &coord_cells[0]));
	for (int i = 0; i < nCells; i++)
		xyzCell[i * 3 + 2] = coord_cells[i];

	//for (int i = 0; i < nCells; i++) {
	//	double x = max_rho * cos(latCell[i]) * cos(lonCell[i]);
	//	double y = max_rho * cos(latCell[i]) * sin(lonCell[i]);
	//	double z = max_rho * sin(latCell[i]);
	//	assert(abs(x - xyzCell[i * 3]) < eps && abs(y - xyzCell[i * 3 + 1]) < eps && abs(z - xyzCell[i * 3 + 2])< eps);
	//}

	const size_t start_vertices[1] = { 0 }, size_vertices[1] = { nVertices };
	latVertex.resize(nVertices);
	lonVertex.resize(nVertices);
	xVertex.resize(nVertices);
	yVertex.resize(nVertices);
	zVertex.resize(nVertices);
	indexToVertexID.resize(nVertices);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_indexToVertexID, start_vertices, size_vertices, &indexToVertexID[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_latVertex, start_vertices, size_vertices, &latVertex[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_lonVertex, start_vertices, size_vertices, &lonVertex[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_xVertex, start_vertices, size_vertices, &xVertex[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_yVertex, start_vertices, size_vertices, &yVertex[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_zVertex, start_vertices, size_vertices, &zVertex[0]));

	//for (int i = 0; i < nVertices; i++) {
	//	double x = max_rho * cos(latVertex[i]) * cos(lonVertex[i]);
	//	double y = max_rho * cos(latVertex[i]) * sin(lonVertex[i]);
	//	double z = max_rho * sin(latVertex[i]);
	//	assert(abs(x - xVertex[i]) < eps && abs(y - yVertex[i]) < eps && abs(z - zVertex[i]) < eps);
	//}

	for (int i = 0; i < nVertices; i++) {
		vertexIndex[indexToVertexID[i]] = i;
		// fprintf(stderr, "%d, %d\n", i, indexToVertexID[i]);
	}

	const size_t start_edges[1] = { 0 }, size_edges[1] = { nEdges };
	indexToEdgeID.resize(nEdges);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_indexToEdgeID, start_edges, size_edges, &indexToEdgeID[0]));

	const size_t start_edges2[2] = { 0, 0 }, size_edges2[2] = { nEdges, 2 };
	verticesOnEdge.resize(nEdges * 2);
	cellsOnEdge.resize(nEdges * 2);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_verticesOnEdge, start_edges2, size_edges2, &verticesOnEdge[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_cellsOnEdge, start_edges2, size_edges2, &cellsOnEdge[0]));

	//for (int i=0; i<nEdges; i++) 
	//   fprintf(stderr, "%d, %d\n", verticesOnEdge[i*2], verticesOnEdge[i*2+1]);

	const size_t start_vertex_cell[2] = { 0, 0 }, size_vertex_cell[2] = { nVertices, 3 };
	cellsOnVertex.resize(nVertices * 3);
	edgesOnVertex.resize(nVertices * 3);

	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_cellsOnVertex, start_vertex_cell, size_vertex_cell, &cellsOnVertex[0]));
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_edgesOnVertex, start_vertex_cell, size_vertex_cell, &edgesOnVertex[0]));

	const size_t start_cell_vertex[2] = { 0, 0 }, size_cell_vertex[2] = { nCells, maxEdges };
	verticesOnCell.resize(nCells * maxEdges);
	
	NC_SAFE_CALL(nc_get_vara_int(ncid, varid_verticesOnCell, start_cell_vertex, size_cell_vertex, &verticesOnCell[0]));

	const size_t start_time_cell_vertLevel[3] = { 0, 0, 0 }, size_time_cell_vertLevel[3] = { Time, nCells, nVertLevels };
	temperature.resize(Time * nCells * nVertLevels);
	salinity.resize(Time * nCells * nVertLevels);

	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_temperature, start_time_cell_vertLevel, size_time_cell_vertLevel, &temperature[0]));
	NC_SAFE_CALL(nc_get_vara_double(ncid, varid_salinity, start_time_cell_vertLevel, size_time_cell_vertLevel, &salinity[0]));

	NC_SAFE_CALL(nc_close(ncid));

	fprintf(stderr, "%zu, %zu, %zu, %zu\n", nCells, nEdges, nVertices, nVertLevels);
}

void initBuffers() {
	// set up vertex data (and buffer(s)) and configure vertex attributes
	// ------------------------------------------------------------------
	glGenVertexArrays(1, vao);
	glGenBuffers(2, gbo);

	glBindVertexArray(vao[0]);

	unsigned int veridxdat = gbo[0];
	unsigned int layeriddat = gbo[1];

	glBindBuffer(GL_ARRAY_BUFFER, veridxdat);
	std::size_t size = indexToVertexID.size();
	indexToVertexID.resize(size * (nVertLevels - 1));
	vector<int>::iterator it = indexToVertexID.begin() + size;
	for (int i = 0; i < nVertLevels - 2; i++) {
		copy(indexToVertexID.begin(), it, it + size * i);
	}
	glBufferData(GL_ARRAY_BUFFER, nVertices * (nVertLevels - 1) * sizeof(int), &indexToVertexID[0], GL_STATIC_DRAW);
	glVertexAttribIPointer(0, 1, GL_INT, sizeof(int), 0);
	glEnableVertexAttribArray(0);	

	glBindBuffer(GL_ARRAY_BUFFER, layeriddat);
	vector<int> vertexLayerID;
	for (int i = 0; i < nVertLevels - 1; i++) {
		vector<int> temp;
		temp.assign(nVertices, i);
		vertexLayerID.insert(vertexLayerID.end(), temp.begin(), temp.end());
	}
	glBufferData(GL_ARRAY_BUFFER, nVertices * (nVertLevels - 1) * sizeof(int), &vertexLayerID[0], GL_STATIC_DRAW);
	glVertexAttribIPointer(1, 1, GL_INT, sizeof(int), 0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

void initTextures() {
	//// Coordinates of cells
	glGenBuffers(1, &latCellBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, latCellBuf);
	vector<float> latCellFloat(latCell.begin(), latCell.end());
	glBufferData(GL_TEXTURE_BUFFER, nCells * sizeof(float), &latCellFloat[0], GL_STATIC_DRAW);
	glGenTextures(1, &latCellTex);

	glGenBuffers(1, &lonCellBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, lonCellBuf);
	vector<float> lonCellFloat(lonCell.begin(), lonCell.end());
	glBufferData(GL_TEXTURE_BUFFER, nCells * sizeof(float), &lonCellFloat[0], GL_STATIC_DRAW);
	glGenTextures(1, &lonCellTex);

	// Coordinates of vertices 
	glGenBuffers(1, &latVertexBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, latVertexBuf);
	vector<float> latVertexFloat(latVertex.begin(), latVertex.end());
	glBufferData(GL_TEXTURE_BUFFER, nVertices * sizeof(float), &latVertexFloat[0], GL_STATIC_DRAW);
	glGenTextures(1, &latVertexTex);

	glGenBuffers(1, &lonVertexBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, lonVertexBuf);
	vector<float> lonVertexFloat(lonVertex.begin(), lonVertex.end());
	glBufferData(GL_TEXTURE_BUFFER, nVertices * sizeof(float), &lonVertexFloat[0], GL_STATIC_DRAW);
	glGenTextures(1, &lonVertexTex);

	// verticesOnCell
	glGenBuffers(1, &verticesOnCellBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, verticesOnCellBuf);
	glBufferData(GL_TEXTURE_BUFFER, nCells * maxEdges * sizeof(int), &verticesOnCell[0], GL_STATIC_DRAW);
	glGenTextures(1, &verticesOnCellTex);

	// nEdgesOnCell
	glGenBuffers(1, &nEdgesOnCellBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, nEdgesOnCellBuf);
	glBufferData(GL_TEXTURE_BUFFER, nCells * sizeof(int), &nEdgesOnCell[0], GL_STATIC_DRAW);
	glGenTextures(1, &nEdgesOnCellTex);

	// cellsOnVertex 
	glGenBuffers(1, &cellsOnVertexBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, cellsOnVertexBuf);
	glBufferData(GL_TEXTURE_BUFFER, nVertices * 3 * sizeof(int), &cellsOnVertex[0], GL_STATIC_DRAW);
	glGenTextures(1, &cellsOnVertexTex);

	// edgesOnVertex
	glGenBuffers(1, &edgesOnVertexBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, edgesOnVertexBuf);
	glBufferData(GL_TEXTURE_BUFFER, nVertices * 3 * sizeof(int), &edgesOnVertex[0], GL_STATIC_DRAW);
	glGenTextures(1, &edgesOnVertexTex);

	// cellsOnEdge
	glGenBuffers(1, &cellsOnEdgeBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, cellsOnEdgeBuf);
	glBufferData(GL_TEXTURE_BUFFER, nEdges * 2 * sizeof(int), &cellsOnEdge[0], GL_STATIC_DRAW);
	glGenTextures(1, &cellsOnEdgeTex);

	// verticesOnEdge
	glGenBuffers(1, &verticesOnEdgeBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, verticesOnEdgeBuf);
	glBufferData(GL_TEXTURE_BUFFER, nEdges * 2 * sizeof(int), &verticesOnEdge[0], GL_STATIC_DRAW);
	glGenTextures(1, &verticesOnEdgeTex);

	// maxLevelCell
	glGenBuffers(1, &maxLevelCellBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, maxLevelCellBuf);
	glBufferData(GL_TEXTURE_BUFFER, nCells * sizeof(int), &maxLevelCell[0], GL_STATIC_DRAW);
	glGenTextures(1, &maxLevelCellTex);

	// temperature 
	glGenBuffers(1, &temperatureBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, temperatureBuf);
	vector<float> temperatureFloat(temperature.begin(), temperature.end());
	glBufferData(GL_TEXTURE_BUFFER, Time * nCells * nVertLevels * sizeof(float), &temperatureFloat[0], GL_STATIC_DRAW);
	glGenTextures(1, &temperatureTex);
	
	// salinity 
	glGenBuffers(1, &salinityBuf);
	glBindBuffer(GL_TEXTURE_BUFFER, salinityBuf);
	vector<float> salinityFloat(salinity.begin(), salinity.end());
	glBufferData(GL_TEXTURE_BUFFER, Time * nCells * nVertLevels * sizeof(float), &salinityFloat[0], GL_STATIC_DRAW);
	glGenTextures(1, &salinityTex);
}

void setMatrixUniforms(Shader ourShader) {
	// Pass the vertex shader the projection matrix and the model-view matrix.
	ourShader.setMat4("uMVMatrix", mvMatrix);
	ourShader.setMat4("uPMatrix", pMatrix);

	// Pass the vertex normal matrix to the shader so it can compute the lighting calculations.
	normalMatrix = glm::transpose(glm::inverse(glm::mat3(mvMatrix)));
	ourShader.setMat3("uNMatrix", normalMatrix);
	
	// inverse of projection matrix 
	inv_pMatrix = glm::inverse(pMatrix);
	ourShader.setMat4("uInvPMatrix", inv_pMatrix);

	inv_mvMatrix = glm::inverse(mvMatrix);
	ourShader.setMat4("uInvMVMatrix", inv_mvMatrix);
	return;
}

unsigned int gBuffer;
unsigned int gPosition, gNormal, gSalinity, gMask, gDepth;
void setUpgBuffer(){
    	// set up G-buffer 
	// 5 textures
	// 1. Position (RGB)
	// 2. Color(RGBA)
	// 3. Normals 
	// 4. Masks
	// 5. Depth
	glGenFramebuffers(1, &gBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
	
	// position color buffer 
	glGenTextures(1, &gPosition);
	glBindTexture(GL_TEXTURE_2D, gPosition);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
	// normal color buffer 
	glGenTextures(1, &gNormal);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
    // salinity color buffer 
	glGenTextures(1, &gSalinity);
	glBindTexture(GL_TEXTURE_2D, gSalinity);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gSalinity, 0);
	// mask color buffer
	glGenTextures(1, &gMask);
	glBindTexture(GL_TEXTURE_2D, gMask);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, SCR_WIDTH, SCR_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, gMask, 0);
	// depth color buffer
	glGenTextures(1, &gDepth);
	glBindTexture(GL_TEXTURE_2D, gDepth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, gDepth, 0);

	// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
	unsigned int attachment[5] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4 };
	glDrawBuffers(5, attachment);
	
	// create and attach depth buffer (renderbuffer)
	unsigned int rboDepth;
	glGenRenderbuffers(1, &rboDepth);
	glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

	//finally check if framebuffer is complete
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cout << "Framebuffer not complete!" << std::endl;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

unsigned int gBufferPost;
unsigned int gPositionPost, gNormalPost, gSalinityPost, gMaskPost, gDepthPost;
void setUpgBufferPost(){
    // set up G-buffer Post
	// 5 textures
	// 1. PositionPost (RGB)
	// 2. NormalsPost 
	// 3. Salinity
	// 4. MasksPost
	// 5. DepthPost
	glGenFramebuffers(1, &gBufferPost);
	glBindFramebuffer(GL_FRAMEBUFFER, gBufferPost);
	
	// position color buffer 
	glGenTextures(1, &gPositionPost);
	glBindTexture(GL_TEXTURE_2D, gPositionPost);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPositionPost, 0);
	// normal color buffer 
	glGenTextures(1, &gNormalPost);
	glBindTexture(GL_TEXTURE_2D, gNormalPost);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormalPost, 0);
	// diffuseColor color buffer 
	glGenTextures(1, &gSalinityPost);
	glBindTexture(GL_TEXTURE_2D, gSalinityPost);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gSalinityPost, 0);
	// mask color buffer
	glGenTextures(1, &gMaskPost);
	glBindTexture(GL_TEXTURE_2D, gMaskPost);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, SCR_WIDTH, SCR_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, gMaskPost, 0);
	// depth color buffer
	glGenTextures(1, &gDepthPost);
	glBindTexture(GL_TEXTURE_2D, gDepthPost);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, gDepthPost, 0);

	// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
	unsigned int attachmentPost[5] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4 };
	glDrawBuffers(5, attachmentPost);

	//finally check if framebuffer is complete
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cout << "Framebuffer not complete!" << std::endl;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0); 
}

int main(int argc, char **argv)
{
	char input_name[1024];
	sprintf(input_name, argv[1]);
	fprintf(stderr, "%s\n", argv[1]);
    
	string input_name_s = input_name;
	int pos_last_dot = input_name_s.rfind(".");
	int pos_last_dash = input_name_s.rfind("_");
	int simIdx = stoi(input_name_s.substr(0, pos_last_dash));

	char input_path[1024];
	sprintf(input_path, "/fs/project/PAS0027/MPAS1/Results/%s", input_name);
 	loadMeshFromNetCDF(input_path);
	
// 	FILE *h5Names = NULL;
// 	FILE *imageNames = NULL;

//     //save filename 
//     char h5NamesFilename[1024];
// 	if (train)
// 	    sprintf(h5NamesFilename, "/fs/project/PAS0027/bufferLearning/data/MPAS/train_toy/%04d/h5Names.txt", simIdx);
// 	else 
// 		sprintf(h5NamesFilename, "/fs/project/PAS0027/bufferLearning/data/MPAS/test_toy/%04d/h5Names.txt", simIdx);
// 	h5Names = fopen(h5NamesFilename, "w");

//     char imageNamesFilename[1024];
// 	if (train)
// 	    sprintf(imageNamesFilename, "/fs/project/PAS0027/bufferLearning/data/MPAS/train_toy/%04d/imageNames.txt", simIdx);
// 	else 
// 		sprintf(imageNamesFilename, "/fs/project/PAS0027/bufferLearning/data/MPAS/test_toy/%04d/imageNames.txt", simIdx);
// 	imageNames = fopen(imageNamesFilename, "w");
	
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// configure global opengl state
	// -----------------------------
	glEnable(GL_DEPTH_TEST);

	// build and compile shaders
	// -------------------------
	Shader shader("../res/shaders/triangle_frustrum.vs", "../res/shaders/dvr.fs", "../res/shaders/dvr.gs");
	Shader shaderPost("../res/shaders/process_tex.vs", "../res/shaders/process_tex.fs");
	//Shader shader("triangle_center.vs", "triangle_center.fs");
	//Shader shader("hexagon_center.vs", "hexagon_center.fs");
	//Shader shader("triangle_mesh.vs", "triangle_mesh.fs", "triangle_mesh.gs");
	//Shader shader("hexagon_mesh.vs", "hexagon_mesh.fs", "hexagon_mesh_fill.gs");
	Shader shaderLightingPass("../res/shaders/deferred_shading.vs", "../res/shaders/deferred_shading.fs");

    setUpgBuffer();
	setUpgBufferPost();

	initTextures();
	initBuffers();
	
	// render loop
	// -----------
	//while (!glfwWindowShouldClose(window))
	for (int  i=0; i < nSample; i++)
	{
		float time_now = glfwGetTime();
		if (time_last != 0) {
			float time_delta = (time_now - time_last);

			rotation_radians += rotation_radians_step * time_delta;

			if (rotation_radians > 360)
				rotation_radians = 0.0;
		}

		time_last = time_now;

		// create the projection matrix 
	    float near = 1.79f;
		float far = 2.81f;
		float fov_r = 30.0f / 180.0f * M_PI;

		if (perspective_projection) {
			// Resulting perspective matrix, FOV in radians, aspect ratio, near, and far clipping plane.
			pMatrix = glm::perspective(fov_r, (float)SCR_WIDTH / (float)SCR_HEIGHT, near, far);
		}
		else {
			// The goal is to have the object be about the same size in the window
			// during orthographic project as it is during perspective projection.

			float a = (float)SCR_WIDTH / (float)SCR_HEIGHT;
			float h = 2 * (25 * tan(fov_r / 2)); // Window aspect ratio.
			float w = h * a; // Knowing the new window height size, get the new window width size based on the aspect ratio.

			// The canvas' origin is the upper left corner. To the right is the positive x-axis. 
			// Going down is the positive y-axis.

			// Any object at the world origin would appear at the upper left hand corner.
			// Shift the origin to the middle of the screen.

			// Also, invert the y-axis as WebgL's positive y-axis points up while the canvas' positive
			// y-axis points down the screen.

			//           (0,O)------------------------(w,0)
			//               |                        |
			//               |                        |
			//               |                        |
			//           (0,h)------------------------(w,h)
			//
			//  (-(w/2),(h/2))------------------------((w/2),(h/2))
			//               |                        |
			//               |         (0,0)          |gbo
			//               |                        |
			// (-(w/2),-(h/2))------------------------((w/2),-(h/2))

			// Resulting perspective matrix, left, right, bottom, top, near, and far clipping plane.
			pMatrix = glm::ortho(-(w / 2),
				(w / 2),
				-(h / 2),
				(h / 2),
				near,
				far);
		}

		// Move to the 3D space origin.
		mvMatrix = glm::mat4(1.0f);

		// transform
		theta = view_params[i * 3 + 2] / 180.0f * M_PI;
		phi = view_params[i * 3 + 1] / 180.0f * M_PI;
		//theta = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * M_PI;
    	//phi = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2 * M_PI;
    	//theta = 132.0f / 180.0f * M_PI;
    	//phi = 314.0f / 180.0f * M_PI;
    	direction = glm::vec3(sin(theta) * cos(phi) * dist, sin(theta) * sin(phi) * dist, cos(theta) * dist);
    	up = glm::vec3(sin(theta - M_PI / 2) * cos(phi), sin(theta - M_PI / 2) * sin(phi), cos(theta - M_PI / 2));
    	//isoValue = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * (25.0 - 15.0) + 15.0;
    	isoValue = view_params[i * 3 + 0];    
		
        center = glm::vec3(0.0f, 0.0f, 0.0f);
		eye = center + direction;
		
		view = glm::lookAt(eye, center, up);
		model = glm::mat4(1.0f);
		//model *= glm::rotate(rotation_radians, glm::vec3(0.0f, 1.0f, 0.0f));

		mvMatrix = view * model;
	
		// render
		// ------
		// 1. Geometry pass: render scene's geometry/color data into gbuffer
		glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
		glClearColor(base_color[0], base_color[1], base_color[2], 1.0); // Set the WebGL background color.
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// draw points
		shader.use();

		// settting uniforms such as 
		// modelview and projection matrix
		// textures that hold the lat and lon of vertices 
		// GLOBAL_RADIUS
		setMatrixUniforms(shader);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_BUFFER, latCellTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, latCellBuf); 
		shader.setInt("latCell", 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_BUFFER, lonCellTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, lonCellBuf);
		shader.setInt("lonCell", 1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_BUFFER, cellsOnVertexTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32I, cellsOnVertexBuf);
		shader.setInt("cellsOnVertex", 2);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_BUFFER, edgesOnVertexTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32I, edgesOnVertexBuf);
		shader.setInt("edgesOnVertex", 3);

		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_BUFFER, cellsOnEdgeTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32I, cellsOnEdgeBuf);
		shader.setInt("cellsOnEdge", 4);

		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_BUFFER, verticesOnEdgeTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32I, verticesOnEdgeBuf);
		shader.setInt("verticesOnEdge", 5);

		glActiveTexture(GL_TEXTURE6);
		glBindTexture(GL_TEXTURE_BUFFER, verticesOnCellTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, verticesOnCellBuf);
		shader.setInt("verticesOnCell", 6);

		glActiveTexture(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_BUFFER, nEdgesOnCellTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, nEdgesOnCellBuf);
		shader.setInt("nEdgesOnCell", 7);

		glActiveTexture(GL_TEXTURE8);
		glBindTexture(GL_TEXTURE_BUFFER, maxLevelCellTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, maxLevelCellBuf);
		shader.setInt("maxLevelCell", 8);

		glActiveTexture(GL_TEXTURE9);
		glBindTexture(GL_TEXTURE_BUFFER, temperatureTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, temperatureBuf);
		shader.setInt("temperature", 9);
		
		glActiveTexture(GL_TEXTURE10);
		glBindTexture(GL_TEXTURE_BUFFER, salinityTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, salinityBuf);
		shader.setInt("salinity", 10);
		
		shader.setFloat("GLOBAL_RADIUS", 0.5f);
		shader.setFloat("THICKNESS", 0.5f * layerThickness / max_rho);
		shader.setInt("maxEdges", maxEdges);
		shader.setInt("TOTAL_LAYERS", nVertLevels);
		shader.setFloat("threshold", isoValue);

		glBindVertexArray(vao[0]);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glDrawArrays(GL_POINTS, 0, nVertices * (nVertLevels - 1));
		//glDrawArrays(GL_POINTS, 0, 1);
		
		// 2. Post pass: post processing framebuffer, remove discontinuity
		glBindFramebuffer(GL_FRAMEBUFFER, gBufferPost);
		glClearColor(base_color[0], base_color[1], base_color[2], 1.0); // Set the WebGL background color.
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shaderPost.use();
        shaderPost.setInt("gPosition", 0);
		shaderPost.setInt("gNormal", 1);
		shaderPost.setInt("gSalinity", 2);
		shaderPost.setInt("gMask", 3);
		shaderPost.setInt("gDepth", 4);
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, gPosition);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, gNormal);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, gSalinity);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, gMask);
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, gDepth);
		
		// finally render quad
		renderQuad();

		// write framebuffer to hdf5 file
		char filename[1024];
	    sprintf(filename, "iso%d.h5", i);
		if (dump_buffer) {
			char filepath[1024];
			sprintf(filepath, "/fs/project/PAS0027/MPAS1/Results/%04d/%s", simIdx, filename);
			    
			hid_t file, space3, space1, dset_position, dset_normal, dset_mask, dset_depth, dset_salinity;
			herr_t status;
			hsize_t dims3[1] = { SCR_WIDTH * SCR_HEIGHT * 3 };
			hsize_t dims1[1] = { SCR_WIDTH * SCR_HEIGHT };

			// Create a new file using the default properties 
			file = H5Fcreate(filepath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

			// Create dataspace.  Setting maximum size to NULL sets the maximum size to be the current size.
			space3 = H5Screate_simple(1, dims3, NULL);
			space1 = H5Screate_simple(1, dims1, NULL);

			// Create the dataset. We will use all default properties for this
			dset_position = H5Dcreate(file, "position", H5T_NATIVE_FLOAT, space3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			dset_normal = H5Dcreate(file, "normal", H5T_NATIVE_FLOAT, space3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			dset_salinity = H5Dcreate(file, "salinity", H5T_NATIVE_FLOAT, space1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			dset_depth = H5Dcreate(file, "depth", H5T_NATIVE_FLOAT, space1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			dset_mask = H5Dcreate(file, "mask", H5T_NATIVE_FLOAT, space1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

			// write the data to the dataset
			float* pBuffer = new float[SCR_WIDTH * SCR_HEIGHT * 3];
			glReadBuffer(GL_COLOR_ATTACHMENT0);
			glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_FLOAT, pBuffer);
			status = H5Dwrite(dset_position, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, pBuffer);

			float* nBuffer = new float[SCR_WIDTH * SCR_HEIGHT * 3];
			glReadBuffer(GL_COLOR_ATTACHMENT1);
			glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_FLOAT, nBuffer);
			status = H5Dwrite(dset_normal, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, nBuffer);
			
			float* sBuffer = new float[SCR_WIDTH * SCR_HEIGHT];
			glReadBuffer(GL_COLOR_ATTACHMENT2);
			glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RED, GL_FLOAT, sBuffer);
			status = H5Dwrite(dset_salinity, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, sBuffer);

			float* dBuffer = new float[SCR_WIDTH * SCR_HEIGHT];
			glReadBuffer(GL_COLOR_ATTACHMENT4);
			glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RED, GL_FLOAT, dBuffer);
			status = H5Dwrite(dset_depth, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dBuffer);

			float* mBuffer = new float[SCR_WIDTH * SCR_HEIGHT];
			glReadBuffer(GL_COLOR_ATTACHMENT3);
			glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RED, GL_FLOAT, mBuffer);
			status = H5Dwrite(dset_mask, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mBuffer);

			delete pBuffer;
			delete nBuffer;
			delete sBuffer;
			delete dBuffer;
			delete mBuffer;

			status = H5Dclose(dset_position);
			status = H5Dclose(dset_normal);
			status = H5Dclose(dset_salinity);
			status = H5Dclose(dset_depth);
			status = H5Dclose(dset_mask);
			status = H5Sclose(space1);
			status = H5Sclose(space3);
			status = H5Fclose(file);
		}
		
		// 3. Lighting pass: calculate lighting by iterating over a screen filled quad pixel-by-pixel using the gbuffer's content. 
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClearColor(base_color[0], base_color[1], base_color[2], 1.0); // Set the WebGL background color.
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shaderLightingPass.use();
		shaderLightingPass.setInt("gPosition", 0);
		shaderLightingPass.setInt("gNormal", 1);
		shaderLightingPass.setInt("gSalinity", 2);
		shaderLightingPass.setInt("gMask", 3);
		shaderLightingPass.setInt("gDepth", 4);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, gPositionPost);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, gNormalPost);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, gSalinityPost);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, gMaskPost);
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, gDepthPost);

		// set lighting sources 
		// Let the fragment shader know that perspective projection is being used.
		if (perspective_projection) {
			shaderLightingPass.setInt("uPerspectiveProjection", 1);
		}
		else {
			shaderLightingPass.setInt("uPerspectiveProjection", 0);
		}

		shaderLightingPass.setInt("uShowDepth", show_depth);
		shaderLightingPass.setInt("uShowNormals", show_normals);
		shaderLightingPass.setInt("uShowPosition", show_position);
		setMatrixUniforms(shaderLightingPass);

		// Disable alpha blending.
		glDisable(GL_BLEND);
		if (use_lighting == 1) {
			// Pass the lighting parameters to the fragment shader.
			// Global ambient color. 
			shaderLightingPass.setVec3("uAmbientColor", base_color);

			// Point light 1.
			float point_light_dist = 2.3;
			glm::vec3 point_light_direction = direction / dist * point_light_dist;
			glm::vec3 point_light_position = center + point_light_direction;
			glm::vec4 light_pos(point_light_position.x, point_light_position.y, point_light_position.z, 1.0);
			light_pos = view * light_pos;
			
			shaderLightingPass.setVec3("uPointLightingColor", lighting_power, lighting_power, lighting_power);
			shaderLightingPass.setVec3("uPointLightingLocation", light_pos[0], light_pos[1], light_pos[2]);

			// Point light 2.
			//float point_light_position_x1 = 0 + 8.0 * cos(point_light_theta1) * sin(point_light_phi1);
			//float point_light_position_y1 = 0 + 8.0 * sin(point_light_theta1) * sin(point_light_phi1);
			//float point_light_position_z1 = 0 + 8.0 * cos(point_light_phi1);

			//shaderLightingPass.setVec3("uPointLightingColor1", lighting_power, lighting_power, lighting_power);
			//shaderLightingPass.setVec3("uPointLightingLocation1", point_light_position_x1, point_light_position_y1, point_light_position_z1;
		}

		shaderLightingPass.setInt("uUseLighting", use_lighting);

		// finally render quad
		renderQuad();
		
		//stbi_flip_vertically_on_write(1);
		string filename_s = filename;
		int pos_last_dot = filename_s.rfind(".");
		char imagename[1024];
		strcpy(imagename, filename_s.substr(0, pos_last_dot).c_str());
		strcat(imagename, ".png");
		char imagepath[1024];
		sprintf(imagepath, "/fs/project/PAS0027/MPAS1/Results/%04d/%s", simIdx, imagename);
		
		float* pBuffer = new float[SCR_WIDTH * SCR_HEIGHT * 4];
		unsigned char* pImage = new unsigned char[SCR_WIDTH * SCR_HEIGHT * 3];
		glReadBuffer(GL_BACK);
		glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_FLOAT, pBuffer);
		for (unsigned int j = 0; j < SCR_HEIGHT; j++) {
			for (unsigned int k = 0; k < SCR_WIDTH; k++) {
				int index = j * SCR_WIDTH + k;
				pImage[index * 3 + 0] = GLubyte(min(pBuffer[index * 4 + 0] * 255, 255.0f));
				pImage[index * 3 + 1] = GLubyte(min(pBuffer[index * 4 + 1] * 255, 255.0f));
				pImage[index * 3 + 2] = GLubyte(min(pBuffer[index * 4 + 2] * 255, 255.0f));
			}
		}
		stbi_write_png(imagepath, SCR_WIDTH, SCR_HEIGHT, 3, pImage, SCR_WIDTH * 3);
// 		fprintf(imageNames, "%04d/%s\n", simIdx, imagename);

		delete pBuffer;
		delete pImage;

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// optional: de-allocate all resources once they've outlived their purpose:
	// ------------------------------------------------------------------------
	glDeleteVertexArrays(1, vao);
	glDeleteBuffers(1, gbo);

	glfwTerminate();
	return 0;
}

// renderQuad() renders a 1x1 XY quad in NDC
// -----------------------------------------
unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad() {
	if (quadVAO == 0) {
		float quadVertices[] = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// set up plane VAO 
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	}
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}
