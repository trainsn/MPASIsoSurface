# MPASIsoSurface
isosurface renderer by ray casting for MPAS geodesic cells 

The way to run it on OSC:

First, load the library needed.
``` 
module load intel/16.0.3
module load cmake
module load netcdf/4.3.3.1
module load pnetcdf/1.7.0
module load hdf5/1.8.17
DISPLAY=:0
``` 

Then, complile the source code and run it.
``` 
cd build 
cmake ..
make 
./MPASVis raw_simulation_file_path {0,1}(training or testing data)
``` 

