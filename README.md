# Double Diffusive Convection Examples

This repository contains very simple Dedalus scripts to demonstrate the fingering ("thermohaline") and overstable modes of double diffusive convection. 

The equations are based on those given in 

* [Traxler, Garaud, and Stellmach ApJ (2011)](https://ui.adsabs.harvard.edu/#abs/2011ApJ...728L..29T/abstract)
* [Mirouh, Garaud, Stellmach, Traxler, and Wood ApJ (2012)](https://ui.adsabs.harvard.edu/#abs/2012ApJ...750...61M/abstract)

## Running the Examples
Once you have [Dedalus](http://dedalus-project.org) installed and activated, you can simply change to either the ```fingering_convection``` or ```ODDC``` directory and do 

1. mpirun -np 4 python3 oddc.py
2. python3 merge.py snapshots/
3. mpirun -np 4 python3 plot_2d_series.py snapshots/snapshots_s*h5

Then ```cd frames/```, and you will find a series of .png images which can be made into a movie.
