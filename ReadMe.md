#===============================================================================
"""
This is a readme/log file for all python programs in the F3D project.
"""
#===============================================================================
#-------------------------------------------------------------------------------
1. numpy_utils.py
This file contains all the numpy helper and utility functions used in the 
F3D project.
#-------------------------------------------------------------------------------
2. f3dfuncs.py
This file contains all the base functions used in reading binary data from the 
.mov files in the F3D project.
#-------------------------------------------------------------------------------
3. multiread.py
This script is used to read binary data from .mov files and either populates 
the namespace if it is run as a script or returns a dictionary containing the 
data if it is run as a function.
#-------------------------------------------------------------------------------
4. image_cont.py
This script is used to read binary data from .mov files and plot the image data. 
It uses imshow() to generate the heatmap.
#-------------------------------------------------------------------------------
5. multigray.py
This script is used to read binary data from .mov files and plot the image data 
in a grid subplot format.
#-------------------------------------------------------------------------------
6. image_stream.py
This script is used to read binary data from .mov files and plot the 
streamlines of the vector fields.
#-------------------------------------------------------------------------------
4. multicomp_bpres.py
This is a python script for reading bx, by and bz from F3D program and compute 
and save the bpres_xz - magnetic pressure in the xz plane, bpresaz_xz - 
azimuthal component of the magnetic pressure in the xz plane.
#-------------------------------------------------------------------------------
5. multilims.py
This is a python script for computing the min/max values of any variables from 
the F3D program that is in binary format (.mov) or numpy format (.npy).
#-------------------------------------------------------------------------------
6. multicomp_bmag.py
This is a python script for reading bx and bz from F3D program and compute 
and save the b(x,z) magnitude - magnetic field in the xz plane.
#-------------------------------------------------------------------------------
7. mulitmovie.py
This is a python script for creating a series of image_cont plots of a variable 
as .png files for all time slices from a binary F3D file or a .npy file.
#-------------------------------------------------------------------------------
8. multimovie_strm.py
This is a python script for creating a series of image_stream plots of two 
vectors as .png files for all time slices from a binary F3D file or a .npy file. 
#-------------------------------------------------------------------------------
9. multimovie_plt.py
This is a python script for creating a series of line plots of a variable 
as .png files for all time slices from an IDL save file. 
#-------------------------------------------------------------------------------
10. efieldcomp.py
This is a module that reads from the F3D binary file for J and B fields and 
computes the corresponding electric field.
#-------------------------------------------------------------------------------