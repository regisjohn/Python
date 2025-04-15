#!/usr/bin/env python3
#===============================================================================
"""
This is a python script for reading bx, by and bz from F3D program and compute 
and save the bpres_xz - magnetic pressure in the xz plane, bpresaz_xz - 
azimuthal component of the magnetic pressure in the xz plane.
|Author: Regis | Date Created/Last Modified: Aug 16, 2024/Aug 24, 2024|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
from f3dfuncs import read_first_metadata, read_binary_f3d
import numexpr as ne
ne.set_num_threads(64)
#-------------------------------------------------------------------------------
# Configuration Class for parameters and flags
#-------------------------------------------------------------------------------
class Config:
    """
    This configuration class is used to set the parameters and flags for the
    script.
    """
    # Flags:
    # This program is meant to be used in non-interactive mode.
    interactive_flag = False # Set to False to manually set runno, variables, and slices.

    # Run number, variables, and slices, this is used if interactive_flag = False:
    runno_ = 20723 # Set to the run number.
    variables_ = 'bx,by,bz' # Set to the variables to read.
    first_slice_ = 0 # Set to the index of the first time slice.
    last_slice_ = 350 # Set to the index of the last time slice.
    skip_slice_ = 1 # Set to the number of time slices to skip.
    
    # Cuts and locations:
    plane_ = 'xz' # Set to 'xy' or 'yz' or 'xz' or 'xyz'.
    plane_val_ = 0.0 # Set to the value of the plane for 2D data.
    ax1_cut_ = 0.0 # Set to the value of the first axis cut for 1D data.
    ax2_cut_ = 0.0 # Set to the value of the second axis cut for 1D data.

    # Output file:
    bpres_file = f'bpres_{plane_}{int(plane_val_)}_{runno_}.npy'
    bpresaz_file = f'bpresaz_{plane_}{int(plane_val_)}_{runno_}.npy'
#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
def main():
    """
    This function handles the logical flow of the script.
    """

    print("File being used as a script!")

    # Call the multicomp_bpres function:
    if Config.interactive_flag:
        multicomp_bpres(plane=Config.plane_, plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, 
            ax2_cut=Config.ax2_cut_)
    else:
        multicomp_bpres(runno=Config.runno_, variables=Config.variables_, first_slice=Config.first_slice_, 
            last_slice=Config.last_slice_, skip_slice=Config.skip_slice_, plane=Config.plane_, 
            plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, ax2_cut=Config.ax2_cut_)
        
    print("\nAll Done!")
#-------------------------------------------------------------------------------
# Function To Get Binary Data and Compile into a Dictionary
#-------------------------------------------------------------------------------
def multicomp_bpres(runno=None, variables=None, first_slice=None, last_slice=None, 
              skip_slice=None, plane='xyz',plane_val=0.0, ax1_cut=0.0, 
              ax2_cut=0.0):
    """
    Reads binary data from F3D binary files and processes it based on the 
    provided parameters. Saves the processed data into .npy files.

    Args:
    - runno (int): The run number.
    - variables (str): The variables to read, separated by commas.
    - first_slice (int): The index of the first time slice to read.
    - last_slice (int): The index of the last time slice to read.
    - skip_slice (int): The number of time slices to skip.
    - plane (str): The plane to read ('xy', 'yz', 'xz', or 'xyz').
    - plane_val (float): The value of the plane for 2D data.
    - ax1_cut (float): The value of the first axis cut for 1D data.
    - ax2_cut (float): The value of the second axis cut for 1D data.

    Returns:
    - None
    Saves the processed data into .npy files.
    """
    # If runno, variables, and slices are not specified, ask user for them
    if runno is None: 
        runno, variables, first_slice, last_slice, skip_slice, filenames, nx,\
        ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, zz = interactive_mode()
    else:
        # Remove spaces from string and convert to list with ','
        variables = (variables.replace(' ','')).split(',') 
        # Create a list of filenames
        filenames = [f'{variables[i]}_{runno}.mov' for i in range(len(variables))]
        # Reading in metadata from the first file
        with open(filenames[0], 'rb') as mfile:
            nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, \
                zz = read_first_metadata(mfile)
        # Displaying metadata
        print(f'\nInitial metadata read from : {filenames[0]}!')
        print(f'nx={nx}, ny={ny}, nz={nz}, dx={dx}, dy={dy}, dz={dz}')
        print(f'\nTotal Times slices run from 0 to {ntimes-1}')
        print(f'\nLoading time slices from {first_slice} to {last_slice} in steps of {skip_slice}')

    # Setting limits of the data:
    x_start, x_end, y_start, y_end, z_start, z_end = 0, nx, 0, ny, 0, nz

    # Data loading code
    variables_data = [] # List to store data of all variables/filenames
    time_range = range(first_slice,last_slice+1,skip_slice)

    # Preallocating arrays:
    bpres_xz = np.empty((len(time_range), nx, nz))
    bpresaz_xz = np.empty((len(time_range), nx, nz))

    time_slice_data = [] # list to store data of all time slices
    for time_slice in time_range: # Loop over time slices
        print('............')

        # Reading bx, by, bz data:
        with open(filenames[0], 'rb') as bfile:
            bx = read_binary_f3d(bfile, time_slice, nx, ny, nz, xx, yy, zz, 
                    x_start, x_end, y_start, y_end, z_start, z_end, plane, 
                    plane_val, ax1_cut, ax2_cut, metadat_len_bytes)
            
        with open(filenames[1], 'rb') as bfile:
            by = read_binary_f3d(bfile, time_slice, nx, ny, nz, xx, yy, zz, 
                    x_start, x_end, y_start, y_end, z_start, z_end, plane, 
                    plane_val, ax1_cut, ax2_cut, metadat_len_bytes)
            
        with open(filenames[2], 'rb') as bfile:
            bz = read_binary_f3d(bfile, time_slice, nx, ny, nz, xx, yy, zz, 
                    x_start, x_end, y_start, y_end, z_start, z_end, plane, 
                    plane_val, ax1_cut, ax2_cut, metadat_len_bytes)
            
        bpres_xz[time_slice] = ne.evaluate('(bx**2 + by**2 + bz**2)/2') # total magnetic pressure
        bpresaz_xz[time_slice] = ne.evaluate('(bx**2 + bz**2)/2') # azimuthal magnetic pressure
        print(f'frame = {time_slice} read, bpres_xz and bpresaz_xz computed!')

    # Save bpress_xz and bpresaz_xz as npy files:
    np.save(Config.bpres_file, bpres_xz)
    np.save(Config.bpresaz_file, bpresaz_xz)
#-------------------------------------------------------------------------------
# Function for interactive mode
#-------------------------------------------------------------------------------
def interactive_mode():
    """
    Interactive mode function that prompts the user for input and returns a 
    variable data & metadata.

    Args:
    - None

    Returns:
    - runno (int): The run number.
    - variables (list): A list of variable names selected by the user.
    - first_slice (int): The first time slice to be loaded.
    - last_slice (int): The last time slice to be loaded.
    - skip_slice (int): The number of time slices to skip between each loaded time slice.
    - filenames (list): A list of filenames generated based on the selected variables and run number.
    - nx (int): The number of grid points in the x-direction.
    - ny (int): The number of grid points in the y-direction.
    - nz (int): The number of grid points in the z-direction.
    - dx (float): The spacing between grid points in the x-direction.
    - dy (float): The spacing between grid points in the y-direction.
    - dz (float): The spacing between grid points in the z-direction.
    - dtime (float): The time between time slices.
    - ntimes (int): The total number of time slices.
    - metadat_len_bytes (int): The length of metadata in bytes.
    - xx (numpy.ndarray): The x-coordinate array.
    - yy (numpy.ndarray): The y-coordinate array.
    - zz (numpy.ndarray): The z-coordinate array.
    """
   
    choices = {1:'bx', 2:'by', 3:'bz', 4:'curpx', 5:'curpy', 6:'curpz', 
                7:'curx', 8:'cury', 9:'curz', 10:'den', 11:'pfi', 12:'psi', 
                13:'efx', 14:'efy', 15:'efz', 16:'epz'}
    
    # Prompt user for run number
    runno = int(input("\nEnter the run number: \n"))
    # Display available variables
    print(f'\n{choices}')
    # Get variable IDs from user
    varId = input("Enter the numbers of variables that you wish to have " 
                    "loaded separated by comma like: 1,2,3\n"
                    "If you wish to load all variables, enter 99\n"
                    "To load all variables but eflds, enter 98\n"
                    "For all EMHD data, enter 93: \n")
    
    # Create variables from user input
    if varId == '99': # Load all variables
        variables = list(choices.values())
    elif varId == '98': # Load all variables except eflds
        variables = list(choices.values())[0:12]
    elif varId == '93': # Load all EMHD data
        variables = list(choices.values())[0:3] + list(choices.values())[6:9]
    else:
        varId = [int(idx.strip()) for idx in varId.split(',')] # Convert IDs to integers
        variables = [choices[idx] for idx in varId] # Create variables from IDs

    # Create filenames from arguments
    filenames = [f'{variables[i]}_{runno}.mov' for i in range(len(variables))]
        
    # Read metadata from the first file
    with open(filenames[0], 'rb') as mfile:
        nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, \
            zz = read_first_metadata(mfile)
    # Displaying metadata
    print(f'\nInitial metadata read from : {filenames[0]}!')
    print(f'nx={nx}, ny={ny}, nz={nz}, dx={dx}, dy={dy}, dz={dz}')
    print(f'\nTotal Times slices run from 0 to {ntimes-1}\n')
    
    # Prompt user for first, last, and skip times
    sliceID = input("Enter first,last, and num to skip data values " 
                    "separated by comma.\nFirst possible frame is 0 and" 
                    "put skip=1 to skip none, like = 0,10,1: \n")
    first_slice = int(sliceID.split(',')[0].strip())
    last_slice = int(sliceID.split(',')[1].strip())
    skip_slice = int(sliceID.split(',')[2].strip())
    print(f'\nLoading time slices from {first_slice} to {last_slice} in steps of {skip_slice}')
    
    return runno, variables, first_slice, last_slice, skip_slice, filenames,\
          nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, zz

#-------------------------------------------------------------------------------
# Script Execution or Module Import Check
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
else:
    print("Module multicomp_bpres imported!")
#-------------------------------------------------------------------------------
