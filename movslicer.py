#!/usr/bin/env python3
#===============================================================================
"""
This is a python script for reading binary files from F3D program and creating a
compressed hdf5 file. This is quite useful if one wants to 
transfer huge binary data to local pc. 
|Author: Regis | Date Created/Last Modified: Mar 19, 2025/Mar 20, 2025|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
from f3dfuncs import read_first_metadata, read_binary_f3d
import h5py
import hdf5plugin
#-------------------------------------------------------------------------------
# Configuration Class for parameters and flags
#-------------------------------------------------------------------------------
class Config:
    """
    This configuration class is used to set the parameters and flags for the
    script.
    """
    # Flags:
    interactive_flag = True # Set to False to manually set runno, variables, and slices.

    # Run number, variables, and slices, this is used if interactive_flag = False:
    runno_ = 226 # Set to the run number.
    variables_ = 'cury' # Set to the variables to read.
    first_slice_ = 0 # Set to the index of the first time slice.
    last_slice_ = 5 # Set to the index of the last time slice.
    skip_slice_ = 1 # Set to the number of time slices to skip.
    
    # Cuts and locations:
    plane_ = 'xyz' # Set to 'xy' or 'yz' or 'xz' or 'xyz'.
    plane_val_ = 0.0 # Set to the value of the plane for 2D data.
    ax1_cut_ = 0.0 # Set to the value of the first axis cut for 1D data.
    ax2_cut_ = 0.0 # Set to the value of the second axis cut for 1D data.  
#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
def main():
    """
    This function handles the logical flow of the script.
    """

    print("File being used as a script!")

    # Call the multiread function:
    if Config.interactive_flag:
        multiread(plane=Config.plane_, plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, 
            ax2_cut=Config.ax2_cut_)
    else:
        multiread(runno=Config.runno_, variables=Config.variables_, first_slice=Config.first_slice_, 
            last_slice=Config.last_slice_, skip_slice=Config.skip_slice_, plane=Config.plane_, 
            plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, ax2_cut=Config.ax2_cut_)
        
    print("\nAll Done!")
#-------------------------------------------------------------------------------
# Function To Get Binary Data and Compile into a Dictionary
#-------------------------------------------------------------------------------
def multiread(runno=None, variables=None, first_slice=None, last_slice=None, 
              skip_slice=None, plane='xyz',plane_val=0.0, ax1_cut=0.0, 
              ax2_cut=0.0):
    """
    Reads binary data from F3D binary files and returns a dictionary of data if 
    it is imported as a module and populates the namespace if run as a script.
    
    Args:
    - runno (int): The run number.
    - variables (str): The variables to read, separated by commas.
    - first_slice (int): The index of the first time slice to read.
    - last_slice (int): The index of the last time slice to read.
    - skip_slice (int): The number of time slices to skip between each read.
    - plane (str): The plane to read from, either 'xyz', 'xy', or 'yz'.
    - plane_val (float): The value where to take the plane cut.
    - ax1_cut (float): The value of the first axis when taking an axis cut.
    - ax2_cut (float): The value of the second axis when taking an axis cut.
        
    Returns:
    - dict: A dictionary containing the data and field parameters if `main_calling` is False.
    - populates the namespace with variables: If `main_calling` is True.
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

    # Data loading code --------------------------------------------------------
    time_range = range(first_slice,last_slice+1,skip_slice)
    
    # Create dictionary with field axes and other parameters
    grid_params = {'x':xx, 'y':yy, 'z':zz, 'nx': nx, 'ny': ny, 
            'nz': nz, 'dx': dx, 'dy': dy, 'dz': dz, 'dtime': dtime, 
            'tot_timeslices': ntimes, 'var_names': variables}
    
    shape = (nx, ny, nz); dtype = np.float64; 
    num_slices = len(time_range)
      
    for indx, filename in enumerate(filenames): # Loop over variables
        print(f'Reading {filename}....')
        with open(filename, 'rb') as bfile:
            # Create an HDF5 file
            with h5py.File(f'{variables[indx]}_{runno}.h5', "w") as hdf:
                # Add grid_params as attributes to the root group
                for key, value in grid_params.items():
                    hdf.attrs[key] = value  # Save each key-value pair as an attribute
                # Create a preallocated dataset with lzf compression
                dset = hdf.create_dataset(
                    "timeslices", shape=(num_slices, *shape), maxshape=(num_slices, *shape),
                    dtype=dtype, chunks=(1, *shape), 
                    compression=hdf5plugin.Zstd(9))
                for time_slice in time_range: # Loop over time slices
                    time_slice_data = read_binary_f3d(bfile, time_slice, nx, ny,
                                        nz, xx, yy, zz, x_start, x_end, y_start,
                                        y_end, z_start, z_end, plane, plane_val, 
                                        ax1_cut, ax2_cut, metadat_len_bytes)
                    # Printing the frame read
                    print(f'frame = {time_slice} read!') 
                    # Write data
                    dset[time_slice] = time_slice_data
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
    print("Module multiread imported!")
#-------------------------------------------------------------------------------
