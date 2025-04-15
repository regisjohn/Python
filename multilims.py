#!/usr/bin/env python3
#===============================================================================
"""
This is a python script for computing the min/max values of any variables from 
the F3D program that is in binary format (.mov) or numpy format (.npy). The 
interactive_mode() in this script is modified to include other derived 
variables such as magnetic pressure. This is quite useful for setting the 
limits of a colormap when comparing all time slices. This script also makes use 
of joblib for parallelization of the function process_file().
|Author: Regis | Date Created/Last Modified: Aug 6, 2023/Sep 18, 2024|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
import pickle
from f3dfuncs import read_first_metadata, read_binary_f3d
from joblib import Parallel, delayed
from tqdm import tqdm
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
    variables_ = 'curx,cury,curz' # Set to the variables to read.
    first_slice_ = 0 # Set to the index of the first time slice.
    last_slice_ = 10 # Set to the index of the last time slice.
    skip_slice_ = 1 # Set to the number of time slices to skip.
    
    # Cuts and locations:
    plane_ = 'xz' # Set to 'xy' or 'yz' or 'xz'.
    plane_val_ = -24.0 # Set to the value of the plane for 2D data.
    ax1_cut_ = 0.0 # Set to the value of the first axis cut for 1D data.
    ax2_cut_ = 0.0 # Set to the value of the second axis cut for 1D data.  

    # Output file:
    out_file = f'vars_{plane_}{int(plane_val_)}_lims.pkl'
#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
def main():
    """
    This function handles the logical flow of the script.
    """

    print("File being used as a script!")
        
    # Call the multilim function:
    if Config.interactive_flag:
        vars_minmax = multilims(plane=Config.plane_, plane_val=Config.plane_val_, 
                    ax1_cut=Config.ax1_cut_, ax2_cut=Config.ax2_cut_)
    else:
        vars_minmax = multilims(runno=Config.runno_, variables=Config.variables_, 
                first_slice=Config.first_slice_, last_slice=Config.last_slice_, 
                skip_slice=Config.skip_slice_, plane=Config.plane_, 
                plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, 
                ax2_cut=Config.ax2_cut_)
    
    # Save the var_minmax dictionary to a pickle file:
    with open(Config.out_file, 'wb') as f:
        pickle.dump(vars_minmax, f)

    print("\nAll Done!")
#-------------------------------------------------------------------------------
# Function To Get Binary Data and Compile into a Dictionary
#-------------------------------------------------------------------------------
def multilims(runno=None, variables=None, first_slice=None, last_slice=None, 
              skip_slice=None, plane='xyz',plane_val=0.0, ax1_cut=0.0, ax2_cut=0.0):
    """
    Reads binary data from F3D binary files and returns a dictionary containing 
    the minimum and maximum values of specified variables for a cut for all 
    time slices.

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
    - dict: A dictionary containing the minimum and maximum values of the specified variables 
    """
    # If runno, variables, and slices are not specified, ask user for them
    if runno is None: 
        runno, first_slice, last_slice, skip_slice, filenames, der_filenms, nx,\
        ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, zz = interactive_mode()
    else:
        der_filenms = []
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

    vars_minmax = {} # Dictionary to store global min/max of all variables/filenames

    #Function for parallel processing-------------------------------------------
    def process_file(filename, count):
        """
        Reads binary data & finds the min/max values for each time slice, 
        storing these values in a dictionary.

        Parameters:
        - filename (str): The name of the file to process.
        - count (int): The index of the file in the list of files.

        Returns:
        - dict: A dictionary containing the min/max values.
        """
        vars_minmax_temp = {}
        # Data loading code
        with open(filename, 'rb') as bfile:
            frame_minmax = [] # list to store data of all time slices
            min_val = []
            max_val = []
            for time_slice in tqdm(time_range, desc=f'Processing {filename} time slices...', 
                                   position=count):
                # Appending for multiple time slices
                time_dat = read_binary_f3d(bfile, time_slice, nx, ny, nz, xx, 
                                yy, zz, x_start, x_end, y_start, y_end, 
                                z_start, z_end, plane, plane_val, ax1_cut, 
                                ax2_cut, metadat_len_bytes)
                # Finding min and max values for each time slice
                min_val.append(time_dat.min())
                max_val.append(time_dat.max())
                # print(f'frame = {time_slice} read!')

            global_min_val = min(min_val)
            global_max_val = max(max_val)
            frame_minmax.append(global_min_val)
            frame_minmax.append(global_max_val)

            # Associate the list of min/max values with the filename
            vars_minmax_temp[filename.replace('.mov','')] = frame_minmax
            # print(f'Filename = {filename} read and min/max values stored!')

        return vars_minmax_temp
    # End function for parallel processing--------------------------------------

    # Data loading code
    time_range = range(first_slice,last_slice+1,skip_slice)

    # Reading in .mov files:
    if filenames:
        vars_minmax_temp = Parallel(n_jobs=len(filenames))(delayed(process_file)
                                (filename, count) for count,filename in enumerate(filenames))
        print() # - add empty lines after tqdm calls.

        for vars_minmax_temp in vars_minmax_temp:
            vars_minmax.update(vars_minmax_temp)
        
    # Reading in bpres_xz and bpresaz data:
    if der_filenms: 
        vars_minmaxder_temp = {}
        for der_var in der_filenms: # No parallelization, since .npy files are read at once.
            der_data_temp = np.load(der_var) 
            min_der_data_temp = der_data_temp.min()
            max_der_data_temp = der_data_temp.max()
            der_name_parts = der_var.split('_')
            der_name_last_part = der_name_parts[-1].split('.')[0]
            der_name = f'{der_name_parts[0]}_{der_name_last_part}'
            vars_minmaxder_temp[der_name] = [min_der_data_temp, max_der_data_temp]

        vars_minmax.update(vars_minmaxder_temp) # Update the vars_minmax dictionary
    
    return vars_minmax
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
                13:'efx', 14:'efy', 15:'efz', 16:'epz', 17:'bpres', 18:'bpresaz',
                19:'bpresfx', 20:'bpresfy', 21:'bpresfz', 22:'btenfx', 
                23:'btenfy', 24: 'btenfz'}
    
    choices_ext = {'bx':'.mov', 'by':'.mov', 'bz':'.mov', 'curpx':'.mov', 
                  'curpy':'.mov', 'curpz':'.mov', 'curx':'.mov', 'cury':'.mov', 
                  'curz':'.mov', 'den':'.mov', 'pfi':'.mov', 'psi':'.mov', 
                  'efx':'.mov', 'efy':'.mov', 'efz':'.mov', 'epz':'.mov', 
                  'bpres':'.npy', 'bpresaz':'.npy', 'bpresfx':'.npy', 'bpresfy':'.npy',
                  'bpresfz':'.npy', 'btenfx':'.npy', 'btenfy':'.npy', 'btenfz':'.npy'}
    
    # Prompt user for run number
    runno = int(input("\nEnter the run number: \n"))
    # Display available variables
    print(f'\n{choices}')
    # Get variable IDs from user
    varId = input("Enter the numbers of variables that you wish to have " 
                    "loaded separated by space like: 1 2 3\n"
                    "If you wish to load all variables, enter 99\n"
                    "To load all variables but eflds, enter 98\n"
                    "For all EMHD data, enter 93: \n"
                    "For all EMHD data plus Magnetic pressure and forces enter 94: \n")
    
    # Create variables from user input
    if varId == '99': # Load all variables
        variables = [*choices.values()]
    elif varId == '98': # Load all variables except eflds
        variables = [*choices.values()][0:12]
    elif varId == '93': # Load all EMHD data
        variables = [*choices.values()][0:3] + list(choices.values())[6:9]
    elif varId == '94': # Load all EMHD data plus Magnetic pressure
        variables = [*choices.values()][0:3] + [*choices.values()][6:9] + [*choices.values()][16:24]
    else:
        varId = [int(idx.strip()) for idx in varId.split(',')] # Convert IDs to integers
        variables = [choices[idx] for idx in varId] # Create variables from IDs

    # Create filenames from arguments
    mov_vars = [var for var in variables if choices_ext[var] == '.mov']
    npy_vars = [var for var in variables if choices_ext[var] == '.npy']
    filenames = [f'{var}_{runno}{choices_ext[var]}' for var in mov_vars]
    der_filenms = [f'{var}_{Config.plane_}{int(Config.plane_val_)}_{runno}{choices_ext[var]}' 
                   for var in npy_vars]

    # Read metadata from the first .mov file:
    if filenames:
        metadat_file = filenames[0]
        with open(metadat_file, 'rb') as mfile:
            nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, \
                zz = read_first_metadata(mfile)
    else:
        variables_= [*choices.values()][0]
        metadat_file = f'{variables_}_{runno}.mov'
        with open(metadat_file, 'rb') as mfile:
            nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, \
                zz = read_first_metadata(mfile)

    # Displaying metadata
    print(f'\nInitial metadata read from : {metadat_file}!')
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
    
    return runno, first_slice, last_slice, skip_slice, filenames, der_filenms, \
        nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, zz
#-------------------------------------------------------------------------------
# Script Execution or Module Import Check
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
else:
    raise ImportError("\nHi this script is not meant to be used as a module!" 
          "\nRun the script directly as `run -m multilim` or `python -m multilim`"
          "\nFor more info refer to documentation.")
#-------------------------------------------------------------------------------
