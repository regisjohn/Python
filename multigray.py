#!/usr/bin/env python3
#===============================================================================
"""
This is a python script for plotting binary data from F3D program. This script 
is useful for reuse if you need to read through multiple files for a single 
time slice and then loop through each of the time slices similarly.
|Author: Regis | Date Created/Last Modified: July 31, 2024/Aug 24, 2024|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from f3dfuncs import read_first_metadata, read_binary_f3d
from image_cont import image_cont
#-------------------------------------------------------------------------------
# Configuration Class for parameters and flags
#-------------------------------------------------------------------------------
class Config:
    """
    This configuration class is used to set the parameters and flags for the
    script.
    """
    # Flags:
    interactive_flag = False # Set to False to manually set runno, variables, and slices.
    save_flag = False # Set to True if you want to save the plots. Scroll below to set save parameters.

    # Run number, variables, and slices, this is used if interactive_flag = False:
    runno_ = 226 # Set to the run number.
    variables_ = 'curx,cury,curz' # Set to the variables to read.
    first_slice_ = 0 # Set to the index of the first time slice.
    last_slice_ = 5 # Set to the index of the last time slice.
    skip_slice_ = 1 # Set to the number of time slices to skip.

    # Cuts and locations:
    plane_ = 'xz' # Set to 'xy' or 'yz' or 'xz' or 'xyz'.
    plane_val_ = 0.0 # Set to the value of the plane for 2D data.
    ax1_cut_ = 0.0 # Set to the value of the first axis cut for 1D data.
    ax2_cut_ = 0.0 # Set to the value of the second axis cut for 1D data.

    # Plot Window Settings:
    rows_ = 3
    cols_ = 3
    main_figsize = (15,12)
    fig_, axes_ = plt.subplots(rows_, cols_, figsize=main_figsize, dpi=100)

    # Save Parameters:
    output_dir = "pyframes" # Directory name where you want to save the frames.
    savedpi = 100 # Higher dpi means slower runtime, default is 100
        
#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
def main():
    """
    This configuration class is used to set the parameters and flags for the
    script. 
    """ 
    
    # If save_flag is True:
    if Config.save_flag:
        # Get current backend and change it to non-interactive one for saving.
        import matplotlib
        curr_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        # If output_dir doesn't exist create it
        import os 
        if not os.path.exists(Config.output_dir): 
            os.makedirs(Config.output_dir)

    # Call the multigray function:
    if Config.interactive_flag:
        multigray(mfig=Config.fig_, axes=Config.axes_, plane=Config.plane_, 
            plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, ax2_cut=Config.ax2_cut_)
    else:
        multigray(mfig=Config.fig_, axes=Config.axes_, runno=Config.runno_, 
            variables=Config.variables_, first_slice=Config.first_slice_, 
            last_slice=Config.last_slice_, skip_slice=Config.skip_slice_, plane=Config.plane_, 
            plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, ax2_cut=Config.ax2_cut_)
        
    # Archiving the saved plots into a tar file:
    if Config.save_flag:
        matplotlib.use(curr_backend) # setting backend to default of the system after saving.
        import tarfile # for creating tar file
        out_filename = f'{Config.output_dir}.tar'
        with tarfile.open(out_filename, 'w') as tar:
            tar.add(Config.output_dir, arcname='.')
        print(f"Folder '{Config.output_dir}' has been archived into '{out_filename}'!")

    print("All Done!")
#-------------------------------------------------------------------------------
# Function To Get Binary Data and Compile into a Dictionary
#-------------------------------------------------------------------------------
def multigray(mfig, axes, runno=None, variables=None, first_slice=None, 
              last_slice=None, skip_slice=None, plane='xz', plane_val=0.0, 
              ax1_cut=0.0, ax2_cut=0.0):
    """
    Generate a multi-image plot for a given set of variables and time slices.

    Parameters:
    - mfig (Figure): The figure object to plot on.
    - axes (Axes): The axes object to plot on.
    - runno (int, optional): The run number. Defaults to None.
    - variables (list, optional): List of variables to plot. Defaults to None.
    - first_slice (int, optional): The first time slice to plot. Defaults to None.
    - last_slice (int, optional): The last time slice to plot. Defaults to None.
    - skip_slice (int, optional): The number of slices to skip between plots. Defaults to None.
    - plane (str, optional): The plane to plot on. Defaults to 'xz'.
    - plane_val (float, optional): The value of the plane. Defaults to 0.0.
    - ax1_cut (float, optional): The value to cut the first axis. Defaults to 0.0.
    - ax2_cut (float, optional): The value to cut the second axis. Defaults to 0.0.

    Returns:
    - None
    """
    # If runno, variables, and slices are not specified, ask user for them
    if runno is None:
        runno, variables, first_slice, last_slice, skip_slice, filenames, nx, ny, \
        nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, zz = interactive_mode()
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
    time_range = range(first_slice,last_slice+1,skip_slice)
    for time_slice in time_range: # Loop over time slices
        timedt_ = time_slice * dtime # code time of the slice
        axin = 0 # axis index
        print('............')
        for filename in filenames: # Loop over variables
            with open(filename, 'rb') as bfile:
                time_slice_data = read_binary_f3d(bfile, time_slice, nx, ny,
                                    nz, xx, yy, zz, x_start, x_end, y_start, 
                                    y_end, z_start, z_end, plane, plane_val, 
                                    ax1_cut, ax2_cut, metadat_len_bytes)
                
                # Printing the filename and time read
                # print(f'{filename} for the plane = {plane} and ' 
                
                # Plotting the data in 2D plane
                ax_ = axes.flat[axin] # Set the axis
                # image_cont already clears axes.
                image_cont(time_slice_data, fig=mfig, ax=ax_, timedt=timedt_)
                axin += 1
        # Printing the frame read.
        print(f'frame = {time_slice} read!')
        
        plt.tight_layout()

        # Save plot if save_flag is True, else display the plot
        if Config.save_flag:
            mfig.savefig(f'{Config.output_dir}/frame_{time_slice:04d}.png',dpi=Config.savedpi)
            print('............')
            print('Figure saved!')                
        else:
            plt.pause(0.00001) # Displaying plot interactively
            a = input('Enter 0 to continue: ')  


    plt.close()

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
    
    return runno, variables, first_slice, last_slice, skip_slice, filenames, \
        nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, zz

#-------------------------------------------------------------------------------
# Script Execution or Module Import Check
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
else:
    raise ImportError("\nHi this script is not meant to be used as a module!" 
          "\nRun the script directly as `run -m multigray` or `python -m multigray`"
          "\nFor more info refer to documentation.")
#-------------------------------------------------------------------------------
