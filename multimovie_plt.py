#!/usr/bin/env python3
#===============================================================================
"""
This is a python script for creating a series of line plots of a variable 
as .png files for all time slices from an IDL save file. 
|Author: Regis | Date Created/Last Modified: Sept 2, 2024/Sept 13, 2024|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os; import matplotlib
from f3dfuncs import read_first_metadata#, read_binary_f3d
# from image_cont import image_cont
# import pickle
# from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.io import readsav
#-------------------------------------------------------------------------------
# Configuration Class for parameters and flags
#-------------------------------------------------------------------------------
class Config:
    """
    This configuration class is used to set the parameters and flags for the
    script.
    """
    # Flags:
    # interactive_flag = True # Set to False to manually set runno, variables, and slices.
    save_flag = False # Set to True if you want to save the plots. Scroll below to set save parameters.
    zoom_flag = False # Set to True if you want to zoom in on the plot.
    cbar_global_flag = False # Set to True if you want to set global max/min values for the colorbar.

    # Run number, variables, and slices, this is used if interactive_flag = False:
    runno_ = 226 # Set to the run number, this will be the prefix of the save folder.
    mov_vars_ = 'curx, cury, curz' # Variables with a .mov extension.
    npy_vars_ = '' # Variables with a .npy extension.
    first_slice_ = 0 # Set to the index of the first time slice.
    last_slice_ = 10 # Set to the index of the last time slice.
    skip_slice_ = 1 # Set to the number of time slices to skip.

    # Cuts and locations:
    plane_ = 'xz' # Set to 'xy' or 'yz' or 'xz' or 'xyz'.
    plane_val_ = 0.0 # Set to the value of the plane for 2D data.
    ax1_cut_ = 0.0 # Set to the value of the first axis cut for 1D data.
    ax2_cut_ = 0.0 # Set to the value of the second axis cut for 1D data.

    # Axes labels and cut label:
    ax_label1 = 'x'; ax_label2 = 'z'
    plane_label = 'x,z'
    cut_label = 'y'

    # Zoom in limits:
    xmin, xmax = -5, 5
    ymin, ymax = -5, 5
    zmin, zmax = -5, 5

    # Colorbar parameters:
    colmp = 'inferno' # color map

    # Plot Window Settings:
    rows = 1
    cols = 1
    main_figsize = (8,6)
    fig, axes = plt.subplots(rows, cols, figsize=main_figsize, dpi=100)

    # Save Parameters:
    output_dir_pre = f'pyframes' # Prefix for output directory
    savedpi = 300 # Higher dpi means slower runtime, default is 100
    if cbar_global_flag:
        output_dir_pre = f'pyframesg' # Prefix for output directory if cbar_global_flag = True
        
#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
def main():
    """
    This configuration class is used to set the parameters and flags for the
    script. 
    """ 
    # If Save Flag is True, set backend for saving.
    if Config.save_flag:
        # Get current backend and change it to non-interactive one for saving.
        curr_backend = matplotlib.get_backend()
        print(f"Current Backend: {curr_backend}")
        matplotlib.use('Agg')

    # Call the multimovie function:
    multimovie(plane=Config.plane_, plane_val=Config.plane_val_, 
                ax1_cut=Config.ax1_cut_, ax2_cut=Config.ax2_cut_)
        
    # Setting backend to default of the system after saving:
    if Config.save_flag:
        matplotlib.use(curr_backend) # setting backend to default of the system after saving.

    print("\nAll Done!")
#-------------------------------------------------------------------------------
# Function To Get Binary Data
#-------------------------------------------------------------------------------
def multimovie(runno=None, mov_vars=None, first_slice=None, 
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
        runno, first_slice, last_slice, skip_slice, plt_vars, plt_filenms, \
            nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, zz = interactive_mode()
    else:
        # Remove spaces from string and convert to list with ','
        mov_vars = (mov_vars.replace(' ','')).split(',') 
        # Create a list of filenames
        filenames = [f'{mov_vars[i]}_{runno}.mov' for i in range(len(mov_vars))]
        # Reading in metadata from the first file
        with open(filenames[0], 'rb') as mfile:
            nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, \
                zz = read_first_metadata(mfile)
        # Displaying metadata
        print(f'\nInitial metadata read from : {filenames[0]}!')
        print(f'nx={nx}, ny={ny}, nz={nz}, dx={dx}, dy={dy}, dz={dz}')
        print(f'\nTotal Times slices run from 0 to {ntimes-1}')
        print(f'\nLoading time slices from {first_slice} to {last_slice} in steps of {skip_slice}')

    
    # Zoom in limits computation:
    if Config.zoom_flag:
        x_in = np.where((xx >= Config.xmin) & (xx <= Config.xmax))[0] # x indices for the zoomed-in region
        y_in = np.where((yy >= Config.ymin) & (yy <= Config.ymax))[0]
        z_in = np.where((zz >= Config.zmin) & (zz <= Config.zmax))[0]
        x_start, x_end = x_in[0], x_in[-1]+1 # x start and end indices of the zoomed-in region
        y_start, y_end = y_in[0], y_in[-1]+1 # these indices are to be passed to the vars for slicing them.
        z_start, z_end = z_in[0], z_in[-1]+1
        xx = xx[x_in]; yy = yy[y_in]; zz = zz[z_in]
    else:
        x_start, x_end, y_start, y_end, z_start, z_end = 0, nx, 0, ny, 0, nz

    # # If colorbar flag is True, load the colorbar limits:
    # v_min, v_max = [None]*len(mov_vars), [None]*len(mov_vars)
    # dv_min, dv_max = [None]*len(npy_vars), [None]*len(npy_vars)
    # if Config.cbar_global_flag:
    #     colmp_file = f'vars_{plane}{int(plane_val)}_lims.pkl' # Colormap file
    #     with open(colmp_file, 'rb') as f:
    #         vars_minmax = pickle.load(f)
    #     print(f'Colorbar limits loaded from {colmp_file}!')
    #     v_min = []; v_max = [] # Max/Min values of .mov variables
    #     for vars in mov_vars:
    #         minval, maxval = vars_minmax[f'{vars}_{runno}']
    #         v_min.append(minval); v_max.append(maxval)
    #     dv_min = []; dv_max = [] # Max/Min values of .npy variables
    #     for vars in npy_vars:
    #         minval, maxval = vars_minmax[f'{vars}_{runno}']
    #         dv_min.append(minval); dv_max.append(maxval)

    output_dir_pref = f'{Config.output_dir_pre}{runno}' # Savepath prefix
        
    # Reading jyc data:
    jyc = readsav(plt_filenms) 
    # Python reads in row-major order similar to C opposite to IDL, so transpose 
    jyc = jyc['jyc'].T  
    print('jyc data read!')

    # Data loading code
    time_range = range(first_slice,last_slice+1,skip_slice)

    for time_slice in tqdm(time_range, desc=f'Processing {plt_filenms} time slices...'):
        timedt_ = time_slice*dtime
        title = f'jyc.mov at t = {timedt_:.3f} s'
        ax = Config.axes; fig = Config.fig
        ax.clear() # To prevent overplotting in a loop
        ax.plot(yy,jyc[:,time_slice], 'k', linewidth=1.5)
        ax.set_xlim(yy[0],yy[-1]+1)
        ax.set_xlabel('Distance along flux rope', fontsize=12)
        ax.set_ylabel('Jy', fontsize=12)
        ax.set_title(title, fontsize=16)
        # Save plot if save_flag is True, else display the plot
        if Config.save_flag:
            output_dir = f'{output_dir_pref}/{plt_vars}_frames/{plane}_{int(plane_val)}'
            os.makedirs(output_dir, exist_ok=True)
            Config.fig.savefig(f'{output_dir}/frame_{time_slice:04d}.png',
                    dpi=Config.savedpi, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.pause(0.00001) # Displaying plot interactively
            a = input('Enter 0 to continue: ') 
    
    print() # - add empty lines after tqdm calls.  
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
                13:'efx', 14:'efy', 15:'efz', 16:'epz', 17:'bpres', 18:'bpresaz'}
    
    choices_ext = {'bx':'.mov', 'by':'.mov', 'bz':'.mov', 'curpx':'.mov', 
                  'curpy':'.mov', 'curpz':'.mov', 'curx':'.mov', 'cury':'.mov', 
                  'curz':'.mov', 'den':'.mov', 'pfi':'.mov', 'psi':'.mov', 
                  'efx':'.mov', 'efy':'.mov', 'efz':'.mov', 'epz':'.mov', 
                  'bpres':'.npy', 'bpresaz':'.npy'}
    
    # Prompt user for run number
    runno = int(input("\nEnter the run number: \n"))
    # Display available variables
    print(f'\n{choices}')
    # # Get variable IDs from user
    # varId = input("Enter the numbers of variables that you wish to have " 
    #                 "loaded separated by space like: 1 2 3\n"
    #                 "If you wish to load all variables, enter 99\n"
    #                 "To load all variables but eflds, enter 98\n"
    #                 "For all EMHD data, enter 93: \n"
    #                 "For all EMHD data plus Magnetic pressure, enter 94: \n")
    print("\nReading in jyc.mov file...")
    
    # # Create variables from user input
    # if varId == '99': # Load all variables
    #     variables = list(choices.values())
    # elif varId == '98': # Load all variables except eflds
    #     variables = list(choices.values())[0:12]
    # elif varId == '93': # Load all EMHD data
    #     variables = list(choices.values())[0:3] + list(choices.values())[6:9]
    # elif varId == '94': # Load all EMHD data plus Magnetic pressure
    #     variables = [*choices.values()][0:3] + [*choices.values()][6:9] + [*choices.values()][16:18]
    # else:
    #     varId = [int(idx.strip()) for idx in varId.split(',')] # Convert IDs to integers
    #     variables = [choices[idx] for idx in varId] # Create variables from IDs

    # Create filenames from arguments
    plt_vars = 'jyc'
    plt_filenms = f'{plt_vars}_{runno}.mov'
    filenames = []
    # mov_vars = [var for var in variables if choices_ext[var] == '.mov']
    # npy_vars = [var for var in variables if choices_ext[var] == '.npy']
    # filenames = [f'{var}_{runno}{choices_ext[var]}' for var in mov_vars]
    # der_filenms = [f'{var}_{Config.plane_}{int(Config.plane_val_)}_{runno}{choices_ext[var]}' 
    #                for var in npy_vars]

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
    
    return runno, first_slice, last_slice, skip_slice, plt_vars, plt_filenms, \
        nx, ny, nz, dx, dy, dz, dtime, ntimes, metadat_len_bytes, xx, yy, zz

#-------------------------------------------------------------------------------
# Script Execution or Module Import Check
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
else:
    raise ImportError("\nHi this script is not meant to be used as a module!" 
          "\nRun the script directly as `run -m multimovie` or `python -m multimovie`"
          "\nFor more info refer to documentation.")
#-------------------------------------------------------------------------------
