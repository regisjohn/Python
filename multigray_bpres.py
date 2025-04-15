#!/usr/bin/env python3
#===============================================================================
"""
This is a python script for plotting bpres_xz, bpresaz_xz, cury and b(x,z) 
fields in different orientation of F3D data in a 2x3 grid.
|Author: Regis | Date Created/Last Modified: Aug 16, 2024/Aug 24, 2024|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.io import readsav
import numexpr as ne
import pickle
from f3dfuncs import read_first_metadata, read_binary_f3d
from image_cont_loop import image_cont
from numpy_utils import findin
ne.set_num_threads(64) # Set numexpr to use as many threads as possible
#-------------------------------------------------------------------------------
# Configuration Class for parameters and flags
#-------------------------------------------------------------------------------
class Config:
    """
    This configuration class is used to set the parameters and flags for the
    script.
    """
# Flags:
    interactive_flag = False # Don't run this script in interactive mode.
    save_flag = False # Set to True if you want to save the figure.
    zoom_flag = True # Set to True if you want to zoom in on the data.

    # Run number, variables, and slices, this is used if interactive_flag = False:
    runno_ = 20723 # Set to the run number.
    variables_ = 'bx,by,bz,cury' # Set to the variables to read.
    first_slice_ = 0 # Set to the index of the first time slice.
    last_slice_ = 350 # Set to the index of the last time slice.
    skip_slice_ = 50 # Set to the number of time slices to skip.
    
    # Cuts and locations:
    plane_ = 'xz' # Set to 'xy' or 'yz' or 'xz'.
    plane_val_ = 24.0 # Set to the value of the plane for 2D data.
    ax1_cut_ = 0.0 # Set to the value of the first axis cut for 1D data.
    ax2_cut_ = 0.0 # Set to the value of the second axis cut for 1D data.

    # Zoom in limits:
    xmin, xmax = -5, 5
    zmin, zmax = -5, 5

    # Plot Window Settings:
    rows_ = 2
    cols_ = 3
    main_figsize = (15,12)
    fig_, axes_ = plt.subplots(rows_, cols_, figsize=main_figsize, dpi=100)
    fig_.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.30, hspace=0.25)

    # Colorbar parameters:
    colmp_ = 'inferno' # color map
    colmp_file = f'globallims_{plane_}{int(plane_val_)}_{runno_}.pkl' # Colormap file
    with open(colmp_file, 'rb') as f:
        vars_minmax = pickle.load(f)
    print(f'Colorbar limits loaded from {colmp_file}!')

    # Magnetic Pressure files:
    bpres_file = f'bpres_{plane_}{int(plane_val_)}_{runno_}.npy'
    bpresaz_file = f'bpresaz_{plane_}{int(plane_val_)}_{runno_}.npy'

    # Save Parameters:
    output_dir = f"pyfps_bpres_y{plane_val_}" # Directory name where you want to save the frames.
    savedpi = 100 # Higher dpi means slower runtime, default is 100
#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
def main():
    """
    This function handles the logical flow of the script.
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

    # Call the multigray_bpres function:
    if Config.interactive_flag:
        multigray_bpres(mfig=Config.fig_, axes=Config.axes_, plane=Config.plane_, 
            plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, ax2_cut=Config.ax2_cut_)
    else:
        multigray_bpres(mfig=Config.fig_, axes=Config.axes_, runno=Config.runno_, 
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
def multigray_bpres(mfig, axes, runno=None, variables=None, 
    first_slice=None, last_slice=None, skip_slice=None, plane='xz', 
    plane_val=0.0, ax1_cut=0.0, ax2_cut=0.0):
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
    
    # Zoom in limits computation:
    x_in = np.where((xx >= Config.xmin) & (xx <= Config.xmax))[0] # x indices for the zoomed-in region
    z_in = np.where((zz >= Config.zmin) & (zz <= Config.zmax))[0]
    x_start, x_end = x_in[0], x_in[-1]+1 # x start and end indices of the zoomed-in region
    z_start, z_end = z_in[0], z_in[-1]+1
    x_zoom = xx[x_in]; z_zoom = zz[z_in] # new x and z axes for the zoomed-in region

    # Color map inputs:
    # bx_max, bx_min = Config.vars_minmax[variables[0]]
    # by_max, by_min = Config.vars_minmax[variables[1]]
    # bz_max, bz_min = Config.vars_minmax[variables[2]] # not used for now 
    cury_xz_min, cury_xz_max = Config.vars_minmax['cury_xz']
    cury_yz_min, cury_yz_max = Config.vars_minmax['cury_yz']
    bpres_min, bpres_max = Config.vars_minmax['bpres']
    bpres_az_min, bpres_az_max = Config.vars_minmax['bpres_az']
    bxz_mag_min, bxz_mag_max = Config.vars_minmax['bxz_mag']

    # Reading jyc data:
    jycfile = f'jyc_{runno}.mov' # This is an IDL save file
    jyc = readsav('jyc_20723.mov') 
    # Python reads in row-major order similar to C opposite to IDL, so transpose 
    jyc = jyc['jyc'].T  
    print('jyc data read!')

    # Reading in bpres_xz and bpresaz data:
    bpres_xz = np.load(Config.bpres_file)
    bpresaz_xz = np.load(Config.bpresaz_file)
    print('bpres and bpresaz data read!')

    # Setting limits of the data:
    x_start, x_end, y_start, y_end, z_start, z_end = 0, nx, 0, ny, 0, nz

    # Data loading code
    time_range = range(first_slice,last_slice+1,skip_slice)
    for time_slice in time_range: # Loop over time slices
        timedt_ = time_slice * dtime # code time of the slice
        axin = 0 # axis index
        print('............')

        # Reading bx data:
        with open(filenames[0], 'rb') as bfile:
            bx = read_binary_f3d(bfile, time_slice, nx, ny, nz, xx, yy, zz,
                    x_start, x_end, y_start, y_end, z_start, z_end, plane, 
                    plane_val, ax1_cut, ax2_cut, metadat_len_bytes)

        # Reading by data:
        with open(filenames[1], 'rb') as bfile:
            by = read_binary_f3d(bfile, time_slice, nx, ny, nz, xx, yy, zz,
                    x_start, x_end, y_start, y_end, z_start, z_end, plane, 
                    plane_val, ax1_cut, ax2_cut, metadat_len_bytes)
        print('file read!')

        # Reading bz data:
        with open(filenames[2], 'rb') as bfile:
            bz = read_binary_f3d(bfile, time_slice, nx, ny, nz, xx, yy, zz,
                    x_start, x_end, y_start, y_end, z_start, z_end, plane, 
                    plane_val, ax1_cut, ax2_cut, metadat_len_bytes)
        print('file read!')

        # Reading 3D cury data:
        with open(filenames[3], 'rb') as bfile:
            cury = read_binary_f3d(bfile, time_slice, nx, ny, nz, xx, yy, zz, 
                    x_start, x_end, y_start, y_end, z_start, z_end, 'xyz', 
                    plane_val, ax1_cut, ax2_cut, metadat_len_bytes)
        print('file read!')

        # Slicing bx, by, bz:
        bx_sliced = bx[x_start:x_end, z_start:z_end]
        by_sliced = by[x_start:x_end, z_start:z_end]
        bz_sliced = bz[x_start:x_end, z_start:z_end]

        # # Calculating magnetic pressure:
        # bpres = ne.evaluate('(bx_sliced**2 + by_sliced**2 + bz_sliced**2)/2') # total magnetic pressure
        # bpres_az = ne.evaluate('(bx_sliced**2 + bz_sliced**2)/2') # azimuthal magnetic pressure
        # print('bpres and bpres_az calculated!')

        # Calculating bxz magnitude:
        bxz_mag = np.hypot(bx_sliced, bz_sliced)
        print('bxz_mag calculated!')

        # Slicing cury in the x-z plane:
        ycut_ind = findin(yy, plane_val)
        cury_xz = cury[x_start:x_end, ycut_ind, z_start:z_end]
        # for y-z cut of cury is always at x = 0 and is not sliced
        xcut_ind = findin(xx, 0)
        cury_yz = cury[xcut_ind, :, :]

        # Settting figure title:
        mfig.suptitle(f'x-z cuts at y = {plane_val}, time: {timedt_:.3f}', fontsize=16)

        # Subplot 1: cury(x,z) at y=plane_val
        ax_ = axes.flat[axin] # Set the axis
        xlabel_ = 'x'; ylabel_ = 'z'
        # image_cont already clears axes.
        image_cont(cury_xz, x_zoom, z_zoom, fig=mfig, ax=ax_, xlabel=xlabel_, 
                    ylabel=ylabel_, title='cury(x,z)', colmp=Config.colmp_, 
                    vmin_=cury_xz_min, vmax_=cury_xz_max)
        axin += 1

        # Subplot 2: cury(y,z) at x=plane_val
        ax_ = axes.flat[axin] # Set the axis
        xlabel_ = 'y'; ylabel_ = 'z'
        image_cont(cury_yz, yy, zz, fig=mfig, ax=ax_, xlabel=xlabel_, 
                    ylabel=ylabel_, title='cury(y,z) at x = 0.0', colmp=Config.colmp_, 
                    vmin_=cury_yz_min, vmax_=cury_yz_max)
        ax_.axvline(x=plane_val, color='white', linestyle='--', linewidth=2.5)
        axin += 1

        # Subplot 3: jyc(x,z)
        ax_ = axes.flat[axin] # Set the axis
        ax_.clear() # To prevent overplotting in a loop
        ax_.plot(yy,jyc[:,time_slice], 'k', linewidth=1.5)
        ax_.set_xlim(yy[0],yy[-1]+1)
        ax_.set_xlabel('Distance along flux rope')
        ax_.set_ylabel('Jy')
        ax_.set_title('jyc at x = z = 0')
        axin += 1

        # Subplot 4: Magnetic Pressure bpres(x,z)
        ax_ = axes.flat[axin] # Set the axis
        xlabel_ = 'x'; ylabel_ = 'z'
        image_cont(bpres_xz[time_slice], x_zoom, z_zoom, fig=mfig, ax=ax_, xlabel=xlabel_, 
                    ylabel=ylabel_, title='bpres(x,z)', colmp=Config.colmp_, 
                    vmin_=bpres_min, vmax_=bpres_max)
        axin += 1

        # Subplot 5: Azimuthal Magnetic Pressure bpres_az(x,z)
        ax_ = axes.flat[axin] # Set the axis
        xlabel_ = 'x'; ylabel_ = 'z'
        image_cont(bpresaz_xz[time_slice], x_zoom, z_zoom, fig=mfig, ax=ax_, xlabel=xlabel_, 
                    ylabel=ylabel_, title='bpres_az(x,z)', colmp=Config.colmp_, 
                    vmin_=bpres_az_min, vmax_=bpres_az_max)
        axin += 1

        # Subplot 6: Magnetic field lines in x,z plane
        ax_ = axes.flat[axin] # Set the axis
        if hasattr(ax_, "strmbar"):
            ax_.strmbar.remove()
        ax_.clear()
        xlabel_ = 'x'; ylabel_ = 'z'
        norm_ = Normalize(bxz_mag_min, bxz_mag_max)
        X, Z = np.meshgrid(x_zoom, z_zoom)
        bx_sliced_T = bx_sliced.T; bz_sliced_T = bz_sliced.T
        strm = ax_.streamplot(X, Z, bx_sliced_T, bz_sliced_T, color=bxz_mag, cmap=Config.colmp_,
                              norm=norm_)
        height, width = len(z_zoom), len(x_zoom)
        ax_.strmbar = mfig.colorbar(strm.lines, fraction=0.047*height/width, 
                            pad=0.04, format='%.2f')
        ax_.set_xlabel(xlabel_)
        # ax_.set_xlabel(f'x\nt = {timedt_:.5f}')
        ax_.set_ylabel(ylabel_)
        ax_.set_title('B(x,z)')
        ax_.set_aspect('equal')
        axin += 1

        # Printing the frame read.
        print(f'frame = {time_slice} read!')

        # Save plot if save_flag is True, else display the plot
        if Config.save_flag:
            mfig.savefig(f'{Config.output_dir}/frame_{time_slice:04d}.png',dpi=Config.savedpi)
            print(f'Figure saved!')                
        else:
            plt.pause(0.00001) # Displaying plot interactively
            # a = input('Enter 0 to continue: ')    

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
                    "separated by space.\nFirst possible frame is 0 and" 
                    "put skip=1 to skip none, like = 0 10 1: \n")
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
          "\nRun the script directly as `run -m multigray_bpres` or `python -m multigray_bpres`"
          "\nFor more info refer to documentation.")
#-------------------------------------------------------------------------------
