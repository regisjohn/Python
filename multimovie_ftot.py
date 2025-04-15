#!/usr/bin/env python3
#===============================================================================
"""
This is a python script for creating a series of image_cont plots of total force
fx and fz in the xz plane for y=-24, 0 and 24. Here fx, fz are the vector sum 
of magnetic tension and pressure forces. They are saved as .png files for all 
time slices and then written into a movie using imageio library. One can 
zoom-in and also set custom color bar limits to the plots. 
|Author: Regis | Date Created/Last Modified: Feb 12, 2025/Feb 17, 2025|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os; import matplotlib
from f3dfuncs import read_first_metadata, read_binary_f3d
from image_cont import image_cont
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from tqdm import tqdm
import imageio.v3 as imageio
import glob
#-------------------------------------------------------------------------------
# Configuration Class for parameters and flags
#-------------------------------------------------------------------------------
class Config:
    """
    This configuration class is used to set the parameters and flags for the
    script.
    """
    # Flags:
    save_flag = bool(int(input("\nSave plots? (1 for Yes, 0 for No): "))) # Any non-zero value is True
    xz_zoom_flag = True # Set to True if you want to xz-zoom in on the plot.
    cbar_global_flag = False # Set to True if you want to set global max/min values for the colorbar.


    # Cuts and locations:
    plane_ = 'xz' # Set to 'xy' or 'yz' or 'xz' or 'xyz'.
    plane_val_ = [-24.0, 0.0, 24.0] # All 3 y-locations.
    ax1_cut_ = 0.0 # Set to the value of the first axis cut for 1D data.
    ax2_cut_ = 0.0 # Set to the value of the second axis cut for 1D data.

    # Axes labels and cut label:
    ax_label1 = 'x'; ax_label2 = 'z'
    plane_label = 'x,z'
    cut_label = 'y'

    # (x,z) zoom in limits:
    xmin, xmax = -5, 5
    zmin, zmax = -5, 5

    # Colorbar parameters:
    colmp = 'viridis' # color map

    # Save Parameters:
    output_dir_pre = f'pyframes' # Prefix for output directory
    savedpi = 300 # Higher dpi means slower runtime, default is 100
    if cbar_global_flag:
        output_dir_pre = f'pyframesg' # Prefix for output directory if cbar_global_flag = True
    fps = 30
        
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
    multimovie_ftot()
        
    # Setting backend to default of the system after saving:
    if Config.save_flag:
        matplotlib.use(curr_backend) # setting backend to default of the system after saving.

    print("\nAll Done!")
#-------------------------------------------------------------------------------
# Function To Get Binary Data
#-------------------------------------------------------------------------------
def multimovie_ftot():

    runno, first_slice, last_slice, skip_slice, filenames, nx, ny, nz, dx, dy, \
        dz, dtime, ntimes, metadat_len_bytes, xx, yy, zz = interactive_mode()

    
    # (x,z) zoom in limits indices computation:
    if Config.xz_zoom_flag:
        x_in = np.where((xx >= Config.xmin) & (xx <= Config.xmax))[0] # x indices for the zoomed-in region
        z_in = np.where((zz >= Config.zmin) & (zz <= Config.zmax))[0]
        x_start, x_end = x_in[0], x_in[-1]+1 # x start and end indices of the zoomed-in region
        y_start, y_end = 0, ny # these indices are to be passed to the vars for slicing them.
        z_start, z_end = z_in[0], z_in[-1]+1
        xx = xx[x_in]; zz = zz[z_in]
    else:
        x_start, x_end, y_start, y_end, z_start, z_end = 0, nx, 0, ny, 0, nz

    # Function to compute total force ftot for all time slices:-----------------  
    def ftot_compute(plane_ind):

        # Loading the files:
        '''Filenames are stored as a list of lists, each lists contains the 
        6 force components for a single plane - 3 bpresf and 3 btenf in this 
        order. So to access bpresfx we need to use [plane_index][0]'''

        bpresfx = np.load(filenames[plane_ind][0], mmap_mode='r')\
                    [first_slice:last_slice+1:skip_slice, x_start:x_end, z_start:z_end]
        # bpresfy = np.load(filenames[plane_ind][1], mmap_mode='r')
        bpresfz = np.load(filenames[plane_ind][2], mmap_mode='r')\
                    [first_slice:last_slice+1:skip_slice, x_start:x_end, z_start:z_end]

        btenfx = np.load(filenames[plane_ind][3], mmap_mode='r')\
                    [first_slice:last_slice+1:skip_slice, x_start:x_end, z_start:z_end]
        # btenfy = np.load(filenames[plane_ind][4], mmap_mode='r')
        btenfz = np.load(filenames[plane_ind][5], mmap_mode='r')\
                    [first_slice:last_slice+1:skip_slice, x_start:x_end, z_start:z_end]

        # Pre-allocate smoothed arrays:
        bshape = bpresfx.shape
        bpresfx_s = np.empty(bshape)
        # bpresfy_s = np.empty(bshape)
        bpresfz_s = np.empty(bshape)

        btenfx_s = np.empty(bshape)
        # btenfy_s = np.empty(bshape)
        btenfz_s = np.empty(bshape)

        # Smoothing the data:
        bpresfx_s = gaussian_filter(bpresfx, sigma=2)
        # bpresfy_s = gaussian_filter(bpresfy, sigma=2)

        # Smoothing the data:
        bpresfx_s = gaussian_filter(bpresfx, sigma=2)
        # bpresfy_s = gaussian_filter(bpresfy, sigma=2)
        bpresfz_s = gaussian_filter(bpresfz, sigma=2)

        btenfx_s = gaussian_filter(btenfx, sigma=2)
        # btenfy_s = gaussian_filter(btenfy, sigma=2)
        btenfz_s = gaussian_filter(btenfz, sigma=2)

        # Compute ftot:
        ftot_x = bpresfx_s + btenfx_s
        # ftot_y = bpresfy_s + btenfy_s
        ftot_z = bpresfz_s + btenfz_s

        return ftot_x, ftot_z


    # Global colorbar limits or local colorbar limits
    if Config.cbar_global_flag:
        # Global min/max for fx
        v_min_fx_l = np.min(ftot_x_l); v_max_fx_l = np.max(ftot_x_l)
        v_min_fx_m = np.min(ftot_x_m); v_max_fx_m = np.max(ftot_x_m)
        v_min_fx_u = np.min(ftot_x_u); v_max_fx_u = np.max(ftot_x_u)
        # Global min/max for fz
        v_min_fz_l = np.min(ftot_z_l); v_max_fz_l = np.max(ftot_z_l)
        v_min_fz_m = np.min(ftot_z_m); v_max_fz_m = np.max(ftot_z_m)
        v_min_fz_u = np.min(ftot_z_u); v_max_fz_u = np.max(ftot_z_u)
    else:
        v_max_fx_l = None; v_min_fx_l = None
        v_max_fx_m = None; v_min_fx_m = None
        v_max_fx_u = None; v_min_fx_u = None
        
        v_max_fz_l = None; v_min_fz_l = None
        v_max_fz_m = None; v_min_fz_m = None
        v_max_fz_u = None; v_min_fz_u = None

    # Function to set figure size while maintaining aspect ratio----------------
    def set_2dfig_size(fig, arr, base_sz = 6.4, cbar_space = 1.2):
        # base_sz 6.4 in or divisible by 16 for best video codec compatibility
        # cbar_space of 1.2 accounts for 20% colorbar space width
        aspect_ratio = (arr[0].shape[1] / arr[0].shape[0])*cbar_space  # width/height
        # Set figure size
        fig.set_size_inches(base_sz * aspect_ratio, base_sz)


    # Function for plotting & saving .npy files---------------------------------
    def plot_slice_npy(arr, title_l, time_slice, v_min, v_max, output_dir):

        timedt_ = time_slice * dtime # code time of the slice

        title_full = f'{title_l}, time : {timedt_:.3f}'

        image_cont(arr, xx, zz, fig=fig, ax=axes, xlabel=Config.ax_label1, 
            ylabel=Config.ax_label2, title=title_full, colmp=Config.colmp, 
            vmin_= v_min , vmax_= v_max)

        # Save plot if save_flag is True, else display the plot
        if Config.save_flag:
            plt.tight_layout()
            fig.savefig(f'{output_dir}/frame_{time_slice:04d}.png',dpi=Config.savedpi)
            plt.close()
        else:
            plt.tight_layout()
            plt.pause(0.00001) # Displaying plot interactively
            a = input('Enter 0 to continue: ')

    
    # Function to read Jy(x,z) for all planes and all time slices---------------
    def read_cury_xz(filename):
        plane = Config.plane_

        for p_idx, plane_val in enumerate(Config.plane_val_):
            with open(filename, 'rb') as bfile:
                # Looping over time slices of Jy(x,z)
                for t_idx, time_slice in enumerate(tqdm(time_range, 
                    desc=f'Processing time slices of {filename} for y = {plane_val}...')):
                    cury_data[p_idx, t_idx] = read_binary_f3d(bfile, time_slice, nx, ny,
                                                nz, xx, yy, zz, x_start, x_end, y_start,
                                                y_end, z_start, z_end, plane, plane_val, 
                                                Config.ax1_cut_, Config.ax2_cut_, 
                                                metadat_len_bytes)

        return cury_data

    # Function for plotting quiver plots overlayed with Jy image plot-----------
    def plot_slice_quiver(arr, arr1, arr2, title_lq, time_slice, v_min, v_max, output_dirq):

        title_full = f'{title_lq}, time : {timedt_:.3f}'

        if hasattr(axes, "cbar_im"):
            axes.cbar_im.remove()

        if hasattr(axes, "cbar_quiv"):
            axes.cbar_quiv.remove()
        
        axes.clear()

        img = axes.imshow(arr.T, extent=[xx[0], xx[-1], zz[0], zz[-1]],
                        origin='lower', cmap=Config.colmp, vmin=v_min, vmax=v_max)
        axes.set_title(f'{title_full}', fontsize=16)
        axes.set_xlabel(f'{Config.ax_label1}',fontsize=12); axes.set_ylabel(f'{Config.ax_label2}',fontsize=12)
        height, width = len(zz), len(xx+2)
        axes.cbar_im = fig.colorbar(img,ax=axes,fraction=0.0385*height/width, pad=0.04, 
                           format='%.2f')
        axes.cbar_im.set_label('Jy (x,z)', fontsize=12)  # Add label with fontsize 
        axes.tick_params(axis='both', which='major', labelsize=12)
        axes.set_aspect('equal')
        
        
        # Add quiver plot of Fx and Fz in xz plane
        stride = 10  # Adjust this to control arrow density
        arr1T = arr1[::stride, ::stride].T # row - major reading of fortran array
        arr2T = arr2[::stride, ::stride].T
        mag = np.sqrt(arr1T**2 + arr2T**2)
        quiv = axes.quiver(xx[::stride], zz[::stride], arr1T, arr2T, mag, cmap='binary',scale=300)
        axes.cbar_quiv = fig.colorbar(quiv, ax=axes, fraction=0.042*height/width, 
                    pad=0.12, format='%.2f', location='left')
        axes.cbar_quiv.set_label('F(x,z)', fontsize=12)  # Add label with fontsize 


        # Save plot if save_flag is True, else display the plot
        if Config.save_flag:
            plt.tight_layout()
            fig.savefig(f'{output_dirq}/frame_{time_slice:04d}.png',
                    dpi=Config.savedpi)
            plt.close()
        else:
            plt.tight_layout()
            plt.pause(0.00001) # Displaying plot interactively
            a = input('Enter 0 to continue: ')

    # Function for converting to video------------------------------------------
    def convert_to_vid(folder_loc):
        png_files = sorted(glob.glob(f'{folder_loc}/*.png')) # sort & read all .png files
        imframes = [imageio.imread(file) for file in png_files] 
        output_dir_vid = f'{output_dir_pref}/frame_vid' 
        os.makedirs(output_dir_vid, exist_ok=True) # create output folder
        vid_name = os.path.basename(folder_loc) # extract video name 
        imageio.imwrite(f'{output_dir_vid}/{vid_name}.mp4', imframes, fps=Config.fps)

# ------------------------------------------------------------------------------
# Executing the functions
# ------------------------------------------------------------------------------   
    plane_val_len = len(Config.plane_val_)
    plane_rng = range(plane_val_len)
    # Computing forces for all differnt xz planes
    with joblib_progress("Computing forces for different xz planes...", total=plane_val_len):
        force_res = Parallel(n_jobs=plane_val_len)(delayed(ftot_compute)(plane_ind) 
                    for plane_ind in plane_rng)

    ftot_x_l, ftot_z_l = force_res[0]
    ftot_x_m, ftot_z_m = force_res[1]
    ftot_x_u, ftot_z_u = force_res[2]

    # # Reading in time-array:
    timedt = np.load(f'time_{runno}.npy')

    # Initialization for plotting time slices
    time_range = range(first_slice, last_slice+1, skip_slice)
    time_len = len(time_range)
    arrays = [ftot_x_l, ftot_x_m, ftot_x_u, ftot_z_l, ftot_z_m, ftot_z_u]
    title_label = ['fx at y = -24', 'fx at y = 0', 'fx at y = 24', 'fz at y = -24', 
                    'fz at y = 0', 'fz at y = 24']
    v_max = [v_max_fx_l, v_max_fx_m, v_max_fx_u, v_max_fz_l, v_max_fz_m, v_max_fz_u]
    v_min = [v_min_fx_l, v_min_fx_m, v_min_fx_u, v_min_fz_l, v_min_fz_m, v_min_fz_u]
    sav_label = ['fx_xz_-24', 'fx_xz_0', 'fx_xz_24', 'fz_xz_-24', 'fz_xz_0', 'fz_xz_24']
    output_dir_pref = f'{Config.output_dir_pre}{runno}/bf_frames' # Savepath prefix
    output_dirs = [f'{output_dir_pref}/{label}' for label in sav_label]

    # fig, axes = plt.subplots(1, 1) # Use a single figure object

    # Looping over each array and plotting total forces for all time slices
    for indx in range(len(arrays)):
        arr = arrays[indx]
        title_l = title_label[indx]
        output_dir = output_dirs[indx]
        v_max_ = v_max[indx]
        v_min_ = v_min[indx]

        # Code for setting appropriate figure size maintaining aspect ratio
        set_2dfig_size(fig, arr)

        # Parallel looping over time slices with a progress bar
        print(f'\nPlotting {title_l}...')
        if Config.save_flag:
            os.makedirs(output_dir, exist_ok=True)
            with joblib_progress("Processing time slices...", total=time_len):
                Parallel(n_jobs=4)(delayed(plot_slice_npy)(arr[time_slice], title_l, 
                        time_slice, v_min_, v_max_, output_dir) for time_slice in time_range)
        else:
            for time_slice in time_range:
                plot_slice_npy(arr[time_slice], title_l, time_slice, v_min_, v_max_, output_dir)  
    
    plt.close('all')

    # Looping over time slices & plotting Jy(x,z) overlayed with quiver plots
    print()
    cury_data = np.empty((plane_val_len, time_len, len(xx), len(zz))) # preallocating array
    filename = f'cury_{runno}.mov'
    cury_data = read_cury_xz(filename) # Reading Jy(x,z) for all planes and all time slices
    title_lb_q = ['F(x,z) on Jy(x,z) at y = -24', 'F(x,z) on Jy(x,z) at y = 0', 
            'F(x,z) on Jy(x,z) at y = 24']
    sav_label_q = ['F_xz_-24', 'F_xz_0', 'F_xz_24']
    output_dirs_q = [f'{output_dir_pref}/{label}' for label in sav_label_q]

    fig, axes = plt.subplots(1, 1) # Use a single figure object
    for indx in plane_rng: # Looping over all planes
        arr = cury_data[indx,:,:,:]     # Jy(x,z) for all time slices
        arr1 = arrays[indx]     # fx for all times slices 
        arr2 = arrays[indx+3]   # fz for all times slices
        title_lq = title_lb_q[indx]
        output_dir_q = output_dirs_q[indx]
        v_min_ = None; v_max_ = None

        # Code for setting appropriate figure size maintaining aspect ratio
        set_2dfig_size(fig, arr, cbar_space=1.4)

        # Parallel looping over time slices with a progress bar
        print(f'\nPlotting {title_lq}...')
        if Config.save_flag:
            os.makedirs(output_dir_q, exist_ok=True)
            with joblib_progress("Processing time slices...", total=time_len):
                Parallel(n_jobs=4)(delayed(plot_slice_quiver)(arr[time_slice], arr1[time_slice], 
                  arr2[time_slice], title_lq, time_slice, v_min_, v_max_, output_dir_q) 
                  for time_slice in time_range)         

        for i, time_slice in enumerate(time_range):
                timedt_ = timedt[time_slice]
                plot_slice_quiver(arr[i], arr1[i], 
                  arr2[i], title_lq, time_slice, v_min_, v_max_, output_dir_q)

    plt.close('all')

    # Converting to video
    vid_flag = bool(int(input("\nConvert to video? (1 for Yes, 0 for No): ")))
    if vid_flag:
        for output_dir in tqdm(output_dirs, desc='Converting fx & fz to video...'):
            convert_to_vid(output_dir)
        for output_dir in tqdm(output_dirs_q, desc='Converting F(x,z) quiver to video...'):
            convert_to_vid(output_dir)



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
    # print(f'\n{choices}')
    # Get variable IDs from user
    # varId = input("Enter the numbers of variables that you wish to have " 
    #                 "loaded separated by comma like: 1,2,3\n"
    #                 "If you wish to load all variables, enter 99\n"
    #                 "To load all variables but eflds, enter 98\n"
    #                 "For all EMHD data, enter 93: \n"
    #                 "For all EMHD data plus Magnetic pressure, enter 94: \n")
    
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

    variables = ['bpresfx','bpresfy','bpresfz','btenfx','btenfy','btenfz']
    print('\nReading in bpresfx, bpresfy, bpresfz, btenfx, btenfy, btenfz, cury...')
    filenames = [[f'{var}_xz{int(val)}_{runno}.npy' for var in variables] 
                    for val in Config.plane_val_]

    # Create filenames from arguments
    # mov_vars = [var for var in variables if choices_ext[var] == '.mov']
    # npy_vars = [var for var in variables if choices_ext[var] == '.npy']
    # filenames = [f'{var}_{runno}{choices_ext[var]}' for var in mov_vars]
    # der_filenms = [f'{var}_{Config.plane_}{int(Config.plane_val_)}_{runno}{choices_ext[var]}' 
    #                for var in npy_vars]

    # Read metadata from the first .mov file:
    metadat_file = f'cury_{runno}.mov'
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
    
    return runno, first_slice, last_slice, skip_slice, filenames, \
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
