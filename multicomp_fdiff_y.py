#!/usr/bin/env python3
#===============================================================================
"""
This is a python script for reading bx, by and bz from F3D program and compute 
n save the compoennts of total force bfz and bfx, the ratio of bfz to bfx, 
separation 'd' between flux ropes, and the square root calculation along each 
y-value or xz plane. The magnetic pressure force bpresf and tension force btenf 
components are calculated in the xz plane and added together to get the total 
force bf components for all y-values.
|Author: Regis | Date Created/Last Modified: Jan 14, 2025/Jan 24, 2025|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
from f3dfuncs import read_first_metadata, read_binary_f3d
from image_cont import image_cont
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter
import os
#-------------------------------------------------------------------------------
# Configuration Class for parameters and flags
#-------------------------------------------------------------------------------
class Config:
    """
    This configuration class is used to set the parameters and flags for the
    script.
    """
    # Flags:
    # This script is recommended to be used in interactive mode.
    # interactive_flag = True # Set to False to manually set runno, variables, and slices.

    # Run number, variables, and slices, this is used if interactive_flag = False:
    runno_ = 20723 # Set to the run number.
    variables_ = 'bx,by,bz' # Set to the variables to read.
    first_slice_ = 0 # Set to the index of the first time slice.
    last_slice_ = 350 # Set to the index of the last time slice.
    skip_slice_ = 1 # Set to the number of time slices to skip.
    
    # Cuts and locations:
    plane_ = 'xyz' # Set to 'xy' or 'yz' or 'xz' or 'xyz'.
    plane_val_ = 0.0 # Set to the value of the plane for 2D data, keep it 0.0 
    # for this script.

    ax1_cut_ = 0.0 # Set to the value of the first axis cut for 1D data.
    ax2_cut_ = 0.0 # Set to the value of the second axis cut for 1D data.

    # Flux Rope Parameters:
    base_sep = 80e-3 # Set to the base separation between flux ropes in m
    tilt_ang = 1.01 # Set to the tilt angle of flux rope with wall in degrees.
    tot_y_len = 1 # Set to the total axis/parallel length of flux ropes in m.
    norm_len = 2e-2 # Set to the normalization length used in m.

    fig_, ax_ = plt.subplots(2,1,figsize=(8,12), dpi=100)

    # Output file:
    output_dir_pre = f'pyframes' # Prefix for output directory
    savedpi = 300 # Higher dpi means slower runtime, default is 100
#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
def main():
    """
    This function handles the logical flow of the script.
    """

    print("File being used as a script!")

    # Call the multicomp_bmag function:
    # if Config.interactive_flag:
    multicomp_fmax_y(plane=Config.plane_, plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, 
        ax2_cut=Config.ax2_cut_)
    # # else:
    # multicomp_bmag(runno=Config.runno_, variables=Config.variables_, first_slice=Config.first_slice_, 
    #     last_slice=Config.last_slice_, skip_slice=Config.skip_slice_, plane=Config.plane_, 
    #     plane_val=Config.plane_val_, ax1_cut=Config.ax1_cut_, ax2_cut=Config.ax2_cut_)

    print("\nAll Done!")
#-------------------------------------------------------------------------------
# Function To Get Binary Data and Compile into a Dictionary
#-------------------------------------------------------------------------------
def multicomp_fmax_y(runno=None, variables=None, first_slice=None, last_slice=None, 
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
    #---Reads first metadata---------------------------------------------------
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

    #---Read data---------------------------------------------------------------
    # (x,z) zoom in limits:
    xmin, xmax = -3, 3
    zmin, zmax = -1, 5
    x_in = np.where((xx >= xmin) & (xx <= xmax))[0] # x indices for the zoomed-in region
    z_in = np.where((zz >= zmin) & (zz <= zmax))[0]
    x_start, x_end = x_in[0], x_in[-1]+1 # x start and end indices of the zoomed-in region
    y_start, y_end = 0, ny # these indices are to be passed to the vars for slicing them.
    z_start, z_end = z_in[0], z_in[-1]+1
    xx = xx[x_in]; zz = zz[z_in]

    def bdat_read(time_slice):
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

        return bx, by, bz
    
    #---Computing Rope Separation-----------------------------------------------
    def sep_sq_comp(rope_len):
        """
        Computes the separation between the two flux ropes for a given y-value.
        Args
        ----
        rope_len : array
            The y-axis or length values of the flux rope.
        
        Returns
        -------
        d_sep : array
            The half-separation between the two flux ropes.
        """
        rope_len = rope_len[::skip_y]
        # Shifting the y-axis or rope_len to start at 0:
        rope_len_shift = rope_len - rope_len[0]

        # Computing the separation between the flux ropes:
        tilt_angB = np.deg2rad(90 - Config.tilt_ang)
        rope_sep = Config.base_sep/Config.norm_len - 2*rope_len_shift/np.tan(tilt_angB)
        d_sep = rope_sep/2

        rhs_form = np.sqrt(np.square(d_sep)/w0**2 - 1)

        return d_sep, rhs_form, rope_len

    #---Computing tension force-------------------------------------------------
    def btenf_comp(bx, by, bz):
        # Compute the gradients (spatial derivatives) of Bx, By and Bz
        dBx_dx, dBx_dy, dBx_dz = np.gradient(bx, dx, dy, dz)
        dBy_dx, dBy_dy, dBy_dz = np.gradient(by, dx, dy, dz)
        dBz_dx, dBz_dy, dBz_dz = np.gradient(bz, dx, dy, dz)

        # Ignoring the boundaries in y-direction:
        # bx_m = bx[:, 1:-1, :]; by_m = by[:, 1:-1, :]; bz_m = bz[:, 1:-1, :]
        # dBx_dx_m = dBx_dx[:, 1:-1, :]; dBx_dy_m = dBx_dy[:, 1:-1, :]; dBx_dz_m = dBx_dz[:, 1:-1, :]
        # dBy_dx_m = dBy_dx[:, 1:-1, :]; dBy_dy_m = dBy_dy[:, 1:-1, :]; dBy_dz_m = dBy_dz[:, 1:-1, :]
        # dBz_dx_m = dBz_dx[:, 1:-1, :]; dBz_dy_m = dBz_dy[:, 1:-1, :]; dBz_dz_m = dBz_dz[:, 1:-1, :]
        
        # Computing the components of the tension force
        ften_x = bx * dBx_dx + by * dBx_dy + bz * dBx_dz # tension force in the x direction
        ften_y = bx * dBy_dx + by * dBy_dy + bz * dBy_dz # tension force in the y direction
        ften_z = bx * dBz_dx + by * dBz_dy + bz * dBz_dz # tension force in the z direction

        # Smoothing the data
        ften_x = gaussian_filter(ften_x, sigma=2)
        ften_y = gaussian_filter(ften_y, sigma=2)
        ften_z = gaussian_filter(ften_z, sigma=2)

        return ften_x, ften_y, ften_z
    
    #---Compute Pressure Forces-------------------------------------------------
    def bpresf_comp(bx, by, bz):
        # Computing the magnetic pressure
        bpres_xz = (np.square(bx) + np.square(by) + np.square(bz))/2
            
        # Compute the gradients (spatial derivatives) of magnetic pressure
        dbpresxz_dx, dbpresxz_dy, dbpresxz_dz = np.gradient(bpres_xz, dx, dy, dz)

        # Ignoring the boundaries in y-direction:
        # dbpresxz_dxm = dbpresxz_dx[:, 1:-1, :]; dbpresxz_dym = dbpresxz_dy[:, 1:-1, :]
        # dbpresxz_dzm = dbpresxz_dz[:, 1:-1, :]

        # Computing the components of the force
        fpres_x = -dbpresxz_dx # pressure force in the x direction
        fpres_y = -dbpresxz_dy # pressure force in the y direction
        fpres_z = -dbpresxz_dz # pressure force in the z direction

        # Smoothing the data
        fpres_x = gaussian_filter(fpres_x, sigma=2)
        fpres_y = gaussian_filter(fpres_y, sigma=2)
        fpres_z = gaussian_filter(fpres_z, sigma=2)

        return fpres_x, fpres_y, fpres_z

    #---Estimating if they're within the flux rope------------------------------
    def plot_inORout(i, saveFlag):
        
        if saveFlag:
            fig_, ax_ = plt.subplots(2,1,figsize=(8,12), dpi=100)
        else:
            fig_ = Config.fig_
            ax_ = Config.ax_
        
        fx_skip_c = fx_skip[:,i,:]
        fz_skip_c = fz_skip[:,i,:]

        # Estimating the index of the extrema
        fx_min_xin, fx_min_zin = np.unravel_index(fx_skip_c.argmin(), fx_skip_c.shape)
        fx_max_xin, fx_max_zin = np.unravel_index(fx_skip_c.argmax(), fx_skip_c.shape)
        fz_min_xin, fz_min_zin = np.unravel_index(fz_skip_c.argmin(), fz_skip_c.shape)
        fz_max_xin, fz_max_zin = np.unravel_index(fz_skip_c.argmax(), fz_skip_c.shape)

        image_cont(fx_skip[:,i,:], xx, zz, fig=fig_, ax=ax_[0], xlabel='x', ylabel='z', 
                       title=f'F_x at y = {rope_len[i]:02.2f}', colmp='gist_heat')
        # ax_[0].add_patch(plt.Rectangle((xx[x_st_c], zz[z_st_c]), xx[x_en_c] - xx[x_st_c], 
                            # zz[z_en_c] - zz[z_st_c], edgecolor='blue', facecolor='none', linewidth=2))
        ax_[0].add_patch(plt.Circle((0, d_sep[i]), w0, color='cyan', 
                                    fill=False, linewidth=2.0, linestyle='dashed'))
        ax_[0].scatter(xx_c[fx_max_xin], zz_c[fx_max_zin], color='green', s=50, marker='*', 
                       label=f'Max Value: {fx_max[i]:02.2f}')
        ax_[0].scatter(xx_c[fx_min_xin], zz_c[fx_min_zin], color='royalblue', s=50, marker='*',
                       label=f'Min Value: {fx_min[i]:02.2f}')
        ax_[0].legend(fontsize= 12, loc='upper left', framealpha=0.5)

        image_cont(fz_skip[:,i,:], xx, zz, fig=fig_, ax=ax_[1], xlabel='x', ylabel='z', 
                       title=f'$F_z$ at y = {rope_len[i]:02.2f}', colmp='gist_heat')
        # ax_[1].add_patch(plt.Rectangle((xx[x_st_c], zz[z_st_c]), xx[x_en_c] - xx[x_st_c],
                                        # zz[z_en_c] - zz[z_st_c], edgecolor='blue', facecolor='none', linewidth=2))
        ax_[1].add_patch(plt.Circle((0, d_sep[i]), w0, color='cyan', fill=False, linewidth=2.0, linestyle='dashed'))
        ax_[1].scatter(xx_c[fz_max_xin], zz_c[fz_max_zin], color='green', s=50, marker = '*', 
                        label=f'Max Value: {fz_max[i]:02.2f}')
        ax_[1].scatter(xx_c[fz_min_xin], zz_c[fz_min_zin], color='royalblue', s=50, marker = '*', 
                        label=f'Min Value: {fz_min[i]:02.2f}')
        ax_[1].legend(fontsize= 12, loc='upper left', framealpha=0.5)

        plt.tight_layout()

        if saveFlag:
            output_dir = f'{output_dir_pref}/angular_force_check'
            os.makedirs(output_dir, exist_ok=True)
            fig_.savefig(f'{output_dir}/yy_{i:04d}.png', dpi=100, bbox_inches='tight')
            plt.close()

            # Check if any of the points are outside the flux rope
            dist_fxmax = np.hypot(0-xx_c[fx_max_xin], d_sep[i]-zz_c[fx_max_zin])
            dist_fxmin = np.hypot(0-xx_c[fx_min_xin], d_sep[i]-zz_c[fx_min_zin])
            dist_fzmax = np.hypot(0-xx_c[fz_max_xin], d_sep[i]-zz_c[fz_max_zin])
            dist_fzmin = np.hypot(0-xx_c[fz_min_xin], d_sep[i]-zz_c[fz_min_zin])

            out_of_rope = []

            if dist_fxmax >= w0:
                out_of_rope.append(f"fx_max outside at y = {rope_len[i]:02.2f}")
            if dist_fxmin >= w0:
                out_of_rope.append(f"fx_min outside at y = {rope_len[i]:02.2f}")
            if dist_fzmax >= w0:
                out_of_rope.append(f"fz_max outside at y = {rope_len[i]:02.2f}")
            if dist_fzmin >= w0:
                out_of_rope.append(f"fz_min outside at y = {rope_len[i]:02.2f}")
            
            return out_of_rope
        else:
            plt.pause(0.00001)
            a = input("Press Enter to continue...") 
    
    #---Plotting Line Plots-----------------------------------------------------
    def line_plot(rope_len, rhs_form, dfz_dfx):
        
        fig, ax = plt.subplots()
        ax.plot(rope_len, rhs_form, color='black', marker='.', markerfacecolor='lightgray', 
                markeredgecolor='black', markersize=4, label='rhs_form')
        ax.plot(rope_len, dfz_dfx, color='blue', marker='.', markerfacecolor='white', 
                markeredgecolor='blue', markersize=4, label='dfz_dfx')

        # Add labels and a title 
        ax.set_xlabel('Rope Length')
        # ax.set_ylabel('Values')
        ax.set_title(f'Angular Analysis for guide field {gfield} G')
        # Add a legend
        ax.legend()
        plt.pause(0.00001)
        a = input("Press Enter to save n exit...") 
        # Saving the plot
        output_dir = f'{output_dir_pref}/angular_force_plot'
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f'{output_dir}/angular_force_plot_{gfield}.png', dpi=100, bbox_inches='tight')
        plt.close()

    #---Executing the functions-------------------------------------------------
    skip_y = int(input("\nEnter the number of y-indices to skip (min = 1): \n"))

    # Computing the separation
    w0 = 1.0 # Radius of the flux rope
    d_sep, rhs_form, rope_len = sep_sq_comp(yy)
    num_elems = len(d_sep)

    # Calling bdat_read for reading bx, by, bz file:
    time_range = range(first_slice,last_slice+1,skip_slice)
    for time_slice in time_range: 
        bx, by, bz = bdat_read(time_slice)
    print('\nMagnetic Field Data Read!')

   # Computing tension and pressure forces
    with joblib_progress("Computing tension & pressure forces...", total=2):
        force_dat = Parallel(n_jobs=2)([delayed(btenf_comp)(bx, by, bz), 
                                                delayed(bpresf_comp)(bx, by, bz)])
    # Unpacking the results:
    ften_x, ften_y, ften_z = force_dat[0]
    fpres_x, fpres_y, fpres_z = force_dat[1]

    # Computing combined forces
    fx = ften_x + fpres_x
    fy = ften_y + fpres_y
    fz = ften_z + fpres_z

    # Find indices of the bounding box for searching
    xmin_bound = -2; xmax_bound = 2
    zmin_bound = 0; zmax_bound = 4
    x_st_c, x_en_c = np.searchsorted(xx, [xmin_bound, xmax_bound])
    z_st_c, z_en_c = np.searchsorted(zz, [zmin_bound, zmax_bound])
    xx_c = xx[x_st_c:x_en_c] # bounding box axes
    zz_c = zz[z_st_c:z_en_c]

    # Slicing the data
    fx_skip = fx[x_st_c:x_en_c, ::skip_y, z_st_c:z_en_c] # skipping the y indices if any
    fz_skip = fz[x_st_c:x_en_c, ::skip_y, z_st_c:z_en_c]

    # Estimating extrema on this sliced data
    fx_min = np.min(fx_skip, axis=(0,2)); fx_max = np.max(fx_skip, axis=(0,2))
    fz_min = np.min(fz_skip, axis=(0,2)); fz_max = np.max(fz_skip, axis=(0,2))
    fx_diff = fx_max + fx_min
    fz_diff = np.abs(fz_max + fz_min)
    # fz_diff = np.abs(fz_min)
    dfz_dfx = fz_diff/fx_diff

    # Plot Check
    chflag = int(input("\nDo you want to do a plot check? (1 = Yes,0 = No): \n"))
    output_dir_pref = f'{Config.output_dir_pre}{runno}' # Savepath prefix


    if chflag:
        savechkFlag = int(input("\nDo you want to save the plot check? (1 = Yes,0 = No): \n"))
        if savechkFlag:
            curr_backend = matplotlib.get_backend()
            print(f"Current Backend: {curr_backend}")
            matplotlib.use('Agg')
            print(f'New Backend: {matplotlib.get_backend()}')
            with joblib_progress("Saving Figures...", total=num_elems):
                out_results = Parallel(n_jobs=16)(delayed(plot_inORout)(ind, savechkFlag) 
                                        for ind in range(num_elems))
            # writing to file if any extrema is outside:
            print('Writing to File...')
            with open('out_of_rope.txt', 'w') as f:
                for res in out_results:
                    if res:  # If the list is not empty
                        for item in res:
                            f.write(item + '\n')
            matplotlib.use(curr_backend)
        else:
            for ind in range(num_elems):
                plot_inORout(ind, savechkFlag)

        plt.close()

    # Plotting line plots
    linepltFlag = int(input("\nDo you want to line plot? (1 = Yes,0 = No): \n"))
    if linepltFlag:
        gfield = input("\nEnter the guide field value in Gauss (eg: 375): \n")
        print('Plotting Line Plots...')
        line_plot(rope_len, rhs_form, dfz_dfx)
        # Saving data
        print('Saving Data...')
        np.savez('angular_force_dat.npz', fx_min=fx_min, fx_max=fx_max, fz_min=fz_min, 
            fz_max=fz_max, dfz_dfx=dfz_dfx, d_sep=d_sep, rope_len=rope_len, rhs_form=rhs_form)     
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
    # # Get variable IDs from user
    # varId = input("Enter the numbers of variables that you wish to have " 
    #                 "loaded separated by comma like: 1,2,3\n"
    #                 "If you wish to load all variables, enter 99\n"
    #                 "To load all variables but eflds, enter 98\n"
    #                 "For all EMHD data, enter 93: \n")
    
    # # Create variables from user input
    # if varId == '99': # Load all variables
    #     variables = list(choices.values())
    # elif varId == '98': # Load all variables except eflds
    #     variables = list(choices.values())[0:12]
    # elif varId == '93': # Load all EMHD data
    #     variables = list(choices.values())[0:3] + list(choices.values())[6:9]
    # else:
    #     varId = [int(idx.strip()) for idx in varId.split(',')] # Convert IDs to integers
    #     variables = [choices[idx] for idx in varId] # Create variables from IDs

    # Create filenames from arguments
    variables = ['bx','by','bz']
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
    print("Module multicomp_bmag imported!")
#-------------------------------------------------------------------------------

