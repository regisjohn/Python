#!/usr/bin/env python3
#===============================================================================
"""
This is a python script for reading fx and fz extrema values that is locally 
saved as .npz files and make plots of dfx values vs y for different guide 
fields and we also do the same for dfz values. It also makes a plot of the 
grazing angle and the angle of resultant force for different guide fields to 
check if there is any correlation between them.
|Author: Regis | Date Created/Last Modified: Jan 26, 2025/Jan 26, 2025|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
#-------------------------------------------------------------------------------
# Configuration Class for parameters and flags
#-------------------------------------------------------------------------------
class Config:
    """
    This configuration class is used to set the parameters and flags for the
    script.
    """
    pass

#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
def main():
    """
    This function handles the logical flow of the script.
    """

    print("File being used as a script!")

    # Prompt Run Number:
    runno = input("\nEnter the run number: \n")
    # Get Guide Field:
    gfield = input("\nEnter the guide field in G: \n")
    # Output Directory:
    outdir = 'forceplots'
    # Radius of flux rope:
    w0 = 1.0

    # Create npz filename:
    npz_t0 = f"{runno}_t0.npz"
    np_ang_force = f"angular_force_dat_{runno}.npz"

    # Call functions:
    plot_dfvsy(runno, gfield, outdir)
    plot_fextvsy(runno, gfield, outdir)
    plot_grazing_angle(runno, gfield, outdir, w0)
    plot_grazing_angle_subplot(runno, gfield, outdir, w0)

        
    print("\nAll Done!")
#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def plot_dfvsy(runno, gfield, outdir):
    # Creating npz filename:
    np_ang_force = f"angular_force_dat_{runno}.npz"
    # Passing npz filename to function to load data:
    dat_df = load_localnpz(np_ang_force)
    # Extracting data:
    fx_min = dat_df['fx_min']
    fx_max = dat_df['fx_max']
    fz_min = dat_df['fz_min']
    fz_max = dat_df['fz_max']
    dfz_dfx = dat_df['dfz_dfx']
    d_sep = dat_df['d_sep']
    rope_len = dat_df['rope_len']
    rhs_form = dat_df['rhs_form']

    # Calculating dfx:
    dfx = fx_max + fx_min
    # Calculating dfz:
    dfz = np.abs(fz_max + fz_min)

    # Plotting:
    fig, ax = plt.subplots(1,2, figsize=(16,6), dpi=100)
    ax[0].plot(rope_len, dfx, color='black', marker='.', markerfacecolor='white', 
            markeredgecolor='black', markersize=4, label='dfx')

    ax[0].set_xlabel('Rope Length', fontsize=14); ax[0].set_ylabel('dfx', fontsize=14)
    ax[0].legend(fontsize = 14)
    ax[1].plot(rope_len, dfz, color='black', marker='.', markerfacecolor='white', 
            markeredgecolor='black', markersize=4, label='dfz')
    ax[1].set_xlabel('Rope Length', fontsize=14); ax[1].set_ylabel('dfz', fontsize=14)
    ax[1].legend(fontsize = 14)

    fig.suptitle(f'df vs Rope Length for guide field {gfield}G', fontsize=20) 

    plt.tight_layout()
    
    plt.pause(0.00001)
    a = input("\nPress Enter to save n exit...") 
    
    # Saving the plot
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/dfvsy_{gfield}.png', dpi=100, bbox_inches='tight')
    plt.close()

def plot_fextvsy(runno, gfield, outdir):
    # Creating npz filename:
    np_ang_force = f"angular_force_dat_{runno}.npz"
    # Passing npz filename to function to load data:
    dat_df = load_localnpz(np_ang_force)
    # Extracting data:
    fx_min = dat_df['fx_min']
    fx_max = dat_df['fx_max']
    fz_min = dat_df['fz_min']
    fz_max = dat_df['fz_max']
    dfz_dfx = dat_df['dfz_dfx']
    d_sep = dat_df['d_sep']
    rope_len = dat_df['rope_len']
    rhs_form = dat_df['rhs_form']

    # Plotting:
    fig, ax = plt.subplots(2,2, figsize=(16,12), dpi=100)
    ax[0,0].plot(rope_len, fx_max, color='black', marker='.', markerfacecolor='white', 
            markeredgecolor='black', markersize=4, label='fx_max')
    ax[0,1].plot(rope_len, fx_min, color='black', marker='.', markerfacecolor='white', 
            markeredgecolor='black', markersize=4, label='fx_min')
    ax[0,0].set_xlabel('Rope Length', fontsize=14); ax[0,1].set_xlabel('Rope Length', fontsize=14)
    ax[0,0].set_ylabel('fx_max', fontsize=14); ax[0,1].set_ylabel('fx_min', fontsize=14)
    ax[0,0].legend(fontsize=14); ax[0,1].legend(fontsize=14)

    ax[1,0].plot(rope_len, fz_max, color='black', marker='.', markerfacecolor='white', 
            markeredgecolor='black', markersize=4, label='fz_max')
    ax[1,1].plot(rope_len, fz_min, color='black', marker='.', markerfacecolor='white', 
            markeredgecolor='black', markersize=4, label='fz_min')
    ax[1,0].set_xlabel('Rope Length', fontsize=14); ax[1,1].set_xlabel('Rope Length', fontsize=14)
    ax[1,0].set_ylabel('fz_max', fontsize=14); ax[1,1].set_ylabel('fz_min', fontsize=14)
    ax[1,0].legend(fontsize=14); ax[1,1].legend(fontsize=14)

    fig.suptitle(f'Force extrema vs Rope Length for guide field {gfield}G', fontsize=20) 

    plt.tight_layout()
    
    plt.pause(0.00001)
    a = input("\nPress Enter to save n exit...") 
    
    # Saving the plot
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/fextvsy_{gfield}.png', dpi=100, bbox_inches='tight')
    plt.close()

def plot_grazing_angle(runno, gfield, outdir, w0):
    # Creating npz filename:
    np_ang_force = f"angular_force_dat_{runno}.npz"
    # Passing npz filename to function to load data:
    dat_df = load_localnpz(np_ang_force)
    # Extracting data:
    fx_min = dat_df['fx_min']
    fx_max = dat_df['fx_max']
    fz_min = dat_df['fz_min']
    fz_max = dat_df['fz_max']
    dfz_dfx = dat_df['dfz_dfx']
    d_sep = dat_df['d_sep']
    rope_len = dat_df['rope_len']
    rhs_form = dat_df['rhs_form']

    # Calculating grazing angles:
    grazing_val = w0/d_sep
    alpha = np.pi/2 - np.arctan(grazing_val)
    theta = np.arctan(dfz_dfx)

    # Plotting:
    fig, ax = plt.subplots(1,1, figsize=(8,6), dpi=100)
    ax.plot(rope_len, alpha, color='black', marker='.', markerfacecolor='lightgray', 
            markeredgecolor='black', markersize=4, label='Grazing Angle')
    ax.plot(rope_len, theta, color='blue', marker='*', markerfacecolor='white', 
            markeredgecolor='blue', markersize=4, label='Resultant Diff Force Angle')
    ax.set_xlabel('Rope Length', fontsize=14); ax.set_ylabel('Angle (rad)', fontsize=14)
    ax.legend(fontsize = 12)
    ax.set_title(f'Angle vs Rope Length for guide field {gfield}G', fontsize=20)

    plt.tight_layout()
    
    plt.pause(0.00001)
    a = input("\nPress Enter to save n exit...") 
    
    # Saving the plot
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/grazing_angle_{gfield}.png', dpi=100, bbox_inches='tight')
    plt.close()


def plot_grazing_angle_subplot(runno, gfield, outdir, w0):
    # Creating npz filename:
    np_ang_force = f"angular_force_dat_{runno}.npz"
    # Passing npz filename to function to load data:
    dat_df = load_localnpz(np_ang_force)
    # Extracting data:
    fx_min = dat_df['fx_min']
    fx_max = dat_df['fx_max']
    fz_min = dat_df['fz_min']
    fz_max = dat_df['fz_max']
    dfz_dfx = dat_df['dfz_dfx']
    d_sep = dat_df['d_sep']
    rope_len = dat_df['rope_len']
    rhs_form = dat_df['rhs_form']

    # Calculating grazing angles:
    grazing_val = w0/d_sep
    alpha = np.pi/2 - np.arctan(grazing_val)
    theta = np.arctan(dfz_dfx)
    theta_m = np.arctan(fz_max/fx_max)

    # Plotting:
    fig, ax = plt.subplots(1,2, figsize=(16,6), dpi=100)
    ax[0].plot(rope_len, alpha, color='black', marker='.', markerfacecolor='lightgray', 
            markeredgecolor='black', markersize=4, label='Grazing Angle')
    ax[0].plot(rope_len, theta, color='blue', marker='*', markerfacecolor='white', 
            markeredgecolor='blue', markersize=4, label='Resultant Diff Force Angle')
    ax[0].set_xlabel('Rope Length', fontsize=14); ax[0].set_ylabel('Angle (rad)', fontsize=14)
    ax[0].legend(fontsize = 12)
    
    ax[1].plot(rope_len, alpha, color='black', marker='.', markerfacecolor='lightgray', 
            markeredgecolor='black', markersize=4, label='Grazing Angle')
    ax[1].plot(rope_len, theta_m, color='blue', marker='*', markerfacecolor='white', 
            markeredgecolor='blue', markersize=4, label='Resultant Max Force Angle')
    ax[1].set_xlabel('Rope Length', fontsize=14); ax[1].set_ylabel('Angle (rad)', fontsize=14)
    ax[1].legend(fontsize = 12)

    fig.suptitle(f'Angle vs Rope Length for guide field {gfield}G', fontsize=20) 

    plt.tight_layout()
    
    plt.pause(0.00001)
    a = input("\nPress Enter to save n exit...") 
    
    # Saving the plot
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/grazing_angle_subplot_{gfield}.png', dpi=100, bbox_inches='tight')
    plt.close()

def plot_4allguidefield(runno, gfield, outdir):
    # Creating npz filename:
    np_ang_force = f"angular_force_dat_{runno}.npz"
    # Passing npz filename to function to load data:
    dat_df = load_localnpz(np_ang_force)
    # Extracting data:
    fx_min = dat_df['fx_min']
    fx_max = dat_df['fx_max']
    fz_min = dat_df['fz_min']
    fz_max = dat_df['fz_max']
    dfz_dfx = dat_df['dfz_dfx']
    d_sep = dat_df['d_sep']
    rope_len = dat_df['rope_len']
    rhs_form = dat_df['rhs_form']

    # Calculating dfx:
    dfx = fx_max + fx_min
    # Calculating dfz:
    dfz = np.abs(fz_max + fz_min)

    # Plotting:
    fig, ax = plt.subplots(1,2, figsize=(16,6), dpi=100)
    ax[0].plot(rope_len, dfx, color='black', marker='.', markerfacecolor='white', 
            markeredgecolor='black', markersize=4, label='dfx')

    ax[0].set_xlabel('Rope Length', fontsize=14); ax[0].set_ylabel('dfx', fontsize=14)
    ax[0].legend(fontsize = 14)
    ax[1].plot(rope_len, dfz, color='black', marker='.', markerfacecolor='white', 
            markeredgecolor='black', markersize=4, label='dfz')
    ax[1].set_xlabel('Rope Length', fontsize=14); ax[1].set_ylabel('dfz', fontsize=14)
    ax[1].legend(fontsize = 14)

    fig.suptitle(f'df vs Rope Length for guide field {gfield}G', fontsize=20) 

    plt.tight_layout()
    
    plt.pause(0.00001)
    a = input("\nPress Enter to save n exit...") 
    
    # Saving the plot
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/dfvsy_{gfield}.png', dpi=100, bbox_inches='tight')
    plt.close()

#-------------------------------------------------------------------------------
# Function for interactive mode
#-------------------------------------------------------------------------------
def load_localnpz(filenm):
    dat_npz = np.load(filenm)
    print("\nDisplaying the keys of data loaded from .npz file\n")
    print(dat_npz.files)
    
    return  dat_npz

#-------------------------------------------------------------------------------
# Script Execution or Module Import Check
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
else:
    print("Module multiread imported!")
#-------------------------------------------------------------------------------
