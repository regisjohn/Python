#!/usr/bin/env python3
"""
This is a module that reads from the F3D binary file for J and B fields and 
computes the corresponding electric field.
|Author: Regis | Date Created/Last Modified: Oct 22, 2023/Oct 22, 2023|
"""

import numpy as np

from .f3dfuncs import read_binary_f3d

def efieldcomp(runno=None, first_slice=None, plane='xyz', plane_val=0.0):
    
    if None in [runno, first_slice]:
        # Enter interactive mode to get filenames
        filenames = interactive_mode()
        
        # Reading in metadata from the first file:
        nx, ny, nz, dx, dy, dz, dt, ntimes, metadat_len_bytes, xx, yy, \
            zz = read_first_metadata(filenames[0])  
        
        # Get first, last, and skip times from user
        sliceID = input("Enter timeslice to read. First possible frame is 0.\n")
        # Set the first, last, and skip times
        first_slice = int(sliceID)
        last_slice = first_slice
        skip_slice = 1
        ax1_cut=0.0; ax2_cut=0.0
    else:   
        # Creating filenames from arguments
        variables = 'bx by bz curx cury curz'
        variables = variables.split()
        filenames = [f'{variables[i]}_{runno}.mov' for 
                    i in range(len(variables))]
        
        last_slice = first_slice
        skip_slice = 1
        ax1_cut=0.0; ax2_cut=0.0

        # Reading in metadata from the first file:
        nx, ny, nz, dx, dy, dz, dt, ntimes, metadat_len_bytes, xx, yy, \
            zz = read_first_metadata(filenames[0])
        
    print(f'\nLoading time slices from {first_slice} to {last_slice} in steps of {skip_slice}')

    # Loading the curresponding data:
    data_list = []
    time_range = range(first_slice,last_slice+1,skip_slice)
    nframes = len(time_range)
    for filename in filenames:
        print('............')
        with open(filename, 'rb') as bfile:
            for timeslice in time_range:
                data= read_binary_f3d(bfile, timeslice, nx, ny, nz, xx, yy, zz, 
                                      plane, plane_val, ax1_cut, ax2_cut, metadat_len_bytes)
                # printing the filename and time read
                print(f'{filename} for the plane = {plane} and ' 
                    f'time = {timeslice * dt} read!')
            data_list.append(data)

    # Calculating the electric field
    bx = data_list[0]
    by = data_list[1]
    bz = data_list[2]
    curx = data_list[3]
    cury = data_list[4]
    curz = data_list[5]

    print("\nHold your horses! Calculating the electric fields....")
    print('............')

    # Calculating electric field
    ex = cury*bz - curz*by
    ey = curz*bx - curx*bz
    ez = curx*by - cury*bx

    print("\nElectric fields computed!")
    return ex, ey, ez

def interactive_mode():
    choices = {1:'bx', 2:'by', 3:'bz', 4:'curpx', 5:'curpy', 6:'curpz', 
                7:'curx', 8:'cury', 9:'curz', 10:'den', 11:'pfi', 12:'psi', 
                13:'efx', 14:'efy', 15:'efz', 16:'epz'}
    # Display usage instructions
    print("Entering Interactive Mode...\n"
    "You can use this function non-interactively as\n" 
    "`efieldcomp(runno, first_slice, last_slice, skip_slice)`\n"
    "where runno is in int, First_slice, Last_slice, and skip_slice is in int.\n"
    "Eg:- efieldcomp(226,1,10,1)")
    
    # Get run number from user
    runno = int(input("\nEnter the run number: \n"))
    
    # Display the available variables
    print(f'\n{choices}')

    print('Since we are computing electric fields only reading in '
          'bx, by, bz and curx, cury, curz')
    
    # variables = 'bx by bz curx cury curz'
    variables = 'curx cury curz'
    variables = variables.split()
    # Creating filenames
    filenames = [f'{variables[i]}_{runno}.mov' for 
                    i in range(len(variables))]

    return filenames
        
def read_first_metadata(filename):
    with open(filename, 'rb') as file:
        # Read the first two lines of metadata
        metadata_text1 = file.readline()
        metadata_text2 = (file.readline()).decode('utf-8').split()
        metadat_len_bytes = len(metadata_text1)*2

        # Extract values from metadata
        nx = int(metadata_text2[2])
        ny = int(metadata_text2[3])
        nz = int(metadata_text2[4])
        xmin = float(metadata_text2[5])
        dx = float(metadata_text2[6])
        ymin = float(metadata_text2[7])
        dy = float(metadata_text2[8])
        zmin = float(metadata_text2[9])
        dz = float(metadata_text2[10])

        # Estimating size of x-axis, y-axis, and z-axis
        xarr = np.linspace(xmin, xmin + dx * (nx - 1), nx)
        yarr = np.linspace(ymin, ymin + dy * (ny - 1), ny)
        zarr = np.linspace(zmin, zmin + dz * (nz - 1), nz)

        # Estimating the number of time-step and time slices
        file.seek(nx * ny * nz + metadat_len_bytes)
        file.readline()
        metadata_text3 = (file.readline()).decode('utf-8').split()
        dt = float(metadata_text3[11]) - float(metadata_text2[11])
        file.seek(0, 2)
        ntimes = int(file.tell() / (nx * ny * nz))
    
    # Display metadata
    print(f'\nInitial metadata read from : {filename}!')
    print(f'nx={nx}, ny={ny}, nz={nz}, dx={dx}, dy={dy}, dz={dz}')
    print(f'\nTotal Times slices run from 0 to {ntimes-1}')

    return nx, ny, nz, dx, dy, dz, dt, ntimes, metadat_len_bytes, \
        xarr, yarr, zarr

if __name__ == "__main__":
    print("File being used as a script, so running efieldcomp()")
    ex, ey, ez = efieldcomp()