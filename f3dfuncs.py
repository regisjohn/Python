#!/usr/bin/env python3
#===============================================================================
"""
This is a module that contains all the core functions for F3D data analysis.
|Author: Regis | Date Created/Last Modified: Oct 20, 2023/Sept 17, 2024|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
import numexpr as ne
#-------------------------------------------------------------------------------
ne.set_num_threads(64) # Set numexpr to use as many threads as possible
#-------------------------------------------------------------------------------
# Function to Read Metadata
#-------------------------------------------------------------------------------
def read_first_metadata(mfile_):
    """
    Read the first two lines of metadata from the given file object and extract
    relevant values. Estimate the size of the x-axis, y-axis, and z-axis based 
    on the extracted values. Estimate the time-step and number of time slices.

    Parameters:
    - mfile_ (file object): The file object to read metadata from.

    Returns:
    tuple: A tuple containing the following values:
    - nx_ (int): The number of points in the x-axis.
    - ny_ (int): The number of points in the y-axis.
    - nz_ (int): The number of points in the z-axis.
    - dx_ (float): The spacing between points in the x-axis.
    - dy_ (float): The spacing between points in the y-axis.
    - dz_ (float): The spacing between points in the z-axis.
    - dtime_ (float): The time between time slices.
    - ntimes_ (int): The number of time slices.
    - metadat_len_bytes_ (int): The length of the metadata in bytes.
    - xx_ (numpy.ndarray): The array of x-axis values.
    - yy_ (numpy.ndarray): The array of y-axis values.
    - zz_ (numpy.ndarray): The array of z-axis values.
    """
    # Read the first two lines of metadata
    metadata_text1 = mfile_.readline()
    metadata_text2 = (mfile_.readline()).decode('utf-8').split()
    metadat_len_bytes_ = len(metadata_text1)*2

    # Extract values from metadata
    nx_ = int(metadata_text2[2])
    ny_ = int(metadata_text2[3])
    nz_ = int(metadata_text2[4])
    xmin_ = float(metadata_text2[5])
    dx_ = float(metadata_text2[6])
    ymin_ = float(metadata_text2[7])
    dy_ = float(metadata_text2[8])
    zmin_ = float(metadata_text2[9])
    dz_ = float(metadata_text2[10])

    # Estimating size of x-axis, y-axis, and z-axis
    xx_ = np.arange(xmin_, xmin_ + dx_ * nx_, dx_)
    yy_ = np.arange(ymin_, ymin_ + dy_ * ny_, dy_)
    zz_ = np.arange(zmin_, zmin_ + dz_ * nz_, dz_)

    # Estimating the time-step and number of time slices
    mfile_.seek(nx_ * ny_ * nz_ + metadat_len_bytes_)
    mfile_.readline()
    metadata_text3 = (mfile_.readline()).decode('utf-8').split()
    dtime_ = float(metadata_text3[11]) - float(metadata_text2[11])
    mfile_.seek(0, 2) # moving pointer to the end of the file
    ntimes_ = int(mfile_.tell() / (nx_ * ny_ * nz_))

    return nx_, ny_, nz_, dx_, dy_, dz_, dtime_, ntimes_, metadat_len_bytes_, \
        xx_, yy_, zz_  
#-------------------------------------------------------------------------------
# Function to Read F3D Binary File
#-------------------------------------------------------------------------------
def read_binary_f3d(bfile, time_slice, nx, ny, nz, xx, yy, zz, xstart, xend, 
                    ystart, yend, zstart, zend, plane, plane_val, ax1_cut, 
                    ax2_cut, metadat_len_bytes):
    """
    Reads binary data from an F3D file and returns a sliced portion of the data.

    Parameters:
    - bfile (file): The binary file to read from.
    - time_slice (int): The time slice to read from the file.
    - nx (int): The number of x points in the data.
    - ny (int): The number of y points in the data.
    - nz (int): The number of z points in the data.
    - xx (array): The x coordinates of the data points.
    - yy (array): The y coordinates of the data points.
    - zz (array): The z coordinates of the data points.
    - xstart (int): The starting x index of the slice.
    - xend (int): The ending x index of the slice.
    - ystart (int): The starting y index of the slice.
    - yend (int): The ending y index of the slice.
    - zstart (int): The starting z index of the slice.
    - zend (int): The ending z index of the slice.
    - plane (str): The plane to slice the data in ('xyz', 'xy', 'yz', 'xz', 'x', 'y', 'z').
    - plane_val (float): The value at which to slice the data in the specified plane.
    - ax1_cut (float): The value at which to slice the data in the first axis.
    - ax2_cut (float): The value at which to slice the data in the second axis.
    - metadat_len_bytes (int): The length of the metadata in bytes.

    Returns:
    - field_sliced (array): The sliced portion of the data.
    """

    bfile.seek(time_slice * nx * ny * nz + time_slice * metadat_len_bytes)
    # Skipping one line and reading in metadata for fmin and fmax
    bfile.readline()
    metadata_temp = (bfile.readline()).decode('utf-8').split()
    fmin = float(metadata_temp[0]); fmax = float(metadata_temp[1])
    # Reading in the data and converting it into float using fmin and fmax
    field = np.fromfile(bfile, dtype='uint8', count=nx * ny * nz).reshape((nx, ny, nz), order='F')
    # field = field.reshape((nx, ny, nz), order='F')
    # field = fmin + (fmax - fmin) * field / 255.
   
    # Slicing the data according to the plane/cut:
    if plane == 'xyz': 
        field_sliced = field[xstart:xend, ystart:yend, zstart:zend]
    elif plane == 'xy': # 2-D x-z array cut at a z location
        cutloc = np.abs(zz - plane_val).argmin()
        # copying to avoid carrying over the whole array which might lead to memory issues
        field_sliced = field[xstart:xend, ystart:yend, cutloc]
    elif plane == 'yz': # 2-D y-z array cut at an x location
        cutloc = np.abs(xx - plane_val).argmin()
        field_sliced = field[cutloc, ystart:yend, zstart:zend]
    elif plane == 'xz': # 2-D x-y array cut at a y location
        cutloc = np.abs(yy - plane_val).argmin()
        field_sliced = field[xstart:xend, cutloc, zstart:zend]
    elif plane == 'x': # 1-D x array cut at y and z locations
        cutloc1 = np.abs(yy - ax1_cut).argmin()
        cutloc2 = np.abs(zz - ax2_cut).argmin()
        field_sliced = field[xstart:xend, cutloc1, cutloc2]
    elif plane == 'y': # 1-D y array cut at x and z locations
        cutloc1 = np.abs(xx - ax1_cut).argmin()
        cutloc2 = np.abs(zz - ax2_cut).argmin()
        field_sliced = field[cutloc1,ystart:yend, cutloc2]
        # field_sliced = np.array(field[cutloc1,:,cutloc2])
    else: # self.plane == 'z': # 1-D z array cut at x and y locations
        cutloc1 = np.abs(xx - ax1_cut).argmin()
        cutloc2 = np.abs(yy - ax2_cut).argmin()
        field_sliced= field[cutloc1, cutloc2, zstart:zend]
    
    # Below numexpr operation creates a new array field_sliced instead of a view.
    field_sliced = ne.evaluate("fmin + (fmax - fmin) * field_sliced / 255.")

    return field_sliced
#-------------------------------------------------------------------------------