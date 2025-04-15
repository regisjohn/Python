#!/usr/bin/env python3
#===============================================================================
"""
This is a module to make contour plots of data from F3D.
|Author: Regis | Date Created/Last Modified: Oct 12, 2023/Feb 12, 2025|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
#-------------------------------------------------------------------------------
# Function to make contour plots using imshow() quicker than pcolormesh()
#-------------------------------------------------------------------------------
def image_cont(pltdat, xx=None, yy=None , fig=None, ax=None, title=None, 
              xlabel = 'x', ylabel = 'y', timedt=None, dpi_val=100, 
              figsz = None, colmp = 'gist_heat', vmin_=None, vmax_=None):
    """
    This function generates a heatmap of the input data using imshow(). 
    One can simply call this function as `image_cont(data[x,y])` 
    since other parameters are optional. To use this simply as a plotting 
    function, define and pass the figure object to image_cont().

    Parameters:
    - pltdat (np.ndarray): The input data to be visualized.
    - xx (np.ndarray, optional): The x-coordinates of the data points.
    - yy (np.ndarray, optional): The y-coordinates of the data points.
    - fig (matplotlib.figure.Figure, optional): The figure object. If not provided,
        a new figure and axes will be created.
    - ax (matplotlib.axes.Axes, optional): The axes object. If provided, the function
        will clear any existing content and remove any existing colorbar from the axes. 
    - title (str, optional): The title of the plot.
    - xlabel (str, optional): The label of the x-axis.
    - ylabel (str, optional): The label of the y-axis.
    - timedt (float, optional): The time value associated with the data. 
    - dpi_val (int, optional): The dots per inch of the figure, default is 100.
    - figsz (tuple, optional): The figure size in inches, default is None.
    - colmp (str, optional): The colormap to use for the heatmap, default is 'hot'.
    - vmin_ (float, optional): The minimum value for the colormap in the colormap. Default is None.
    - vmax_ (float, optional): The maximum value for the colormap in the colormap. Default is None.

    Returns:
    - Image attached to axes.

    Example:
    `   import numpy as np
        import matplotlib.pyplot as plt
        from image_cont import image_cont

        data = np.random.rand(10, 10)
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)

        image_cont(pltdat=data, xx_=x, yy_=y, title="Sample Plot", 
                   xlabel="X-axis", ylabel="Y-axis", colmp='viridis')`
    """ 
    # Create a new figure and axes if not provided
    fig_flag = False # Flag to check if image_cont created figure or not.
    if fig is None:
        fig, ax = plt.subplots(figsize=figsz,dpi=dpi_val)
        fig_flag = True
    else:
        # Removes the colorbar if it exists, since the default one height is small 
        # & in a loop, it will lead to overplotting. Do this only if ax is provided.
        if hasattr(ax, "cbar"):
            ax.cbar.remove()
        # clearing the axis of anything remaining from the previous plotting, don't
        # reverse the order of these two statements of cbar.remove and ax.clear()!
        ax.clear()
    
    # Parse the data
    pltdat = pltdat.T  # Transpose to match the row-column plotting order in matplotlib
    pltdat_max = pltdat.max()  # Find the maximum value in the data
    pltdat_min = pltdat.min()  # Find the minimum value in the data
    
    # Plot the data
    if xx is None or yy is None:  # If no x and y coordinates are provided
        img = ax.imshow(pltdat, origin='lower', cmap=colmp, vmin=vmin_, vmax=vmax_)
    else:  # If x and y coordinates are provided
        img = ax.imshow(pltdat, extent=[xx[0], xx[-1], yy[0], yy[-1]],
                        origin='lower', cmap=colmp, vmin=vmin_, vmax=vmax_)

    # Set the title of the plot
    if title is None:
        ax.set_title(f'{pltdat_min:.3f}  {pltdat_max:.3f}', fontsize=16)
    else: 
        # ax.set_title(f'{title}: {pltdat_min:.3f}  {pltdat_max:.3f}') 
        ax.set_title(f'{title}', fontsize=16) 
    
    # Set the labels of the x and y axes
    if timedt is None:
        ax.set_xlabel(f'{xlabel}',fontsize=12)
    else: 
        ax.set_xlabel(f'{xlabel}\nt = {timedt:.5f}',fontsize=12)
    ax.set_ylabel(f'{ylabel}',fontsize=12)
    
    # Set the colorbar
    if xx is None or yy is None:
        height, width = np.shape(pltdat)
    else:
        height, width = len(yy), len(xx)
    ax.cbar = fig.colorbar(img,ax=ax,fraction=0.047*height/width, pad=0.04, 
                           format='%.2f') # Adding an appropriate sized colorbar
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_aspect('equal') 
    
    # Display the plot if fig object is not passed as parameter
    if fig_flag:
        plt.pause(0.1)  # Pause for a short time to allow the plot to be displayed
        plt.tight_layout()  # Adjust the padding between and around subplots          

#-------------------------------------------------------------------------------
# Script Execution or Module Import Check
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    raise RuntimeError("\nHi this script is not meant to be run directly!" 
          "\nCall the function as image_cont(data[x,y]) for quick use."
          "\nFor more info refer to documentation.\n")
# else:
#     print("Module image_cont imported!")