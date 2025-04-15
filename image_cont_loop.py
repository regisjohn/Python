#!/usr/bin/env python3
#===============================================================================
"""
This is a module to make contour plots of data from F3D.
|Author: Regis | Date Created/Last Modified: Oct 12, 2023/Aug 12, 2024|
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
def image_cont(pltdat, xx, yy , fig, ax, title, 
              xlabel = 'x', ylabel = 'y', dpi_val=100, figsz = None, 
              colmp = 'hot', vmin_=None, vmax_=None):
    """
    This function generates a heatmap of the input data using imshow(). This 
    function image_cont_loop as compared to the original image_cont is highly 
    optimized to be called in a loop as it avoids redrawing the axes, labels, 
    and colorbar on subsequent calls. It checks if the image object already 
    exists using `hasattr` and only updates the image data using 
    `set_data` and `set_clim`. This does not support changing axes in a loop.

    Parameters:
    - pltdat (np.ndarray): The input data to be visualized.
    - xx (np.ndarray): The x-coordinates of the data points.
    - yy (np.ndarray): The y-coordinates of the data points.
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes.Axes): The axes object. 
    - title (str, optional): The title of the plot.
    - xlabel (str): The label of the x-axis.
    - ylabel (str): The label of the y-axis.
    - dpi_val (int, optional): The dots per inch of the figure, default is 100.
    - figsz (tuple, optional): The figure size in inches, default is None.
    - colmp (str, optional): The colormap to use for the heatmap, default is 'hot'.
    - vmin_ (float, optional): The minimum value for the colormap in the colormap. Default is None.
    - vmax_ (float, optional): The maximum value for the colormap in the colormap. Default is None.

    Returns:
    - Image attached to axes.

    Example:
        import numpy as np
        import matplotlib.pyplot as plt
        from image_cont1 import image_cont_loop

        data = np.random.rand(10, 10)
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        fig, ax = plt.subplots()

        # Initial plot
        image_cont_loop(pltdat=data, xx=x, yy=y, fig=fig, ax=ax, title="Sample Plot", 
                        xlabel="X-axis", ylabel="Y-axis", colmp='viridis', vmin_=0, vmax_=1)
        
        # Update plot in a loop
        for _ in range(10):
            data = np.random.rand(10, 10)
            image_cont_loop(pltdat=data, xx=x, yy=y, fig=fig, ax=ax, colmp='viridis', vmin_=0, vmax_=1)
    """
 
    
    # Parse the data
    pltdat = pltdat.T  # Transpose to match the row-column plotting order in matplotlib
    # pltdat_max = pltdat.max()  # Find the maximum value in the data
    # pltdat_min = pltdat.min()  # Find the minimum value in the data
    
    # Plot the data
    if hasattr(ax, "image_cont"):
        ax.image_cont.set_data(pltdat)
        ax.image_cont.set_clim(vmin=vmin_, vmax=vmax_)
    else:
        ax.image_cont = ax.imshow(pltdat, extent=[xx[0], xx[-1], yy[0], yy[-1]],
                        origin='lower', cmap=colmp, vmin=vmin_, vmax=vmax_)
        # Set the title and axes labels.
        ax.set_title(f'{title}') 
        ax.set_xlabel(f'{xlabel}')
        ax.set_ylabel(f'{ylabel}')
        # Set the colorbar
        height, width = len(yy), len(xx)
        ax.cbar = fig.colorbar(ax.image_cont,ax=ax,fraction=0.047*height/width, pad=0.04, 
                            format='%.2f') # Adding an appropriate sized colorbar
        ax.set_aspect('equal')
        

#-------------------------------------------------------------------------------
# Script Execution or Module Import Check
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    raise RuntimeError("\nHi this script is not meant to be run directly!" 
          "\nThis function image_cont_loop(data[x,y],*) is to be used in a for loop."
          "\nFor more info refer to documentation.\n")
else:
    print("Module image_cont_loop imported!")