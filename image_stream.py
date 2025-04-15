#!/usr/bin/env python3
#===============================================================================
"""
This is a module to make streamline plots of data from F3D.
|Author: Regis | Date Created/Last Modified: Aug 11, 2024/Sept 2, 2024|
"""
#===============================================================================
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
#-------------------------------------------------------------------------------
# Function to make streamline plots using streamplot()
#-------------------------------------------------------------------------------
def image_stream(X, Y, U, V, mag=None, mag_min=None, mag_max=None , 
                    fig=None, ax = None, colmp='inferno', xlabel='x', 
                    ylabel='y', title=None, density=1, dpi_val=100, figsz = None):
    """
    Creates a 2D streamline plot of the given vector field.

    Parameters:
    - X (array): The x-coordinates of the grid points. Can be 1D or 2D.
    - Y (array): The y-coordinates of the grid points. Can be 1D or 2D.
    - U (2D array): The x-components of the vector field.
    - V (2D array): The y-components of the vector field.
    - mag (2D array, optional): The magnitude of the vector field. If not provided, 
        it will be calculated using the magnitude of U and V. Defaults to None.
    - mag_min (float, optional): The minimum value of the magnitude for colormap normalization. 
        If not provided, it will be calculated from mag.min(). Defaults to None.
    - mag_max (float, optional): The maximum value of the magnitude for colormap normalization. 
        If not provided, it will be calculated from mag.max(). Defaults to None.
    - fig (matplotlib.figure.Figure, optional): The figure to plot on. 
        If not provided, a new figure will be created. Defaults to None.
    - ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, 
        a new axes will be created. Defaults to None.
    - colmp (str, optional): The colormap to use for the streamlines. Defaults to 'inferno'.
    - xlabel (str, optional): The label for the x-axis. Defaults to 'x'.
    - ylabel (str, optional): The label for the y-axis. Defaults to 'y'.
    - title (str, optional): The title of the plot. Defaults to None.
    - dpi_val (int, optional): The DPI of the figure. Defaults to 100.
    - figsz (tuple, optional): The size of the figure. Defaults to None.

    Returns:
    - Image attached to axes.

    Example:
    `   import numpy as np
        import matplotlib.pyplot as plt

        # Create a grid
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.linspace(0, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, y)

        # Define a vector field
        U = np.cos(X) * np.sin(Y)
        V = -np.sin(X) * np.cos(Y)

        # Create the plot
        image_stream(X, Y, U, V, title='Streamline Plot')`
    """
    # Parse the data:
    U = U.T  # Transpose to match the row-column plotting order in matplotlib
    V = V.T

    # Checking if X and Y are 1D or 2D:
    if X.ndim == 1:
        X, Y = np.meshgrid(X, Y)

    # Create a new figure and axes if not provided
    fig_flag = False # Flag to check if image_cont created figure or not.
    if fig is None:
        fig, ax = plt.subplots(figsize=figsz,dpi=dpi_val)
        fig_flag = True
    else:
        # Removes the colorbar if it exists, since the default one height is small 
        # & in a loop, it will lead to overplotting. Do this only if ax is provided.
        if hasattr(ax, "strmbar"):
            ax.strmbar.remove()
        # clearing the axis of anything remaining from the previous plotting, don't
        # reverse the order of these two statements of cbar.remove and ax.clear()!
        ax.clear()
    
    # Create a magnitude array if not provided
    if mag is None:
        mag = np.hypot(U, V)

    # Create normalization if magnitude limits: mag_min and mag_max are provided
    if mag_min is not None:
        norm_ = Normalize(mag_min, mag_max)
    else:
        mag_min = mag.min(); mag_max = mag.max()
        norm_ = Normalize(mag_min, mag_max)
    
    # Plot the streamlines
    strm = ax.streamplot(X, Y, U, V, color=mag, cmap=colmp, norm=norm_, density=density)
    
    # Add a colorbar
    height, width = len(Y), len(X)
    ax.strmbar = ax.figure.colorbar(strm.lines, ax=ax, fraction=0.047 * height / width, pad=0.04, format='%.2f')
    
    # Set plot labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    # Set the title of the plot
    if title is None:
        ax.set_title(f'{mag_min:.3f}  {mag_max:.3f}', fontsize=16)
    else: 
        # ax.set_title(f'{title}: {pltdat_min:.3f}  {pltdat_max:.3f}') 
        ax.set_title(f'{title}', fontsize=16)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Set aspect ratio
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
          "\nCall the function as image_stream(X, Y, U, V) for quick use."
          "\nFor more info refer to documentation.\n")
# else:
#     print("Module image_stream imported!")