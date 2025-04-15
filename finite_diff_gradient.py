import numpy as np

def fd_gradient(f, dx=1.0, dy=1.0, dz=1.0):
    """
    Compute the gradient of a 3D scalar field using central differences in the interior
    and one-sided differences on the edges.
    
    Parameters:
    f  -- 3D numpy array representing the scalar field.
    dx, dy, dz -- Spacing between points in the x, y, and z directions.
    
    Returns:
    dfx_dx, dfx_dy, dfx_dz -- Gradients of the field along x, y, and z directions.
    """
    # Get the shape of the 3D array
    nx, ny, nz = f.shape

    # Initialize the gradients using np.empty for faster initialization
    dfx_dx = np.empty_like(f)
    dfx_dy = np.empty_like(f)
    dfx_dz = np.empty_like(f)

    # Compute the gradient along the x-direction (axis=0)
    # Central difference for interior points
    dfx_dx[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * dx)
    # Forward difference for the first point, backward difference for the last
    dfx_dx[0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
    dfx_dx[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx

    # Compute the gradient along the y-direction (axis=1)
    dfx_dy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * dy)
    dfx_dy[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
    dfx_dy[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy

    # Compute the gradient along the z-direction (axis=2)
    dfx_dz[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dz)
    dfx_dz[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
    dfx_dz[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz

    return dfx_dx, dfx_dy, dfx_dz