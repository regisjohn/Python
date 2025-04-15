import numpy as np
from pympler import asizeof, muppy, summary
import pickle
import psutil
import tarfile

def findin(arr, value):
    """
    Find the index in the numpy array `arr` that is closest to the given 
    `value`.
    
    Parameters:
    arr (numpy.ndarray): The input array to search for the closest value.
    value (int or float): The value to which the closest index is sought.
    
    Returns:
    int: The index in the array `arr` that is closest to the `value`.
    
    Example:
    >>> import numpy as np
    >>> arr = np.array([1, 3, 7, 10, 15])
    >>> findin(arr, 8)
    2
    """
    closest_index = np.abs(arr - value).argmin()
    return closest_index

def getsize(*arrs):
    """
    Print the size of one or more objects in megabytes. If no arguments are 
    provided, the function will print the size of all interactive variables in 
    the IPython/Jupyter environment. Otherwise, the function will print the 
    size of each object provided as an argument. For multiple objects, separate
    them with commas.

    Parameters:
    - *arrs: One or more objects whose size is to be calculated.

    Returns:
    - None

    Example:
    >>> getsize()
    | Variable     | Memory(MB) |
    |--------------|------------|
    | var1         | 1.234e-01  |
    | var2         | 2.345e-01  |

    >>> import numpy as np
    >>> arr1 = np.zeros((1000, 1000))
    >>> arr2 = np.ones((2000, 2000))
    >>> getsize(arr1, arr2)
    Size of object is 7.629e-03 MB
    Size of object is 3.052e-02 MB
    """
    # size_in_mb = arr.nbytes / (1024 * 1024) - - old code
    views = []

    if not arrs: # Check if no arguments are provided
        try:
            ipython = get_ipython() # Check if we're in an IPython environment
            all_vars = ipython.run_line_magic("who_ls", "") # Get a list of all interactive variables
            # Print the header of the table
            print(f"| {'Variable':^12} | {'Memory(MB)':^10} |")
            print(f"|{'-'*14}|{'-'*12}|")
            # Loop over each variable & get their memory usage
            for var in all_vars: # Access variable in IPython user namespace
                obj = ipython.user_ns[var]
                if isinstance(obj, np.ndarray):
                    if obj.base is None: # Check if the array is original or copy
                        size_in_mb = asizeof.asizeof(obj)/(1024*1024)
                        print(f"| {var:^12} | {size_in_mb:^10.3e} |")
                    else:
                        views.append(var)
                else:
                    size_in_mb = asizeof.asizeof(ipython.user_ns[var])/(1024*1024)
                    print(f"| {var:^12} | {size_in_mb:^10.3e} |")
                
            if views:
                print("\nViews:")
                for view in views:
                    print(view, end=', ')

        except NameError:
            print("Function parameters cannot be empty if used outside of IPython/Jupyter environments!")
    else:
        for arr in arrs:
            if isinstance(arr, np.ndarray):
                if arr.base is None:
                    size_in_mb = asizeof.asizeof(arr)/(1024*1024)
                    print(f"Size of object is {size_in_mb:^10.3e} MB")
                else:
                    print('This is just a view!')
            else:
                size_in_mb = asizeof.asizeof(arr) / (1024 * 1024)
                print(f"Size of object is {size_in_mb:^10.3e} MB")

def pickleloader(file_path):
    """
    Load data from a pickle file.

    Parameters:
    file_path (str): The path to the pickle file.

    Returns:
    object: The data loaded from the pickle file.
    
    Example usage:
    >>> data = pickleloader('path/to/your/file.pkl')
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def print_memory_usage():
    """
    Returns the memory usage of all processes and python objects in MB using 
    psutil.

    Parameters:
    None

    Returns:
    None
    
    Example usage:
    >>> print_memory_usage()
    """

    process = psutil.Process()
    mem_info = (process.memory_info()).rss/(1024*1024)
    print(f"\nTotal memory usage: {mem_info :.2f} MB")


def tarpy(folder_name):
    """
    Archives the specified folder into a tar file.

    Parameters:
    folder_name (str): The name of the folder to archive.
   
    Example usage:
    >>> tarpy('my_folder')
    """
    out_filename = f'{folder_name}.tar'
    with tarfile.open(out_filename, 'w') as tar:
        tar.add(folder_name, arcname='.')
    print(f"Folder '{folder_name}' has been archived into '{out_filename}'!")


def update_pickle_dict(file_path, new_entries):
    """
    Load a pickle dictionary file, update its contents with new entries, and save it back.

    Args:
        file_path (str): Path to the pickle file
        new_entries (dict): Dictionary of new entries to add

    Returns:
        None

    Example usage:
    >>> update_pickle_dict('path/to/your/file.pkl', {'new_key': 'new_value'})
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Contents before update:")
    for key, value in data.items():
        print(f"{key}: {value}")
    print()

    data.update(new_entries)

    print("Contents after update:")
    for key, value in data.items():
        print(f"{key}: {value}")
    print()

    confirmation = input("Save the changes to the file? (y/n): ")
    if confirmation.lower() == 'y':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print("Changes have been saved!")
    else:
        print("Changes were not saved!") 

def set_2dfig_size(fig, arr, base_sz=6.4, cbar_space=1.2, nrows=1, ncols=1):
    # base_sz 6.4 in or divisible by 16 for best video codec compatibility
    # cbar_space of 1.2 accounts for 20% colorbar space width
    """
    Set figure size for a 2D plot based on the size of the input array and number of subplots.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object
        arr (numpy.ndarray): The 2D array to determine the size of the figure
        base_sz (float, optional): The base size of the figure in inches. Defaults to 6.4.
        cbar_space (float, optional): The width of the colorbar as a fraction of the width of the figure. Defaults to 1.2.
        nrows (int, optional): The number of rows of subplots. Defaults to 1.
        ncols (int, optional): The number of columns of subplots. Defaults to 1.

    Returns:
        None
    """
    aspect_ratio = (arr.shape[1] / arr.shape[0]) * cbar_space  # width/height per subplot
    fig_width = base_sz * aspect_ratio * ncols  # Adjust width for multiple columns
    fig_height = base_sz * nrows  # Adjust height for multiple rows
    fig.set_size_inches(fig_width, fig_height)