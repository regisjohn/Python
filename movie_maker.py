import os
import glob
import imageio.v3 as imageio
from tqdm import tqdm

# Get current working directory
current_dir = os.getcwd()
png_files = sorted(glob.glob('*.png')) # sort & read all .png files
print(f"Found {len(png_files)} png files.")

if not png_files:
    print("No PNG files found!")
else:
    # Read the images and output folder
    imframes = [imageio.imread(file) for file in tqdm(png_files, desc="Reading png files...")]
    parent_dir = os.path.dirname(current_dir)
    output_dir_vid = f'{parent_dir}/frame_vid'
    os.makedirs(output_dir_vid, exist_ok=True) # create output folder 
    output_name = os.path.basename(current_dir) + '.mp4'

    # Create and save video
    print('Creating video, hold your horses!...')
    imageio.imwrite(f'{output_dir_vid}/{output_name}', imframes, fps=30)
    print(f"Created video: '{output_name}' at '{output_dir_vid}/'")
