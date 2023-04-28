import imageio
import os
from glob import glob
from PIL import Image

# Define paths for the normal and reduce images
normal_path = 'searchs/fashionmnist/plots/EP*-normal.png'
reduce_path = 'searchs/fashionmnist/plots/EP*-reduce.png'

# Set the desired width and height for the images
width = 1500
height = 500

# Create a list of normal image file paths
normal_files = sorted(glob(normal_path))
# Create a list of reduce image file paths
reduce_files = sorted(glob(reduce_path))

# Create GIF for normal images
with imageio.get_writer('normal.gif', mode='I') as writer:
    for filename in normal_files:
        with Image.open(filename) as img:
            img = img.resize((width, height), Image.ANTIALIAS)
            image = imageio.core.util.asarray(img)
            writer.append_data(image)

# Create GIF for reduce images
with imageio.get_writer('reduce.gif', mode='I') as writer:
    for filename in reduce_files:
        with Image.open(filename) as img:
            img = img.resize((width, height), Image.ANTIALIAS)
            image = imageio.core.util.asarray(img)
            writer.append_data(image)
