
from PIL import Image
import os

# Function to resize images while maintaining aspect ratio
def resize_image(input_path, output_path, base_width):
    img = Image.open(input_path)
    w_percent = base_width / float(img.size[0])
    h_size = int(float(img.size[1]) * float(w_percent))
    img = img.resize((base_width, h_size), Image.ANTIALIAS)
    img.save(output_path)

# Specify the directory containing the original images and where to save the resized images
input_directory = 'result/ng'  # Replace with your actual input directory path
output_directory = 'result/nng'  # Replace with your actual output directory path

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Base width in pixels for resizing
base_width = 512

# Loop through each file in the input directory and resize it
for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        resize_image(input_path, output_path, base_width)

# Output a message indicating that the resizing is complete
print("Image resizing is complete. Resized images are saved in {}".format(output_directory))
