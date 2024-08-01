# Install the required packages
import os
from PIL import Image


# Define the parameters for cropping based on parameters in Edge Impulse
def process_image(input_image_path, output_image_path, crop_size=200, resize_size=96, convert_to_grayscale=True):
    with Image.open(input_image_path) as img:
        # Crop to square
        width, height = img.size
        new_size = min(width, height)
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        img_cropped = img.crop((left, top, right, bottom))
        
        # Resize to the desired size for the ML model
        img_resized = img_cropped.resize((resize_size, resize_size))
        
        # Convert to grayscale if needed
        if convert_to_grayscale:
            img_resized = img_resized.convert('L')
        
        # Save the processed image
        img_resized.save(output_image_path)

# Define the function to process all images in a directory
def batch_process_images(input_directory, output_directory, crop_size=200, resize_size=96, convert_to_grayscale=True):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through the 'Birds' folder inside input_directory
    birds_dir = input_directory  
    for filename in os.listdir(birds_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(birds_dir, filename)
            output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_{resize_size}x{resize_size}.jpg")
            process_image(input_path, output_path, crop_size, resize_size, convert_to_grayscale)
            print(f"Processed {filename} -> {output_path}")


# Select the path of your input directory
input_dir = r'C:\Users\jessi\Documents\UNI\Master\2. Semester\IoT & AI on the Edge\Watchgarden_Dummy_Birds\Unkown'

# Select the path of your output directory
output_dir = r'C:\Users\jessi\Documents\UNI\Master\2. Semester\IoT & AI on the Edge\Watchgarden_Dummy_Birds\Unkown_preprocessed'


# Process the images
batch_process_images(input_dir, output_dir)

