from PIL import Image, ImageDraw
import os


def trim_image_ellipse(image_path, output_path):
    # Open the image
    image = Image.open(image_path)

    # Create a mask in the shape of an ellipse
    mask = Image.new("L", image.size, 0)
    width, height = image.size
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, width, height), fill=255)

    # Apply the mask to the image
    trimmed_image = Image.new("RGBA", image.size)
    trimmed_image.paste(image, mask=mask)

    # Save the trimmed image as PNG format
    trimmed_image.save(output_path, format='PNG')


def trim_images_in_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.endswith((".png", ".jpg", ".jpeg")):
            # Construct the input and output paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Trim the image and save the result
            trim_image_ellipse(input_path, output_path)


# Usage example
input_folder = os.path.join(os.getcwd(), "images")
output_folder = os.path.join(os.getcwd(), "trimmed")
trim_images_in_folder(input_folder, output_folder)
