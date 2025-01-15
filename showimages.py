from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

# Load the dataset
dataset = load_dataset("Sprites20/my-image-dataset")

# Assuming the dataset has an 'image' field, which holds the image file paths or URLs
image_column = "images"  # Adjust this based on the dataset's actual image column name

# Display images one by one
for example in dataset["train"]:  # You can replace "train" with "test" or other split
    image_data = example[image_column]  # Modify if necessary to fit the dataset structure

    # Debug: Check the type and content of the image_data
    print(f"Image data type: {type(image_data)}")
    print(f"Image data content: {image_data}")

    # If image_data is a list, each element is a PIL image
    if isinstance(image_data, list):
        for image in image_data:
            # Directly show the image
            plt.imshow(image)
            plt.axis('off')  # Hide axis
            plt.show()
            input("Press Enter to display the next image...")

    else:
        # If image_data is a single PIL image
        plt.imshow(image_data)
        plt.axis('off')  # Hide axis
        plt.show()
        input("Press Enter to display the next image...")