import os
import shutil
import pandas as pd
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from huggingface_hub import HfApi, HfFolder
from datasets import Dataset, Image as HfImage, Sequence
from datetime import datetime

DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CSV_FILE = os.path.join(DATASET_DIR, "metadata.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)

class DragDropApp(App):
    def build(self):
        self.root = FloatLayout()

        # Instructions Label
        self.instructions = Label(
            text="Drag and drop an image here, then add a description.",
            size_hint=(0.8, 0.1),
            pos_hint={"x": 0.1, "y": 0.85}
        )
        self.root.add_widget(self.instructions)

        # Drop Area
        self.drop_area = BoxLayout(
            orientation='vertical',
            size_hint=(0.8, 0.6),
            pos_hint={"x": 0.1, "y": 0.2},
            padding=10,
            spacing=10
        )
        self.drop_area.add_widget(Label(text="Drop Image Here", size_hint=(1, 0.1)))
        self.root.add_widget(self.drop_area)

        # Text Input for Image Description
        self.text_input = TextInput(
            hint_text="Enter a description for the image...",
            size_hint=(0.8, 0.1),
            pos_hint={"x": 0.1, "y": 0.1}
        )
        self.root.add_widget(self.text_input)

        # Submit Button
        self.submit_button = Button(
            text="Submit",
            size_hint=(0.2, 0.1),
            pos_hint={"x": 0.1, "y": 0.02}
        )
        self.submit_button.bind(on_press=self.process_input)
        self.root.add_widget(self.submit_button)

        # Upload Button
        self.upload_button = Button(
            text="Upload to Hugging Face",
            size_hint=(0.4, 0.1),
            pos_hint={"x": 0.6, "y": 0.02}
        )
        self.upload_button.bind(on_press=self.upload_to_huggingface)
        self.root.add_widget(self.upload_button)

        Window.bind(on_dropfile=self.on_file_drop)
        self.current_image_path = None

        return self.root

    def on_file_drop(self, window, file_path):
        file_path = file_path.decode("utf-8")
        if file_path.endswith((".png", ".jpg", ".jpeg")):
            self.display_image(file_path)
        else:
            self.instructions.text = "Unsupported file type. Please drop an image."

    def display_image(self, file_path):
        self.current_image_path = file_path
        self.drop_area.clear_widgets()

        # Create the Image widget
        image = Image()
        image.source = file_path  # Set the image source here

        # Set the size_hint properties
        image.size_hint = (1, 0.8)

        # Add the image and a label to the drop area
        self.drop_area.add_widget(image)
        self.drop_area.add_widget(Label(text="Image Loaded", size_hint=(1, 0.2)))

        # Update the instructions text
        self.instructions.text = "Enter a description for the image."

    def process_input(self, instance):
        if not self.current_image_path:
            self.instructions.text = "Please drop an image first."
            return

        description = self.text_input.text.strip()
        if description:
            self.save_image_and_metadata(self.current_image_path, description)
            self.instructions.text = f"Image and description saved to {CSV_FILE}!"
            self.clear_inputs()
        else:
            self.instructions.text = "Please enter a description for the image."

    def save_image_and_metadata(self, image_path, description):
        # Add timestamp to the image name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_name = f"{timestamp}_{os.path.basename(image_path)}"
        dest_path = os.path.join(IMAGES_DIR, image_name)
        shutil.copy(image_path, dest_path)

        # Use the relative path with timestamp
        relative_path = f"images/{image_name}"
        data = {"Image_Path": [relative_path], "Description": [description]}
        df = pd.DataFrame(data)

        # Append to or create the CSV file
        if os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(CSV_FILE, mode="w", header=True, index=False)

    def clear_inputs(self):
        self.drop_area.clear_widgets()
        self.drop_area.add_widget(Label(text="Drop Image Here", size_hint=(1, 0.1)))
        self.text_input.text = ""
        self.current_image_path = None

    def upload_to_huggingface(self, instance):
        dataset_name = "my-image-dataset"  # Change to your dataset name
        hf_token = HfFolder.get_token()

        if not hf_token:
            self.instructions.text = "Hugging Face token not found. Log in first."
            return

        try:
            # Fetch image paths and descriptions from the CSV file
            df = pd.read_csv(CSV_FILE)
            img_paths = df["Image_Path"].tolist()
            descriptions = df["Description"].tolist()

            # Create dataset from dict
            dataset = Dataset.from_dict({"messages": descriptions})
            dataset = dataset.add_column("images", [[f"dataset/{img_path}"] for img_path in img_paths])  # Add image paths
            dataset = dataset.cast_column("images", Sequence(HfImage()))  # Convert paths to Image type

            # Get the Hugging Face username
            api = HfApi()
            user_name = api.whoami()['name']  # Fetch the username

            # Push dataset to Hugging Face Hub using the username
            dataset.push_to_hub(f"{user_name}/{dataset_name}")  # Use the username for the dataset repo
            
            self.instructions.text = f"Dataset successfully uploaded to Hugging Face repo: {dataset_name}!"

        except Exception as e:
            self.instructions.text = f"Error uploading dataset: {e}"
            print(f"Error uploading dataset: {e}")

if __name__ == "__main__":
    DragDropApp().run()
