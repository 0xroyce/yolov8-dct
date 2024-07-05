# crop_images.py

import os
import json
import cv2
from tqdm import tqdm


class ImageCropper:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def crop_image(self, image_path, metadata_path):
        # Read the image
        image = cv2.imread(image_path)

        # Read the metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Extract bounding box coordinates
        box = metadata['box']
        x1, y1, x2, y2 = map(int, box)

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        return cropped_image

    def process_images(self):
        for root, dirs, files in os.walk(self.input_dir):
            for file in tqdm(files):
                if file.endswith('.jpg'):
                    image_path = os.path.join(root, file)
                    metadata_path = image_path.replace('.jpg', '_metadata.json')

                    if os.path.exists(metadata_path):
                        try:
                            cropped_image = self.crop_image(image_path, metadata_path)

                            # Create corresponding directory structure in output_dir
                            relative_path = os.path.relpath(root, self.input_dir)
                            output_subdir = os.path.join(self.output_dir, relative_path)
                            os.makedirs(output_subdir, exist_ok=True)

                            # Save the cropped image
                            output_path = os.path.join(output_subdir, file)
                            cv2.imwrite(output_path, cropped_image)
                        except Exception as e:
                            print(f"Error processing {image_path}: {str(e)}")
                    else:
                        print(f"Metadata file not found for {image_path}")


if __name__ == "__main__":
    input_directory = "unknown_objects"
    output_directory = "cropped_unknown_objects"

    cropper = ImageCropper(input_directory, output_directory)
    cropper.process_images()
    print("Cropping complete. Cropped images are saved in:", output_directory)