
# YOLOv8 Object Detection, Cropping, and Classification

This repository contains a set of Python scripts for detecting objects using YOLOv8, capturing images of unknown objects, cropping those images based on detected bounding boxes, and automatically classifying the cropped images using clustering.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Object Detection and Capture](#object-detection-and-capture)
  - [Cropping Images](#cropping-images)
  - [Automatic Classification](#automatic-classification)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The repository includes three main scripts:

1. `object_detector.py`: Detects objects in real-time using YOLOv8, captures images of unknown objects, and saves them along with metadata.
2. `crop_images.py`: Crops the captured images based on the bounding boxes stored in the metadata.
3. `auto_classifier.py`: Automatically classifies the cropped images using clustering and user input for labeling.

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy
- scikit-learn
- tqdm
- ultralytics (for YOLOv8)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/0xroyce/yolov8-dct.git
    cd yolov8-dct
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Object Detection and Capture

Run `object_detector.py` to start detecting objects and capturing images of unknown objects.

```sh
python object_detector.py
```

This script uses your webcam to detect objects in real-time. Known objects are highlighted in green, while unknown objects are highlighted in red and captured for further processing.

### Cropping Images

Run `crop_images.py` to crop the captured images based on the bounding boxes stored in the metadata.

```sh
python crop_images.py
```

This script processes the images in the `unknown_objects` directory and saves the cropped images in the `cropped_unknown_objects` directory.

### Automatic Classification

Run `auto_classifier.py` to classify the cropped images using clustering.

```sh
python auto_classifier.py
```

This script uses KMeans clustering to group similar images together. You will be prompted to label each cluster, and the images will be renamed accordingly.

## Directory Structure

- `unknown_objects/`: Contains the images and metadata of unknown objects captured by `object_detector.py`.
- `cropped_unknown_objects/`: Contains the cropped images processed by `crop_images.py`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
