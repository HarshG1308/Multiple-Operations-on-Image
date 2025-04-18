# Digital Image Processing Application

![DIP App](https://img.shields.io/badge/Streamlit-DIP%20App-FF4B4B?style=for-the-badge&logo=streamlit)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

A powerful digital image processing application built with Python and Streamlit that allows users to upload images and apply various image processing techniques such as filters, edge detection, and morphological operations.

## Features

This application offers a comprehensive suite of image processing tools:

### Basic Image Transformations
- **Grayscale Conversion**: Convert color images to grayscale
- **Negative Transformation**: Invert pixel values to create negative images
- **Thresholding**: Convert grayscale images to binary based on threshold values
- **Brightness & Contrast**: Adjust image brightness and contrast
- **Rotation**: Rotate images at specified angles
- **Flipping**: Flip images horizontally, vertically, or both

### Image Filtering
- **Gaussian Blur**: Blur images using Gaussian filter with adjustable parameters
- **Median Blur**: Apply median filter to remove noise while preserving edges
- **Bilateral Filter**: Edge-preserving noise reduction filter
- **Sharpening**: Enhance image details using unsharp masking
- **Box Blur**: Simple averaging filter for blurring

### Edge Detection
- **Sobel Operator**: Emphasize edges using gradient-based method
- **Laplacian**: Second derivative-based edge detection
- **Canny Edge Detector**: Multi-stage edge detection algorithm
- **Prewitt Operator**: Simple gradient-based edge detector
- **Scharr Operator**: Improved alternative to Sobel for better rotation invariance

### Morphological Operations
- **Erosion & Dilation**: Basic morphological operations
- **Opening & Closing**: Remove noise and close small holes
- **Gradient**: Outline of objects in the image
- **Top Hat & Black Hat**: Extract fine details and enhance dark regions

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dip-streamlit-app.git
   cd dip-streamlit-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Requirements

The application requires the following Python packages:
```
streamlit
numpy
opencv-python-headless
Pillow
```

## Deployment on Streamlit Cloud

To deploy this application on Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub
3. Select your forked repository and the main file (`app.py`)
4. Click "Deploy"

## Usage Guide

1. **Upload an Image**: Use the file uploader in the sidebar to upload an image
2. **Select Processing Technique**: Choose from the sidebar menu (Basic Transforms, Filters, Edge Detection, or Morphological Operations)
3. **Adjust Parameters**: Use the sliders and other controls to fine-tune the processing parameters
4. **View Results**: See the processed image displayed in real-time as you adjust parameters

## Acknowledgements

- [OpenCV](https://opencv.org/) for the image processing capabilities
- [Streamlit](https://streamlit.io/) for the web application framework
- [NumPy](https://numpy.org/) for numerical computing
- [Pillow](https://python-pillow.org/) for image handling

---

## Author
- Harsh Gautam
