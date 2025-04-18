import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

def main():
    st.title("Digital Image Processing App")
    
    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Choose a processing technique",
        ["Home", "Basic Transforms", "Filters", "Edge Detection", "Morphological Operations"]
    )
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # If image is RGBA, convert to RGB
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        # Display the original image
        st.sidebar.subheader("Original Image")
        st.sidebar.image(image, use_column_width=True)
        
        # Process based on selected option
        if option == "Home":
            display_home_page(img_array)
        elif option == "Basic Transforms":
            apply_basic_transforms(img_array)
        elif option == "Filters":
            apply_filters(img_array)
        elif option == "Edge Detection":
            apply_edge_detection(img_array)
        elif option == "Morphological Operations":
            apply_morphological_operations(img_array)
    else:
        if option == "Home":
            st.header("Welcome to the Digital Image Processing App")
            st.write("""
            This app demonstrates various digital image processing techniques.
            Upload an image using the sidebar to get started!
            
            ### Available Features:
            * Basic image transformations (grayscale, negative, thresholding)
            * Image filtering (blur, sharpening)
            * Edge detection algorithms
            * Morphological operations
            """)
        else:
            st.info("Please upload an image to use this feature")

def display_home_page(image):
    st.header("Image Information")
    st.write(f"Image Shape: {image.shape}")
    st.write(f"Image Size: {image.size}")
    st.write(f"Image Data Type: {image.dtype}")
    
    st.header("Color Channels")
    if len(image.shape) > 2 and image.shape[2] == 3:
        st.write("RGB Image")
        
        col1, col2, col3 = st.columns(3)
        
        # Red channel
        red_channel = image.copy()
        red_channel[:, :, 1] = 0  # Zero out G
        red_channel[:, :, 2] = 0  # Zero out B
        col1.header("Red Channel")
        col1.image(red_channel, use_column_width=True)
        
        # Green channel
        green_channel = image.copy()
        green_channel[:, :, 0] = 0  # Zero out R
        green_channel[:, :, 2] = 0  # Zero out B
        col2.header("Green Channel")
        col2.image(green_channel, use_column_width=True)
        
        # Blue channel
        blue_channel = image.copy()
        blue_channel[:, :, 0] = 0  # Zero out R
        blue_channel[:, :, 1] = 0  # Zero out G
        col3.header("Blue Channel")
        col3.image(blue_channel, use_column_width=True)
    else:
        st.write("Grayscale Image")

def apply_basic_transforms(image):
    st.header("Basic Image Transformations")
    
    transform_option = st.selectbox(
        "Select a transformation",
        ["Grayscale", "Negative", "Thresholding", "Brightness and Contrast", "Rotation", "Flip"]
    )
    
    if transform_option == "Grayscale":
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            st.image(gray_image, caption="Grayscale Image", use_column_width=True)
        else:
            st.write("Image is already grayscale")
            st.image(image, caption="Grayscale Image", use_column_width=True)
    
    elif transform_option == "Negative":
        negative_image = 255 - image
        st.image(negative_image, caption="Negative Image", use_column_width=True)
    
    elif transform_option == "Thresholding":
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        threshold_value = st.slider("Threshold Value", 0, 255, 128)
        _, thresh_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        st.image(thresh_image, caption=f"Thresholded Image (value={threshold_value})", use_column_width=True)
    
    elif transform_option == "Brightness and Contrast":
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", 0.0, 3.0, 1.0)
        
        # Apply brightness and contrast adjustment
        adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        st.image(adjusted_image, caption=f"Adjusted Image (brightness={brightness}, contrast={contrast})", use_column_width=True)
    
    elif transform_option == "Rotation":
        angle = st.slider("Rotation Angle", -180, 180, 0)
        
        if angle != 0:
            # Get image dimensions
            height, width = image.shape[:2]
            # Calculate the rotation matrix
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Apply the rotation
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            st.image(rotated_image, caption=f"Rotated Image (angle={angle}°)", use_column_width=True)
        else:
            st.image(image, caption="Original Image", use_column_width=True)
    
    elif transform_option == "Flip":
        flip_option = st.selectbox("Flip Direction", ["Horizontal", "Vertical", "Both"])
        
        if flip_option == "Horizontal":
            flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
            st.image(flipped_image, caption="Horizontally Flipped Image", use_column_width=True)
        elif flip_option == "Vertical":
            flipped_image = cv2.flip(image, 0)  # 0 for vertical flip
            st.image(flipped_image, caption="Vertically Flipped Image", use_column_width=True)
        elif flip_option == "Both":
            flipped_image = cv2.flip(image, -1)  # -1 for both directions
            st.image(flipped_image, caption="Flipped Image (Both Directions)", use_column_width=True)

def apply_filters(image):
    st.header("Image Filtering")
    
    filter_option = st.selectbox(
        "Select a filter",
        ["Gaussian Blur", "Median Blur", "Bilateral Filter", "Sharpening", "Box Blur"]
    )
    
    if filter_option == "Gaussian Blur":
        kernel_size = st.slider("Kernel Size (odd values only)", 1, 25, 5, step=2)
        sigma = st.slider("Sigma", 0.1, 10.0, 1.0)
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        st.image(blurred_image, caption=f"Gaussian Blur (kernel={kernel_size}x{kernel_size}, sigma={sigma})", use_column_width=True)
    
    elif filter_option == "Median Blur":
        kernel_size = st.slider("Kernel Size (odd values only)", 1, 25, 5, step=2)
        blurred_image = cv2.medianBlur(image, kernel_size)
        st.image(blurred_image, caption=f"Median Blur (kernel={kernel_size}x{kernel_size})", use_column_width=True)
    
    elif filter_option == "Bilateral Filter":
        d = st.slider("Diameter", 1, 15, 9)
        sigma_color = st.slider("Sigma Color", 1, 150, 75)
        sigma_space = st.slider("Sigma Space", 1, 150, 75)
        filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        st.image(filtered_image, caption=f"Bilateral Filter (d={d}, sigma_color={sigma_color}, sigma_space={sigma_space})", use_column_width=True)
    
    elif filter_option == "Sharpening":
        kernel_size = st.slider("Kernel Size", 3, 15, 5, step=2)
        sigma = st.slider("Sigma", 0.1, 5.0, 1.0)
        amount = st.slider("Sharpening Amount", 0.0, 5.0, 1.5)
        
        # Create a sharpened image using unsharp masking
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        
        st.image(sharpened, caption=f"Sharpened Image (amount={amount})", use_column_width=True)
    
    elif filter_option == "Box Blur":
        kernel_size = st.slider("Kernel Size", 1, 25, 5, step=2)
        # Create box filter kernel
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        blurred_image = cv2.filter2D(image, -1, kernel)
        st.image(blurred_image, caption=f"Box Blur (kernel={kernel_size}x{kernel_size})", use_column_width=True)

def apply_edge_detection(image):
    st.header("Edge Detection")
    
    edge_option = st.selectbox(
        "Select an edge detection algorithm",
        ["Sobel", "Laplacian", "Canny", "Prewitt", "Scharr"]
    )
    
    # Convert to grayscale if the image is colored
    if len(image.shape) > 2 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    if edge_option == "Sobel":
        ksize = st.slider("Kernel Size (odd values only)", 1, 7, 3, step=2)
        dx = st.slider("X derivative order", 0, 2, 1)
        dy = st.slider("Y derivative order", 0, 2, 1)
        
        # Apply Sobel operator
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, dx, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, dy, ksize=ksize)
        
        # Calculate the gradient magnitude
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        st.image(sobel_combined, caption=f"Sobel Edge Detection (kernel={ksize}x{ksize})", use_column_width=True)
    
    elif edge_option == "Laplacian":
        ksize = st.slider("Kernel Size (odd values only)", 1, 7, 3, step=2)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=ksize)
        laplacian = np.uint8(np.absolute(laplacian))
        
        st.image(laplacian, caption=f"Laplacian Edge Detection (kernel={ksize}x{ksize})", use_column_width=True)
    
    elif edge_option == "Canny":
        low_threshold = st.slider("Low Threshold", 0, 255, 50)
        high_threshold = st.slider("High Threshold", 0, 255, 150)
        
        canny_edges = cv2.Canny(gray_image, low_threshold, high_threshold)
        
        st.image(canny_edges, caption=f"Canny Edge Detection (low={low_threshold}, high={high_threshold})", use_column_width=True)
    
    elif edge_option == "Prewitt":
        # Custom implementation of Prewitt operator
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        
        prewitt_x = cv2.filter2D(gray_image, -1, kernelx)
        prewitt_y = cv2.filter2D(gray_image, -1, kernely)
        
        prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
        
        st.image(prewitt, caption="Prewitt Edge Detection", use_column_width=True)
    
    elif edge_option == "Scharr":
        # Apply Scharr operator
        scharr_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
        
        # Calculate the gradient magnitude
        scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
        scharr_combined = cv2.normalize(scharr_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        st.image(scharr_combined, caption="Scharr Edge Detection", use_column_width=True)

def apply_morphological_operations(image):
    st.header("Morphological Operations")
    
    morph_option = st.selectbox(
        "Select a morphological operation",
        ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]
    )
    
    # Convert to binary image if colored
    if len(image.shape) > 2 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold_value = st.slider("Threshold Value", 0, 255, 128)
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        threshold_value = st.slider("Threshold Value", 0, 255, 128)
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel_size = st.slider("Kernel Size", 1, 25, 5)
    iterations = st.slider("Iterations", 1, 10, 1)
    
    # Create kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if morph_option == "Erosion":
        result = cv2.erode(binary_image, kernel, iterations=iterations)
        st.image(result, caption=f"Erosion (kernel={kernel_size}x{kernel_size}, iterations={iterations})", use_column_width=True)
    
    elif morph_option == "Dilation":
        result = cv2.dilate(binary_image, kernel, iterations=iterations)
        st.image(result, caption=f"Dilation (kernel={kernel_size}x{kernel_size}, iterations={iterations})", use_column_width=True)
    
    elif morph_option == "Opening":
        result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        st.image(result, caption=f"Opening (kernel={kernel_size}x{kernel_size}, iterations={iterations})", use_column_width=True)
    
    elif morph_option == "Closing":
        result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        st.image(result, caption=f"Closing (kernel={kernel_size}x{kernel_size}, iterations={iterations})", use_column_width=True)
    
    elif morph_option == "Gradient":
        result = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
        st.image(result, caption=f"Morphological Gradient (kernel={kernel_size}x{kernel_size})", use_column_width=True)
    
    elif morph_option == "Top Hat":
        result = cv2.morphologyEx(binary_image, cv2.MORPH_TOPHAT, kernel)
        st.image(result, caption=f"Top Hat (kernel={kernel_size}x{kernel_size})", use_column_width=True)
    
    elif morph_option == "Black Hat":
        result = cv2.morphologyEx(binary_image, cv2.MORPH_BLACKHAT, kernel)
        st.image(result, caption=f"Black Hat (kernel={kernel_size}x{kernel_size})", use_column_width=True)

if __name__ == "__main__":
    main()