import os
import cv2 as cv
import numpy as np

extension_pixels = 200
debug_state = True      # Set to False for normal operation, True for debugging mode
output_mode = 0         # Set to 0 for single image, 1 for multiple images
image_number = 1        # Set number to whichever required for single image output
image_count = 30        # Set number to however many images are in database to be analysed

def extend_image(image, extension_pixels):
    """
    Extends the borders of an image by a specified number of pixels, filling the extended area with white.
    
    Parameters:
        image (numpy.ndarray): The original image as a NumPy array, typically in shape (height, width, channels).
        extension_pixels (int): The number of pixels to add to each side of the image.

    Returns:
        numpy.ndarray: A new image with extended borders, where the original image is centered and the added borders are white.      
    """
    # Determine new dimensions
    height, width = image.shape[:2]
    new_height = height + 2 * extension_pixels
    new_width = width + 2 * extension_pixels
    
    # Create a new blank image with white pixels
    extended_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255  # White image
    
    # Calculate position to paste original image
    x_offset = extension_pixels
    y_offset = extension_pixels
    
    # Paste the original image onto the new blank image
    extended_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
    
    return extended_image

def detect_circles(img, original_image):
    """
    Detects circles in a grayscale image using the Hough Circle Transform and overlays valid circles 
    on the original image. Circles are filtered based on the percentage of black pixels in their region.

    Parameters:
        img (numpy.ndarray): Grayscale input image for circle detection.
        original_image (numpy.ndarray): Original color image for overlaying detected circles.

    Returns:
        numpy.ndarray: Annotated image with valid circles, or the original image if no circles are detected.
    """
    # Variables for adjusting the black percentage within a circle to be counted as an apple
    max_black_percentage = 62       # Working: 62
    min_black_percentage = 10       # Working: 10
    circle_thickness = 3            # Working: 3
    medianblur_kernel = 5           # Working: 5
    gaussianblur_kernel = 3         # Working: 3
    gaussian_standarddev = 2        # Working: 2

    # Ensure input image is single-channel (grayscale)
    gray = np.where(img > 0, 255, 0).astype(np.uint8)
    mblur = cv.medianBlur(gray, medianblur_kernel)
    gblur = cv.GaussianBlur(mblur, (gaussianblur_kernel, gaussianblur_kernel), gaussian_standarddev)

    # Apply Hough Circle Transform with adjusted parameters
    circles = cv.HoughCircles(
        gblur,
        cv.HOUGH_GRADIENT,
        dp=0.95,
        minDist=75,
        param1=50,
        param2=10,
        minRadius=75,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        text_position_offset = 20

        # Draw detected circles on the original image
        draw = original_image.copy()  # Use original image directly for drawing
        for idx, circle in enumerate(circles[0, :], start=1):
            center = (circle[0], circle[1])
            radius = circle[2]

            # Calculate the bounding box for the circle ROI
            x1 = max(0, circle[0] - radius)
            y1 = max(0, circle[1] - radius)
            x2 = min(original_image.shape[1], circle[0] + radius)
            y2 = min(original_image.shape[0], circle[1] + radius)

            # Extract ROI around the circle (considering boundary conditions)
            roi = gray[y1:y2, x1:x2]

            # Calculate percentage of black pixels (apples) within the ROI
            black_percentage = np.mean(roi < 128) * 100  # Assuming threshold of 128 for binary

            # Draw circle only if majority of pixels are black (apples)
            if black_percentage > max_black_percentage:
                # Draw the full circle in red
                cv.circle(draw, center, radius, (0, 0, 255), circle_thickness)

                # Draw the center
                cv.circle(draw, center, 2, (0, 255, 0), circle_thickness)

                # Annotate with black pixel percentage
                cv.putText(
                    draw,
                    f"Circle {idx}: | Black Pixels: {black_percentage:.2f}% | Radius: {radius:.2f}",
                    (circle[0] - radius, circle[1] - radius - text_position_offset),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255), 
                    1
                )

            # Highlight circles within the black percentage range and near image borders
            if min_black_percentage < black_percentage < max_black_percentage:
                img_height, img_width = img.shape[:2]

                if (center[0] + radius > img_width - extension_pixels or center[0] - radius < extension_pixels or center[1] + radius > img_height - extension_pixels or center[1] - radius < extension_pixels):
                    cv.circle(draw, center, radius, (0, 0, 255), circle_thickness)

                    # Draw the center
                    cv.circle(draw, center, 2, (0, 255, 0), circle_thickness)

                    # Annotate with black pixel percentage
                    cv.putText(
                        draw,
                        f"Circle {idx}: | Black Pixels: {black_percentage:.2f}% | Radius: {radius:.2f}",
                        (circle[0] - radius, circle[1] - radius - text_position_offset),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255), 
                        1
                    )

            # Below only for debugging and tuning. Displaying all circles that do not meed the black percentage requirements and outputs different stages of filtering
            if debug_state == True:
                # Draw the different filtered layers
                cv.imshow("gray", gray)
                cv.imshow("mblur", mblur)
                cv.imshow("gblur", gblur)

            if black_percentage < min_black_percentage:
                # Draw the full circle in blue
                cv.circle(draw, center, radius, (255, 0, 0), circle_thickness)

                # Draw the center
                cv.circle(draw, center, 2, (0, 255, 0), circle_thickness)

                # Annotate with black pixel percentage
                cv.putText(
                    draw,
                    f"Circle {idx}: | Black Pixels: {black_percentage:.2f}% | Radius: {radius:.2f}",
                    (circle[0] - radius, circle[1] - radius - text_position_offset),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )

        return draw
    else:
        return original_image  # Return original image if no circles detected

def threshold_yuv(img):
    """
    Thresholds an image in the YUV color space to isolate a specific color range and returns a binary mask.

    Parameters:
        img (numpy.ndarray): Input BGR image.

    Returns:
        numpy.ndarray: Binary image highlighting the thresholded region.
    """

    # Convert image to YUV color space
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    # Define the lower and upper bounds for YUV channels based on the specified ranges
    lower = np.array([0, 121, 0], dtype=np.uint8)
    upper = np.array([255, 153, 138], dtype=np.uint8)

    # Create a mask using inRange function to threshold YUV image
    mask = cv.inRange(img_yuv, lower, upper)

    # Apply the mask to the original image to extract the region of interest
    result = cv.bitwise_and(img, img, mask=mask)

    # Convert the result to a binary image (black and white)
    result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(result_gray, 1, 255, cv.THRESH_BINARY)

    # Below only for debugging and tuning. Displaying different image conversion stages
    if(debug_state==True):
        cv.imshow("img_yuv", img_yuv)
        cv.imshow("result_gray",result_gray)
        cv.imshow("binary_image",binary_image)

    return binary_image

def process_single_image():
    """
    Processes a single image by extending its borders, applying a YUV threshold, and detecting circles.
    
    Steps include loading the image, extending its borders, thresholding in the YUV color space, and detecting 
    circles to overlay on the original image. Displays the final result or an error message if no circles are detected.

    Returns:
        None
    """

    # Single image processing
    filepath = os.path.join(os.getcwd(), "validation", f"{image_number}.png")
    img = cv.imread(filepath, cv.IMREAD_COLOR)
    original_image = img

    if img is None:
        print(f"Error: Unable to read the image at '{filepath}'")
        return

    img = extend_image(img, extension_pixels)
    original_image = extend_image(original_image, extension_pixels)

    # Threshold and process the image
    binary_result = threshold_yuv(img)
    result_image = detect_circles(binary_result, original_image)

    if result_image is not None:
        # Display the result
        cv.imshow("Circles Detected", result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("No circles detected or error occurred.")

def process_multiple_images():
    """
    Processes multiple images by extending borders, applying a YUV threshold, and detecting circles.

    Loads images from the "validation" directory, applies image processing steps, and saves the 
    results with detected circles as new files. 
    Name images "0" up to "n" number to be read in without modifying the code.

    Returns:
        None
    """

    # Multi-image processing
    filepath = os.getcwd()
    image_dir = os.path.join(filepath, "validation")

    for i in range(image_count):
        filename = os.path.join(image_dir, f"{i}.png")

        # Load the image
        img = cv.imread(filename, cv.IMREAD_COLOR)
        original_image = img

        if img is None:
            print(f"Error: Unable to read the image at '{filename}'")
            continue

        img = extend_image(img, extension_pixels)
        original_image = extend_image(original_image, extension_pixels)

        # Threshold and process the image
        binary_result = threshold_yuv(img)
        result_image = detect_circles(binary_result, original_image)

        # Save the resulting image with circles detected
        output_filename = os.path.join(image_dir, f"{i}_circle.png")
        cv.imwrite(output_filename, result_image)

        print(f"Processed and saved {output_filename}")

    print("Processing complete.")

def main():
    if output_mode == 0:
        process_single_image()
    elif output_mode == 1:
        process_multiple_images()
    else:
        print("Invalid output_mode. Please use 0 for single image or 1 for multiple images.")

if __name__ == "__main__":
    main()
