import os
import cv2 as cv
import numpy as np
extension_pixels = 200

def extend_image(image, extension_pixels):
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
    # Ensure input image is single-channel (grayscale)
    gray = np.where(img > 0, 255, 0).astype(np.uint8)
    #cv.imshow("gray", gray)
    mblur = cv.medianBlur(gray, 5)
    #cv.imshow("mblur", mblur)
    gblur = cv.GaussianBlur(mblur, (3, 3), 2)
    #cv.imshow("gblur", gblur)

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
            if black_percentage > 62:
                # Draw the full circle in red
                cv.circle(draw, center, radius, (0, 0, 255), 3)

                # Draw the center
                cv.circle(draw, center, 2, (0, 255, 0), 3)

                # Annotate with black pixel percentage
                cv.putText(
                    draw,
                    f"Circle {idx}: | Black Pixels: {black_percentage:.2f}% | Radius: {radius:.2f}",
                    (circle[0] - radius, circle[1] - radius - 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255), 
                    1
                )

            if 10 < black_percentage < 62:
                img_height, img_width = img.shape[:2]
                min_radius = 50
                if (center[0] + radius > img_width - extension_pixels or center[0] - radius < extension_pixels or center[1] + radius > img_height - extension_pixels or center[1] - radius < extension_pixels):
                    cv.circle(draw, center, radius, (0, 0, 255), 3)

                    # Draw the center
                    cv.circle(draw, center, 2, (0, 255, 0), 3)

                    # Annotate with black pixel percentage
                    cv.putText(
                        draw,
                        f"Circle {idx}: | Black Pixels: {black_percentage:.2f}% | Radius: {radius:.2f}",
                        (circle[0] - radius, circle[1] - radius - 20),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255), 
                        1
                    )
                #  #Below only for debugging and tuning. Displaying all circles that do not meed the black percentage requirements 
                # else:
                #         # Draw the full circle in red
                #     cv.circle(draw, center, radius, (255, 0, 0), 3)

                #     # Draw the center
                #     cv.circle(draw, center, 2, (0, 255, 0), 3)

                #     # Annotate with black pixel percentage
                #     cv.putText(
                #         draw,
                #         f"Circle {idx}: | Black Pixels: {black_percentage:.2f}% | Radius: {radius:.2f}",
                #         (circle[0] - radius, circle[1] - radius - 20),
                #         cv.FONT_HERSHEY_SIMPLEX,
                #         0.3,
                #         (255, 0, 0), 
                #         1
                #     )  


                

        return draw
    else:
        return original_image  # Return original image if no circles detected

def threshold_yuv(img):
    # Convert image to YUV color space
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    #cv.imshow("img_yuv", img_yuv)
    # Define the lower and upper bounds for YUV channels based on the specified ranges
    lower = np.array([0, 121, 0], dtype=np.uint8)
    upper = np.array([255, 153, 138], dtype=np.uint8)

    # Create a mask using inRange function to threshold YUV image
    mask = cv.inRange(img_yuv, lower, upper)
    # Apply the mask to the original image to extract the region of interest
    result = cv.bitwise_and(img, img, mask=mask)
    # Convert the result to a binary image (black and white)
    result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    #cv.imshow("result_gray",result_gray)
    _, binary_image = cv.threshold(result_gray, 1, 255, cv.THRESH_BINARY)
    #cv.imshow("binary_image",binary_image)
    return binary_image




# ---------Single Image Start-------------------------------------------------------
def main():
    # Load the pre-segmented black and white image
    filepath = os.getcwd() + "/dataset/validation/5.png"
    img = cv.imread(filepath, cv.IMREAD_COLOR)
    original_image = img

    img = extend_image(img, extension_pixels)
    original_image = extend_image(original_image, extension_pixels)

    if img is None:
        print(f"Error: Unable to read the image at '{filepath}'")
        return

    # Threshold and process the image
    binary_result = threshold_yuv(img)

    # Detect circles in the processed binary image and draw on the original image
    result_image = detect_circles(binary_result, original_image)

    if result_image is not None:
        # Display the result (convert to BGR before display)
        cv.imshow("Circles Detected", result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("No circles detected or error occurred.")

if __name__ == "__main__":
    main()

# ---------Single Image End-------------------------------------------------------



# # ---------Multi Image Start-------------------------------------------------------
# # Directory where images are located
# image_dir = "C:/Users/nikla/OneDrive/Documents/Mechatronics/282.762 Robotics/Assesment 3 - Machine Vision/dataset/validation"

# # Iterate through images "0.png" to "29.png"
# for i in range(30):
#     filename = os.path.join(image_dir, f"{i}.png")

#     # Load the image
#     img = cv.imread(filename, cv.IMREAD_COLOR)
#     original_image = img

#     if img is None:
#         print(f"Error: Unable to read the image at '{image_dir}'")
#         continue

#     img = extend_image(img, extension_pixels)
#     original_image = extend_image(original_image, extension_pixels)

#     # Threshold and process the image
#     binary_result = threshold_yuv(img)
#     result_image = detect_circles(binary_result, original_image)
#     # Save the resulting image with circles detected
#     output_filename = os.path.join(image_dir, f"{i}_circle.png")
#     cv.imwrite(output_filename, result_image)

#     print(f"Processed and saved {output_filename}")

# print("Processing complete.")

# # ---------Multi Image End-------------------------------------------------------