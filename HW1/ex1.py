import cv2
import matplotlib.pyplot as plt
import numpy as np


# Load an image from file as function
def load_image(image_path):
    """Loads an image from a specified path using OpenCV."""
    return cv2.imread(image_path)  # Use cv2.imread to load the image

# Display an image as function
def display_image(image, title="Image"):
    """Displays an image using matplotlib."""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.title(title)
    plt.axis('off')
    plt.show()  # Show the image

# Convert an image to grayscale as function
def grayscale_image(image):
    # Get the image shape
    rows, cols, channels = image.shape

    # Create an empty array for the grayscale image with the same shape
    img_gray = np.zeros((rows, cols), dtype=np.uint8)  # Adjust dtype if needed

    # Iterate through each pixel and convert to grayscale
    for row in range(rows):
        for col in range(cols):
            # Get the pixel values for each channel
            r, g, b = image[row, col]

            # Apply the grayscale formula
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b

            # Set the grayscale value for all channels in the new image
            img_gray[row, col] = gray_value

    return img_gray

# Save an image as function
def save_image(image, output_path):
    """Saves an image to a specified path using OpenCV."""
    cv2.imwrite(output_path, image)  # Use cv2.imwrite to save the image

# Flip an image horizontally as function
def flip_image(image):
    """Flips an image horizontally using OpenCV."""
    return cv2.flip(image, 1)  # Use cv2.flip for horizontal flipping

# Rotate an image as function
def rotate_image(image, angle):
    """Rotates an image using OpenCV."""
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))  # Use cv2.warpAffine for rotation

if __name__ == "__main__":
    # Load an image from file
    img = load_image("uet.png")

    # Display the original image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/uet_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/uet_gray_rotated.jpg")
