import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):

    # Pad the image using replicate padding.
    height, width = img.shape
    half_filter_size = filter_size // 2
    padded_height = height + 2 * half_filter_size
    padded_width = width + 2 * half_filter_size

    padded_img = np.zeros((padded_height, padded_width), dtype=img.dtype)

    for i in range(height):
        for j in range(width):
            padded_img[i + half_filter_size, j + half_filter_size] = img[i, j]
    for i in range(half_filter_size):
        padded_img[i, half_filter_size:-half_filter_size] = img[0]
        padded_img[-i - 1, half_filter_size:-half_filter_size] = img[-1]
        padded_img[half_filter_size:-half_filter_size, i] = img[:, 0]
        padded_img[half_filter_size:-half_filter_size, -i - 1] = img[:, -1]

    return padded_img

def mean_filter(img, filter_size=3):

    padded_img = padding_img(img, filter_size)
    height, width = padded_img.shape

    smoothed_img = np.zeros_like(img)

    for i in range(height - filter_size + 1):
        for j in range(width - filter_size + 1):
            window = padded_img[i:i + filter_size, j:j + filter_size]
            smoothed_img[i, j] = np.mean(window, axis=(0, 1))

    return smoothed_img

def median_filter(img, filter_size=3):

    padded_img = padding_img(img, filter_size)
    height, width = padded_img.shape

    smoothed_img = np.zeros_like(img)

    for i in range(height - filter_size + 1):
        for j in range(width - filter_size + 1):
            window = padded_img[i:i + filter_size, j:j + filter_size]
            smoothed_img[i, j] = np.median(window, axis=(0, 1))

    return smoothed_img

def psnr(gt_img, smooth_img):

    gt_img = gt_img.astype(np.float64)
    smooth_img = smooth_img.astype(np.float64)

    # Calculate the squared error (MSE)
    mse = np.mean((gt_img - smooth_img) ** 2)

    # Find the maximum possible pixel value for the given data type
    # max_pixel_value = np.iinfo(gt_img.dtype).max
    max_pixel_value = 255

    # Calculate PSNR using the formula: PSNR = 10 * log10((max_pixel_value^2) / MSE)
    psnr_score = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr_score

def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

