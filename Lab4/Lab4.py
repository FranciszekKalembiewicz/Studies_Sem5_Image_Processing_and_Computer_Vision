import random
import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

def nothing(x):
    pass

def salt_pepper(img, tresh):
    row, col, ch = img.shape
    noisy = img.copy()
    p_noise = tresh / 100

    num_salt = int(p_noise * row * col)
    num_pepper = int(p_noise * row * col)

    salt_coords = [(random.randint(0, row - 1), random.randint(0, col - 1)) for _ in range(num_salt)]
    pepper_coords = [(random.randint(0, row - 1), random.randint(0, col - 1)) for _ in range(num_pepper)]

    for y, x in salt_coords:
        noisy[y, x] = [255, 255, 255]
    for y, x in pepper_coords:
        noisy[y, x] = [0, 0, 0]

    return noisy


def gaussian_noise(img, sigma_value):
    noise = np.random.normal(0, sigma_value / 2.0, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def mean_filter(img, k):
    return cv2.blur(img, (k, k))

def median_filter(img, k):
    return cv2.medianBlur(img, k)

def gaussian_filter(img, k):
    sigma = 5
    return cv2.GaussianBlur(img, (k, k), sigma)



cv2.namedWindow("Original")
cv2.namedWindow("Noise")
cv2.namedWindow("Filter")

cv2.createTrackbar("Noise Type", "Noise", 0, 1, nothing)
cv2.createTrackbar("Noise %", "Noise", 0, 100, nothing)

cv2.createTrackbar("Filter Type", "Filter", 0, 2, nothing)
cv2.createTrackbar("Kernel", "Filter", 1, 25, nothing)

prev_noise_type = -1
prev_noise_level = -1
noisy = img.copy()

while True:
    cv2.imshow("Original", img)

    noise_type = cv2.getTrackbarPos("Noise Type", "Noise")
    noise_level = cv2.getTrackbarPos("Noise %", "Noise")

    filter_type = cv2.getTrackbarPos("Filter Type", "Filter")
    k = cv2.getTrackbarPos("Kernel", "Filter")

    if k <= 0:
        k = 1
    if k % 2 == 0:
        k += 1

    if noise_type != prev_noise_type or noise_level != prev_noise_level:
        if noise_type == 0:
            noisy = salt_pepper(img, noise_level)
        else:
            noisy = gaussian_noise(img, noise_level)
        prev_noise_type = noise_type
        prev_noise_level = noise_level


    if filter_type == 0:
        filtered = mean_filter(noisy, k)
    elif filter_type == 1:
        filtered = median_filter(noisy, k)
    else:
        filtered = gaussian_filter(noisy, k)


    cv2.imshow("Noise", noisy)
    cv2.imshow("Filter", filtered)

    cv2.waitKey(30)