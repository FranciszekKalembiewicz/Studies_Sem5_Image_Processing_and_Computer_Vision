import random
import cv2
import numpy as np


def bilinear_resize(image, scale):
    h, w = image.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)

    result = np.zeros((new_h, new_w, image.shape[2]), dtype=np.float32)

    for y_new in range(new_h):
        y = y_new / scale
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, h - 1)
        dy = y - y0

        for x_new in range(new_w):
            x = x_new / scale
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, w - 1)
            dx = x - x0

            p00 = image[y0, x0].astype(np.float32)
            p10 = image[y0, x1].astype(np.float32)
            p01 = image[y1, x0].astype(np.float32)
            p11 = image[y1, x1].astype(np.float32)

            top = (1 - dx) * p00 + dx * p10
            bottom = (1 - dx) * p01 + dx * p11
            value = (1 - dy) * top + dy * bottom

            result[y_new, x_new] = value

    return result.astype(np.uint8)

def nothing(x):
    pass

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

cv2.namedWindow("Original")
cv2.namedWindow("Resized")

cv2.createTrackbar("Scale", "Original", 10, 30, nothing)
cv2.createTrackbar("Mode", "Original", 0, 4, nothing)

prev_scale = -1
prev_mode = -1
resised = img.copy()

while True:
    cv2.imshow("Original", img)

    Scale_raw = max(1, cv2.getTrackbarPos("Scale", "Original"))
    Mode = cv2.getTrackbarPos("Mode", "Original")

    Scale = Scale_raw / 10.0
    if Scale != prev_scale or Mode != prev_mode:
        if Mode == 0:
            resised = cv2.resize(img, None, fx=Scale, fy=Scale, interpolation=cv2.INTER_NEAREST)
        elif Mode == 1:
            resised = cv2.resize(img, None, fx=Scale, fy=Scale, interpolation=cv2.INTER_LINEAR)
        elif Mode == 2:
            resised = cv2.resize(img, None, fx=Scale, fy=Scale, interpolation=cv2.INTER_CUBIC)
        elif Mode == 3:
            resised = cv2.resize(img, None, fx=Scale, fy=Scale, interpolation=cv2.INTER_LANCZOS4)
        elif Mode == 4:
            resised = bilinear_resize(img, Scale)

        prev_scale = Scale
        prev_mode = Mode

        resized_back = cv2.resize(resised, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        diff = cv2.absdiff(img, resized_back)
        cv2.imshow("Difference", diff)

    cv2.imshow("Resized", resised)

    cv2.waitKey(30)

cv2.destroyAllWindows()
