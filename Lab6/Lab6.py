import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

def otsu_threshold(image):
    if len(image.shape) == 3:
        image = np.array(Image.fromarray(image).convert('L'))
    else:
        image = np.array(image)

    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist = hist.astype(float)
    total = image.size

    p = hist / total

    max_sigma = 0
    threshold = 0

    for k in range(256):
        w0 = np.sum(p[:k + 1])
        w1 = np.sum(p[k + 1:])
        if w0 == 0 or w1 == 0:
            continue
        mu0 = np.sum(np.arange(0, k + 1) * p[:k + 1]) / w0
        mu1 = np.sum(np.arange(k + 1, 256) * p[k + 1:]) / w1
        sigma_b = w0 * w1 * (mu1 - mu0) ** 2
        if sigma_b > max_sigma:
            max_sigma = sigma_b
            threshold = k
    return threshold


def apply_threshold(image, thresh):
    if len(image.shape) == 3:
        image = np.array(Image.fromarray(image).convert('L'))
    return (image > thresh).astype(np.uint8) * 255

def watershed_segmentation(binary_image):
    mask = binary_image > 0
    distance = ndi.distance_transform_edt(mask)
    local_max = ndi.maximum_filter(distance, size=20) == distance
    markers, _ = ndi.label(local_max)
    labels = ndi.watershed_ift(binary_image.astype(np.uint8), markers)
    return labels

def process_image(path):
    image = np.array(Image.open(path))

    thresh = otsu_threshold(image)
    binary_image = apply_threshold(image, thresh)

    labels = watershed_segmentation(binary_image)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Oryginalny obraz')
    axs[0].axis('off')

    axs[1].imshow(binary_image, cmap='gray')
    axs[1].set_title(f'Binaryzacja Otsu (thresh={thresh})')
    axs[1].axis('off')

    axs[2].imshow(labels, cmap='nipy_spectral')
    axs[2].set_title('Segmentacja Watershed')
    axs[2].axis('off')

    plt.show()

image_paths = ['coins.jpg', 'cells.jpg']

for path in image_paths:
    process_image(path)
