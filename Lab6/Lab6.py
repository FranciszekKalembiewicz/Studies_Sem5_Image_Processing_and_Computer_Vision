import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

def otsu_threshold(image):
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
        sigma_new = w0 * w1 * (mu1 - mu0) ** 2
        if sigma_new > max_sigma:
            max_sigma = sigma_new
            threshold = k
    return threshold

def process_image(path):
    img = np.array(Image.open(path).convert('L'))

    thresh = otsu_threshold(img)
    binary = img > thresh

    foreground = ndi.binary_erosion(binary, iterations=2)
    background = ndi.binary_dilation(~binary, iterations=2)

    markers = np.zeros(img.shape, dtype=np.int32)
    markers[background] = 1
    markers[foreground] = 2

    labels = ndi.watershed_ift((binary*255).astype(np.uint8), markers)

    markers_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    markers_rgb[foreground] = [0, 0, 255]
    markers_rgb[background] = [255, 0, 0]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Oryginał")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Markery kolorowe (Otsu k={thresh})")
    plt.imshow(markers_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Watershed")
    plt.imshow(labels, cmap='nipy_spectral')
    plt.axis('off')

    plt.show()

for path in ['coins.jpg', 'coins_together.jpg']:
    process_image(path)