import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('image.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

def image_show_double(img1, img2, title1, title2):
    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def image_show_gray(img1, img2, title1, title2):
    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

#Zad1
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
image_show_gray(img_rgb, img_gray, "Oryginał", "Obraz w odcieniach szarości")

#Zad2
value = 15
img_gaussian = cv.GaussianBlur(img, (value, value), 0)
img_gaussian_rgb = cv.cvtColor(img_gaussian, cv.COLOR_BGR2RGB)

image_show_double(img_rgb,img_gaussian_rgb, "Oryginał", f"Rozmycie Gaussa ({value}x{value})")

#Zad3
t_low=200
t_high=300
edge = cv.Canny(img, t_low, t_high)
edge_rgb = cv.cvtColor(edge, cv.COLOR_BGR2RGB)

image_show_double(img_rgb, edge_rgb, "Oryginał", "Detekcja krawedzi")

#Zad4
M=255
T=170
_, img_binary = cv.threshold(img_gray, T, M, cv.THRESH_BINARY)
image_show_gray(img_rgb, img_binary, "Oryginał", "Binaryzacja")

#Zad5
kat = 45
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv.getRotationMatrix2D(center, kat, 1.0)
rotated = cv.warpAffine(img, M, (w, h))
rotated_rgb = cv.cvtColor(rotated, cv.COLOR_BGR2RGB)

image_show_double(img_rgb, rotated_rgb, "Oryginał", f"Obrócony o {kat}°")

cv.imshow("Oryginał", img)
cv.imshow(f"Obrócony o {kat}°", rotated)
cv.waitKey(0)
cv.destroyAllWindows()

#Zad6
skala = 0.5
resized = cv.resize(img, None, fx=skala, fy=skala, interpolation=cv.INTER_LINEAR)
cv.imshow("Oryginał", img)
cv.imshow("Pomniejszony", resized)
cv.waitKey(0)
cv.destroyAllWindows()

#Zad7
equalized = cv.equalizeHist(img_gray)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap="gray")
plt.title("Oryginał")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(equalized, cmap="gray")
plt.title("Po wyrównaniu histogramu")
plt.axis("off")

plt.tight_layout()
plt.show()

#Zad8
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

sharpened = cv.filter2D(img, -1, kernel)
sharp_rgb = cv.cvtColor(sharpened, cv.COLOR_BGR2RGB)

image_show_double(img_rgb, sharp_rgb, "Oryginał", "Wyostrzony")