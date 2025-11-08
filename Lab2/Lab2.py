import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

img = cv.imread("image.jpg")
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

def orygunalny_obraz(img_rgb):
    plt.figure(figsize=(10, 5))
    plt.imshow(img_rgb)
    plt.title("Oryginał")
    plt.axis("off")
    plt.show()

orygunalny_obraz(img_rgb)

def detekcja_canny_slider(img, img_rgb):
    t_low = 100
    t_high = 100
    edge = cv.Canny(img, t_low, t_high)
    edge_rgb = cv.cvtColor(edge, cv.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 8))

    ax[0].imshow(img_rgb)
    ax[0].set_title("Oryginał")
    ax[0].axis("off")

    canny_plot = ax[1].imshow(edge_rgb)
    ax[1].set_title(f"Canny t_low:{t_low}, t_high:{t_high}")
    ax[1].axis("off")

    plt.subplots_adjust(bottom=0.5)
    low_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    high_slider = plt.axes([0.25, 0.2, 0.5, 0.03])
    tlow_slider = Slider(low_slider, "t_low", 0, 500, valinit=t_low, valstep=1)
    thigh_slider = Slider(high_slider, "t_high", 0, 500, valinit=t_high, valstep=1)

    def update(val):
        low_val = int(tlow_slider.val)
        high_val = int(thigh_slider.val)

        new_canny = cv.Canny(img, low_val, high_val)
        new_canny_rgb = cv.cvtColor(new_canny, cv.COLOR_BGR2RGB)

        canny_plot.set_data(new_canny_rgb)
        ax[1].set_title(f"t_low {low_val}, t_high {high_val}")

        fig.canvas.draw_idle()

    tlow_slider.on_changed(update)
    thigh_slider.on_changed(update)

    plt.tight_layout()
    plt.show()


detekcja_canny_slider(img, img_rgb)


def binaryzacja(img, img_rgb, M=255):
    t_val = 150
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img_binary = cv.threshold(img_gray, t_val, M, cv.THRESH_BINARY)
    binaryzacja_rgb = cv.cvtColor(img_binary, cv.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img_rgb)
    ax[0].set_title("Oryginał")
    ax[0].axis("off")

    binary_plot = ax[1].imshow(binaryzacja_rgb)
    ax[1].set_title(f"Binararyzacja T:{t_val}, M:{M}")
    ax[1].axis("off")

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])

    slider = Slider(ax_slider, "Treshold", 0, 255, valinit=t_val, valstep=1)

    def update(val):
        v = int(val)

        _, new_img_binary = cv.threshold(img_gray, v, M, cv.THRESH_BINARY)
        new_binaryzacja_rgb = cv.cvtColor(new_img_binary, cv.COLOR_BGR2RGB)

        binary_plot.set_data(new_binaryzacja_rgb)
        ax[1].set_title(f"Binararyzacja T:{v}, M:{M}")

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.tight_layout()
    plt.show()


binaryzacja(img, img_rgb)


def rozmycie_gaussa_slider(img, img_rgb):
    blur_val = 17
    img_gaussian = cv.GaussianBlur(img, (blur_val, blur_val), 0)
    img_gaussian_rgb = cv.cvtColor(img_gaussian, cv.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img_rgb)
    ax[0].set_title("Oryginał")
    ax[0].axis("off")

    blur_plot = ax[1].imshow(img_gaussian_rgb)
    ax[1].set_title(f"Rozmycie Gaussa ({blur_val})")
    ax[1].axis("off")

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])

    slider = Slider(ax_slider, "Rozmycie", 1, 31, valinit=blur_val, valstep=2)

    def update(val):
        v = int(val)
        if v % 2 == 0:
            v += 1

        new_blur = cv.GaussianBlur(img, (v, v), 0)
        new_blur_rgb = cv.cvtColor(new_blur, cv.COLOR_BGR2RGB)

        blur_plot.set_data(new_blur_rgb)
        ax[1].set_title(f"Rozmycie Gaussa ({v})")

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.tight_layout()
    plt.show()

rozmycie_gaussa_slider(img, img_rgb)