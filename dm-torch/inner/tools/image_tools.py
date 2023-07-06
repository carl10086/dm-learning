import matplotlib.pyplot as plt
from PIL import Image


def show_images(images):
    for image in images:
        show_image(image)


def show_image(img):
    plt.imshow(img)
    plt.show()


def show_np(array):
    show_image(Image.fromarray((array * 255).round().astype("uint8")))
