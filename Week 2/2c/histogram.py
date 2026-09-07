# Exercise 2c: Histogram

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def histogram(img):
    """
    Return a 256-bin histogram (counts per intensity 0..255) of a grayscale image.
    """

    hist = np.zeros(256, dtype=int)
    for value in img.ravel():
        hist[value] += 1
    return hist

# Load as black and white (grayscale)
flower = np.array(Image.open("../Images/flower.jpg").convert("L"))
dark = np.array(Image.open("../Images/dark_flower.jpg").convert("L"))

for name, gimg in [("flower", flower), ("dark_flower", dark)]:
    h = histogram(gimg)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(gimg, cmap="gray", vmin=0, vmax=255); ax[0].axis("off")
    ax[0].set_title(name + ".jpg")
    ax[1].bar(range(256), h, width=1.0, color="black")
    ax[1].set_title("Histogram"); ax[1].set_xlabel("Intensity"); ax[1].set_ylabel("Count")
    ax[1].set_xlim(0, 255)
    plt.tight_layout(); plt.savefig(name + "_hist.png", dpi=110); plt.close()
    print(name + ".histogram done")