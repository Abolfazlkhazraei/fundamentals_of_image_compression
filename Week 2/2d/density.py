# Exercise 2d Cumulative Density Function

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def histogram(img):
    hist = np.zeros(256, dtype=int)
    for value in img.ravel():
        hist[value] += 1
    return hist

def cdf_from_histogram(hist):
    """Conver a histogram into a (normalized) cumulative density function."""
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1]  # Normalize to [0, 1]
    return cdf

flower = np.array(Image.open("../Images/flower.jpg").convert("L"))
h = histogram(flower)
c = cdf_from_histogram(h)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(flower, cmap="gray", vmin=0, vmax=255); ax[0].axis("off"); ax[0].set_title("flower.png")
ax[1].bar(range(256), h, width=1.0, color="black"); ax[1].set_title("Histogram")
ax[1].set_xlabel("Intensity"); ax[1].set_ylabel("Count"); ax[1].set_xlim(0, 255)
ax[2].plot(range(256), c, color="red"); ax[2].set_title("CDF")
ax[2].set_xlabel("Intensity"); ax[2].set_ylabel("Cumulative Probability"); ax[2].set_xlim(0, 255); ax[2].set_ylim(0, 1)
plt.tight_layout(); plt.savefig("flower_cdf.png", dpi=110); plt.close()
print("CDF done.")