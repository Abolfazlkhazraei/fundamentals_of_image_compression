# Exercise 4c: 2D Fourier Transform

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

files = ["../Images/vl.png", "../Images/hl.png", "../Images/diag.png"]
titles = ["vl.png (vertical lines)",
          "hl.png (horizontal lines)",
          "diag.png (diagonal lines)"]

fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for row, (fn, title) in enumerate(zip(files, titles)):
    # Load and convert to grayscale
    arr = np.asarray(Image.open(fn).convert("L"), dtype=float)

    # 2-D FFT, shift 0-frequency to the centre
    F = np.fft.fft2(arr)
    F_shifted = np.fft.fftshift(F)

    # Logarithmic magnitude for display
    visual_spectrum = np.log(1 + np.abs(F_shifted))

    axes[row, 0].imshow(arr, cmap="gray")
    axes[row, 0].set_title(title)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(visual_spectrum, cmap="gray")
    axes[row, 1].set_title("magnitude spectrum (log, centered)")
    axes[row, 1].axis("off")

plt.tight_layout()
plt.show()