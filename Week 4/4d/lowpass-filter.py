# Exercise 4d: Frequency domain filtering

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

# Load image as grayscale
arr = np.asarray(Image.open("../Images/fox.jpg").convert("L"), dtype=float)
rows, cols = arr.shape

# Forward FFT, shift 0-frequency to the centre
F = np.fft.fft2(arr)
Fsh = np.fft.fftshift(F)

# Circular low-pass mask: keep frequencies within radius 50 of the centre
crow, ccol = rows // 2, cols // 2
Y, X = np.ogrid[:rows, :cols]
dist = np.sqrt((Y - crow)**2 + (X - ccol)**2)
radius = 50
mask = dist <= radius

Fsh_filtered = Fsh * mask

# Inverse FFT back to an image
img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(Fsh_filtered)))

# Spectra for display (log magnitude)
spec_orig = np.log(1 + np.abs(Fsh))
spec_filt = np.log(1 + np.abs(Fsh_filtered))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(arr, cmap="gray") 
axes[0, 0].set_title("Original image"); axes[0, 0].axis("off")
axes[0, 1].imshow(spec_orig, cmap="gray") 
axes[0, 1].set_title("Original spectrum (log, centered)");                                                                                                                                                                                                                                                                                                                              axes[0, 1].axis("off")
axes[1, 0].imshow(img_back, cmap="gray")
axes[1, 0].set_title("Filtered image (radius <= 50)"); axes[1, 0].axis("off")
axes[1, 1].imshow(spec_filt, cmap="gray") 
axes[1, 1].set_title("Filtered spectrum (low freq kept)"); axes[1, 1].axis("off")

plt.tight_layout()
plt.show()