# Exercise 5d: Two dimensional DCT of an image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dctn

files = ["../Images/smooth.png", "../Images/vertical_stripes.png", "../Images/checkerboard.png"]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for col, fn in enumerate(files):
    img = np.asarray(Image.open(fn).convert("L"), dtype=float)

    # 2-D DCT-II: 1-D DCT along the rows, then along the columns
    C = dctn(img, type=2, norm="ortho")

    print(fn, "DC =", round(C[0, 0], 1))
    print(np.round(C, 1))

    axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(fn)

    im = axes[1, col].imshow(np.log10(1 + np.abs(C)), cmap="viridis")
    axes[1, col].set_title("2-D DCT, log10(1 + |C|)")
    axes[1, col].set_xlabel("horizontal frequency u")
    axes[1, col].set_ylabel("vertical frequency v")
    for (v, u), val in np.ndenumerate(C):
        if abs(val) > 0.5:
            axes[1, col].text(u, v, f"{val:.0f}", ha="center", va="center", color="white", fontsize=7)

    fig.colorbar(im, ax=axes[1, col], fraction=0.046)

plt.tight_layout()
plt.savefig("fig_5d.png", dpi=130)
plt.show()