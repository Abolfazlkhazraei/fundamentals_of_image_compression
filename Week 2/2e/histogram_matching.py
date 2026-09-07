# Exercise 2e Histogram Matching

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def histogram(img):
    hist = np.zeros(256, dtype=int)
    for value in img.ravel():
        hist[value] += 1
    return hist
 
def cdf_from_histogram(hist):
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1]
    return cdf
 
def match_histograms(src, ref):
    """Map intensities of src so its histogram matches ref's. Returns matched image + mapping."""
    src_cdf = cdf_from_histogram(histogram(src))
    ref_cdf = cdf_from_histogram(histogram(ref))
    mapping = np.zeros(256, dtype=np.uint8)
    for s in range(256):
        mapping[s] = np.argmin(np.abs(ref_cdf - src_cdf[s]))
    return mapping[src], mapping
 
dark   = np.array(Image.open("../Images/dark_flower.jpg").convert("L"))
flower = np.array(Image.open("../Images/flower.jpg").convert("L"))
 
matched, mapping = match_histograms(dark, flower)
Image.fromarray(matched).save("dark_flower_matched.png")
 
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].imshow(dark,    cmap="gray", vmin=0, vmax=255); ax[0,0].axis("off"); ax[0,0].set_title("dark_flower.jpg (source)")
ax[0,1].imshow(flower,  cmap="gray", vmin=0, vmax=255); ax[0,1].axis("off"); ax[0,1].set_title("flower.jpg (reference)")
ax[0,2].imshow(matched, cmap="gray", vmin=0, vmax=255); ax[0,2].axis("off"); ax[0,2].set_title("Result (matched)")
ax[1,0].bar(range(256), histogram(dark),    width=1.0, color="black"); ax[1,0].set_title("dark_flower hist"); ax[1,0].set_xlim(0,255)
ax[1,1].bar(range(256), histogram(flower),  width=1.0, color="black"); ax[1,1].set_title("flower hist");      ax[1,1].set_xlim(0,255)
ax[1,2].bar(range(256), histogram(matched), width=1.0, color="black"); ax[1,2].set_title("matched hist");     ax[1,2].set_xlim(0,255)
plt.tight_layout(); plt.savefig("histogram_matching.png", dpi=110); plt.close()
print("matching done")