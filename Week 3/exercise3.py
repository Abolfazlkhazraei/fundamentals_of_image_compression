import numpy as np
from PIL import Image

# Exercise 3a Expanding image for convolution filtering
def expand_image(img, pad=2):
    """Expand the image by `pad` pixels on every edge, copying the adjacent
    (nearest) edge pixel. Works for 2D (grayscale) or 3D (color) arrays."""
    img = np.asarray(img)
    if img.ndim == 2:
        return np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
    else:
        return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

# Test with a random 5x5x3 matrix (y=5, x=5, 3 color channels)
np.random.seed(0)
test = np.random.randint(0, 10, (5, 5, 3))
expanded = expand_image(test, 2)
print("Original shape:", test.shape)
print(test[:, :, 0])
print("Expanded shape:", expanded.shape)
print(expanded[:, :, 0])

# Exercise 3b Convolution Filter

def convolve(img, kernel):
    """Apply a square kernel via convolution. The image is first expanded 
    with the 3a function by k//2 pixels so the output keeps its size."""
    img = np.asarray(img, dtype=np.float64)
    k = kernel.shape[0]; pad = k // 2
    padded = expand_image(img, pad)
    H, W, C = img.shape
    out = np.zeros((H, W, C))
    for c in range(C):
        for i in range(k):
            for j in range(k):
                out[:, :, c] += kernel[i, j] * padded[i:i+H, j:j+W, c]
    return out

gaussian = np.array([
    [0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625],
    [0.015625,   0.0625,   0.09375,   0.0625,   0.015625],
    [0.0234375,  0.09375,  0.140625,  0.09375,  0.0234375],
    [0.015625,   0.0625,   0.09375,   0.0625,   0.015625],
    [0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625],
])

shirt = np.asarray(Image.open("Images/shirt.jpg").convert("RGB"), dtype=np.float64)
blurred = convolve(shirt, gaussian)
Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8)).save("shirt_blurred.png")
print("blurred saved:", blurred.shape)