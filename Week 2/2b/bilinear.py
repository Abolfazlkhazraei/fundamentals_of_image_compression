# Exercise 2b: Bilinear Transformation

from PIL import Image
import numpy as np

def scale_bilinear(img, factor):
    """
    upscale an image by a factor using bilinear interpolation
    """
    img = img.astype(np.float64)
    h, w = img.shape[:2]
    ch = img.shape[2] if img.ndim == 3 else 1
    img = img.reshape(h, w, ch) 
    nh, nw = int(h * factor), int(w * factor)

    # Map each output coordination bak to source (align centers)
    src_y = (np.arange(nh) + 0.5) / factor - 0.5
    src_x = (np.arange(nw) + 0.5) / factor - 0.5
    src_y = np.clip(src_y, 0, h - 1)
    src_x = np.clip(src_x, 0, w - 1)
 
    y0 = np.floor(src_y).astype(int); y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.floor(src_x).astype(int); x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (src_y - y0)[:, None, None]   # vertical weights
    wx = (src_x - x0)[None, :, None]   # horizontal weights
 
    Ia = img[np.ix_(y0, x0)]  # top-left
    Ib = img[np.ix_(y0, x1)]  # top-right
    Ic = img[np.ix_(y1, x0)]  # bottom-left
    Id = img[np.ix_(y1, x1)]  # bottom-right
 
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bot * wy
    return np.clip(out, 0, 255).astype(np.uint8).reshape(nh, nw, ch).squeeze()

shirt = np.array(Image.open("../Images/shirt_small.jpg"))
print("Original size:", shirt.shape)

bl = scale_bilinear(shirt, 4)
Image.fromarray(bl).save("shirt_bilinear_x4.png")