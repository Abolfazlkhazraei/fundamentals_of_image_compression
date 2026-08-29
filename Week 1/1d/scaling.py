from PIL import Image
import numpy as np

def scale_image(pix, factor):
    old_h, old_w = pix.shape[0], pix.shape[1]
    new_h, new_w = int(old_h * factor), int(old_w * factor)
    new_pix = np.zeros((new_h, new_w, pix.shape[2]), dtype=pix.dtype)
    for y in range(new_h):
        src_y = int(y / factor)
        for x in range(new_w):
            src_x = int(x / factor)
            new_pix[y, x] = pix[src_y, src_x]
    return new_pix

shirt = np.array(Image.open("../Image/shirt.jpg"))
scaled_shirt = scale_image(shirt, 0.17)
Image.fromarray(scaled_shirt).save("scaled_shirt.jpg")