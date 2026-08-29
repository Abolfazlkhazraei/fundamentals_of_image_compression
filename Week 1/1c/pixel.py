# Answer of Week 1, Exercise 1c

from PIL import Image
import numpy as np

picture = Image.open("../Image/linnanmaa.jpg")
pixels = np.array(picture)

def print_pixel(pix, y, x):
    r, g, b = pix[y, x, 0], pix[y, x, 1], pix[y, x, 2]
    print(f"Pixel ({x}, {y}) -> R={r}, G={g}, B={b}")

print_pixel(pixels, 0, 0)  #upper left corner
height, width = pixels.shape[0], pixels.shape[1]
print(f"Width = {width} px, Height = {height} px")

# Answer of Week 1, Exercise 1e

def to_greyscale(pix):
    avg = pix.mean(axis=2).astype(pix.dtype)  # average of R,G,B
    grey =  np.stack([avg, avg, avg], axis=2) # copy into 3 channels
    return grey

grey = to_greyscale(pixels)
Image.fromarray(grey).save("linnanmaa_greyscale.jpg")

# Answer of Week 1, Exercise 1f

def quantize(pix, bits):
    shift = 8 - bits
    q = (pix >> shift) << shift  # keep only the top 'bits' bits
    return q.astype(np.uint8)

for bits in (4, 3, 2, 1):
    q = quantize(pixels, bits)
    Image.fromarray(q).save(f"linnanmaa_quantized_{bits}bit.jpg")