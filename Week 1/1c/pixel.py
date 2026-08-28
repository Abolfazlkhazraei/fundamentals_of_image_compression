from PIL import Image
import numpy as np

picture = Image.open("linnanmaa.jpg")
pixels = np.array(picture)

def print_pixel(pix, y, x):
    r, g, b = pix[y, x, 0], pix[y, x, 1], pix[y, x, 2]
    print(f"Pixel ({x}, {y}) -> R={r}, G={g}, B={b}")

print_pixel(pixels, 0, 0)  #upper left corner
height, width = pixels.shape[0], pixels.shape[1]
print(f"Width = {width} px, Height = {height} px")