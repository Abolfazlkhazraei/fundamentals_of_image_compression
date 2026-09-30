# Exercise 5b: Spectral distribution of linear ramp

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, fft

x = np.arange(8, dtype=float)            # linear ramp [0,1,...,7]
n = np.arange(8)

X_dct = dct(x, type=2, norm="ortho")     # DCT-II with the alpha_k scaling
X_dft = fft(x)

print("DCT:", np.round(X_dct, 3))
print("DFT:", np.round(X_dft, 3))

# Share of the total energy in each coefficient
print("DCT energy %:", np.round(100 * X_dct**2 / np.sum(X_dct**2), 2))
print("DFT energy %:", np.round(100 * np.abs(X_dft)**2 / np.sum(np.abs(X_dft)**2), 2))

fig, axes = plt.subplots(1, 4, figsize=(17, 4))
axes[0].stem(n, x)
axes[0].set_title("Signal: linear ramp"); axes[0].set_xlabel("n")
axes[1].stem(n, X_dct, linefmt="C0-")
axes[1].set_title("DCT-II coefficients")
axes[2].stem(n, X_dft.real, linefmt="C1-", markerfmt="C1o")
axes[2].set_title("DFT real part")
axes[3].stem(n, X_dft.imag, linefmt="C2-", markerfmt="C2o")
axes[3].set_title("DFT imaginary part")
for ax in axes[1:]:
    ax.set_xlabel("k")
for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_5b.png", dpi=130)
plt.show()