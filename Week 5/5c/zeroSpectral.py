# Exercise 5c: Error from zeroing spectral components

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct, fft, ifft

N = 8
x = np.arange(N, dtype=float)
n = np.arange(N)
X_dct = dct(x, type=2, norm="ortho")
X_dft = fft(x)

# --- DCT: all 8 coefficients are non-redundant -> zero the 4 smallest ---
Xc = X_dct.copy()
zero_dct = np.argsort(np.abs(Xc))[:N // 2]
Xc[zero_dct] = 0
x_dct = idct(Xc, type=2, norm="ortho")

# --- DFT: only bins 0..N/2 are non-redundant (X_k = conj(X_(N-k))) ---
# Zero the smallest bins together with their mirror bins, so the
# spectrum stays conjugate symmetric and the result stays real.
def dft_zero_smallest(Xf, n_zero):
    Xf = Xf.copy()
    half = np.arange(N // 2 + 1)                        # bins 0..4
    order = half[np.argsort(np.abs(Xf[half]))]          # smallest first
    for k in order[:n_zero]:
        Xf[k] = 0                                       # mirror bin
        Xf[(N - k) % N] = 0
    return Xf, sorted(int(k) for k in order[:n_zero])

Xf2, z2 = dft_zero_smallest(X_dft, 2)   # 5 bins -> zero 2 (keeps 5 real numbers)
Xf3, z3 = dft_zero_smallest(X_dft, 3)   # 5 bins -> zero 3 (keeps 3 real numbers)
x_dft2 = np.real(ifft(Xf2))
x_dft3 = np.real(ifft(Xf3))

mse = lambda y: np.mean((y - x)**2)
print("DCT zeroed k:", sorted(int(k) for k in zero_dct), "MSE:", round(mse(x_dct), 4))
print("DFT zeroed k:", z2, "MSE:", round(mse(x_dft2), 4))
print("DFT zeroed k:", z3, "MSE:", round(mse(x_dft3), 4))

plt.figure(figsize=(9, 5))
plt.plot(n, x, "ko-", lw=2, label="Original ramp")
plt.plot(n, x_dct, "C0s--", label=f"DCT, 4 smallest zeroed, MSE={mse(x_dct):.4f}")
plt.plot(n, x_dft2, "C1^--", label=f"DFT, bins {z2} zeroed, MSE={mse(x_dft2):.3f}")
plt.plot(n, x_dft3, "C3v--", label=f"DFT, bins {z3} zeroed, MSE={mse(x_dft3):.3f}")
plt.xlabel("n"); plt.ylabel("Value")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig("fig_5c.png", dpi=130)
plt.show()