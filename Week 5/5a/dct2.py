# Exercise 5a: Discrete cosine transform (DCT-II) vs. discrete Fourier transform

import numpy as np
import matplotlib.pyplot as plt

N = 8
n = np.arange(N)                    # sample positions
t = np.linspace(0, N - 1, 400)      # dense axis for the continuous curve

def dct_basis(k, n, N):
    alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
    return alpha * np.cos(np.pi / N * (n + 0.5) * k)

def dft_basis(k, n, N):
    return np.exp(-2j * np.pi * k * n / N)      # kernel of X_k = sum x_n e^(-j2pi kn/N)

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for row, k in enumerate([0, 1]):
    # DCT-II basis (real valued)
    ax = axes[row, 0]
    ax.plot(t, dct_basis(k, t, N), "C0--", alpha=0.4)
    ax.stem(n, dct_basis(k, n, N), linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax.set_title(f"DCT-II basis, k={k}")
    ax.set_ylim(-0.6, 0.6)

    # DFT basis: real part, imaginary part and magnitude
    parts = [("real part", np.real, "C1"),
             ("imaginary part", np.imag, "C2"),
             ("magnitude", np.abs, "C3")]

    for col, (name, f, c) in enumerate(parts, start=1):
        ax = axes[row, col]
        ax.plot(t, f(dft_basis(k, t, N)), c + "--", alpha=0.4)
        ax.stem(n, f(dft_basis(k, n, N)), linefmt=c + "-", markerfmt=c + "o", basefmt="k-")
        ax.set_title(f"DFT basis (k={k}), {name}")
        ax.set_ylim(-1.2, 1.2)

for ax in axes.flat:
    ax.set_xlabel("n")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_5a.png", dpi=130)
plt.show()    