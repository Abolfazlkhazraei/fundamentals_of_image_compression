# Exercise 4b: 1D Discrete Fourier Transform

import numpy as np
import matplotlib.pyplot as plt

fs = 100_000
T = 0.01
N = int(fs * T)
t = np.arange(N) / fs

sine1k = np.sin(2 * np.pi * 1000 * t)
square1k = np.sign(np.sin(2 * np.pi * 1000 * t))
sum_sig = np.sin(2*np.pi*1000*t) + np.sin(2*np.pi*8000*t)

signals = [
    ("1 kHz sine wave", sine1k),
    ("1 kHz square wave", square1k),
    ("1 kHz + 8 kHz sine", sum_sig)
]

freqs = np.fft.rfftfreq(N, d=1/fs)

fig, axes = plt.subplots(3, 2, figsize=(13, 9))

for row, (name, sig) in enumerate(signals):

    F = np.fft.rfft(sig)
    mag = np.abs(F) / N * 2
    mag[0] /= 2

    axes[row, 0].plot(t * 1000, sig)
    axes[row, 0].set_xlim(0, 3)
    axes[row, 0].set_title(f"{name} - signal")
    axes[row, 0].set_xlabel("Time [ms]")
    axes[row, 0].set_ylabel("Amplitude")
    axes[row, 0].grid(alpha=0.3)

    axes[row, 1].plot(freqs / 1000, mag, color="red")
    axes[row, 1].set_xlim(0, 25)
    axes[row, 1].set_title(f"{name} - magnitude spectrum")
    axes[row, 1].set_xlabel("Frequency [kHz]")
    axes[row, 1].set_ylabel("Magnitude")
    axes[row, 1].grid(alpha=0.3)

    peaks = sorted(np.argsort(mag)[::-1][:8])
    print(f"\n{name}: strongest frequency components")
    for p in peaks:
        if mag[p] > 0.02:
            print(f"  {freqs[p]/1000:6.2f} kHz -> magnitude {mag[p]:.3f}")

plt.tight_layout()
plt.show()