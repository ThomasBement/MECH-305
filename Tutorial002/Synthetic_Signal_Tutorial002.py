# ------------------------------------------------------#
# Name: Thomas Bement
# Student Number: 24099822
# Date: 01/17/2021
# 
# Tutorial 2 Python code for fourier analysis of
# various chords/notes in a .wav format
# ------------------------------------------------------#

# Import Libraries
import numpy as np # Linear algebra library
import matplotlib.pyplot as plt # Plotting library
from scipy.io import wavfile # Used to read .wav files
from scipy.fft import fft, fftfreq # Used to perform fast fourier transform

sampFreq = 48000 # "sampling frequency"
Duration = 3 # a 3 second signal
Time = np.linspace(0,3, sampFreq*3)
Frq1 = 261.6256 # C4
Frq2 = 329.6276 # E4
Frq3 = 391.9954 # G4
Signal = 1.0 * np.sin(2*np.pi*Frq1*Time) + \
  0.7 * np.sin(2*np.pi*Frq2*Time) + \
  0.5 * np.sin(2*np.pi*Frq3*Time) # A C Major Triad
# the "\" above is just a line break, for aesthetics

plt.plot(Time, Signal)
plt.xlabel('Time/second')
plt.ylabel('Vibrations')
plt.title("Simple sin-wave Signal")
plt.show()


yf = fft(Signal)
plt.plot(yf)
plt.show()

N = sampFreq * Duration # Number of sample points
yf_scaled_single_sided = 2.0/N * np.abs(yf[:N//2])

T = 1.0 / sampFreq # Sample spacing
xf = fftfreq(N, T)[:N//2] # Frequency domain

plt.plot(xf, yf_scaled_single_sided)
plt.title('Single-Sided Amplitude Spectrum of Signal')
plt.xlabel('f (Hz)')
plt.ylabel('Amplitude')
plt.axis([0, 440, 0, 1.5])
plt.grid(True)
plt.show()
