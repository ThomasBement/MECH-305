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

name = "audio_sample_4"
sample_rate, data = wavfile.read("%s.wav" %name)
print("Sample rate:", sample_rate, 'Hz')

# Uncomment if .wav file has two chanels (the fft graph will be flat)
#if data.shape[1] != 1:
#  data = data[:,1]


N = data.shape[0] # number of sampled points
time = np.arange(N)/sample_rate

plt.plot(time, data)
plt.xlabel('Time/second')
plt.ylabel('Amplitude')
plt.title("Sound from %s" %name)
plt.show()

yf = fft(data)
yf_scaled_single_sided = 2.0/N * np.abs(yf[:N//2])

T = 1.0 / sample_rate # Sample spacing
xf = fftfreq(N, T)[:N//2] # Frequency domain

plt.plot(xf, yf_scaled_single_sided)
plt.title('Single-Sided Amplitude Spectrum of %s' %name)
plt.xlabel('f (Hz)')
plt.ylabel('Amplitude')
plt.axis([0, 5000, 0, 600])
plt.grid(True)
plt.show()
