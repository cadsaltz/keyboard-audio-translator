#!/usr/bin/env python3

import wave
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq

def map_keypresses(keypress_data):

	frequency_map = {
		100.00: "A",
		100.00: "B",
		100.00: "C",
		1248.52: "D",
		1180.71: "E",
		100.00: "F",
		100.00: "G",
		750.07: "H",
		100.00: "I",
		100.00: "J",
		100.00: "K",
		1230.85: "L",
		100.00: "M",
		100.00: "N",
		666.27: "O",
		100.00: "P",
		100.00: "Q",
		1246.00: "R",
		100.00: "S",
		100.00: "T",
		100.00: "U",
		100.00: "V",
		661.82: "W",
		100.00: "X",
		100.00: "Y",
		100.00: "Z",
  		100.00: " ",
	}

	detected_frequencies = [entry[2] for entry in keypress_data]

	output_string = ""

	for freq in detected_frequencies:

		closest_match = min(frequency_map.keys(), key=lambda f: abs(f - freq))
		output_string += frequency_map[closest_match]

	print("Decoded string:", output_string)




def bandpass_filter(signal, lowcut=150, highcut=2000, fs=44100, order=5):
	nyquist = 0.5 * fs
	low = lowcut / nyquist
	high = highcut / nyquist
	b, a = butter(order, [low, high], btype='band', analog=False)
	return filtfilt(b, a, signal)

def parabolic_interpolation(x, y):
	idx = np.argmax(y)
	if idx == 0 or idx == len(y) - 1:
		return x[idx]
	num = y[idx - 1] - y[idx + 1]
	denom = 2 * (y[idx - 1] - 2 * y[idx] + y[idx + 1])
	if denom == 0:
		return x[idx]
	return x[idx] + (num / denom) * (x[1] - x[0])

if len(sys.argv) == 1:
	wav_filename = "re_1031.wav"
elif len(sys.argv) == 2:
	wav_filename = sys.argv[1]
else:
	print("Usage: python3 example.py <filename>")
	exit()

obj = wave.open(wav_filename, "rb")
sample_freq = obj.getframerate()
n_samples = obj.getnframes()
audio_length = n_samples / sample_freq

print("Length:", audio_length, "seconds")
frames = obj.readframes(-1)
frame_array = np.frombuffer(frames, dtype=np.int16)

frame_array = bandpass_filter(frame_array)
if frame_array.size == 0:
	print("Error, frame_array is empty")
	exit()

times = np.linspace(0, audio_length, num=len(frame_array))
peak_height_thres = 10000
peak_dist_thres = 18000
peaks, properties = find_peaks(frame_array, height=peak_height_thres, distance=peak_dist_thres)

keypress_data = []
window_size = 6000
first_window = 2000
last_window = 3000

for peak_index in peaks:
	start = max(0, peak_index - first_window)
	end = min(len(frame_array), peak_index + last_window)
	press_signal = frame_array[start:end]
	window = np.hanning(len(press_signal))
	press_signal *= window
    
	N = len(press_signal)
	yf = np.abs(fft(press_signal))
	xf = fftfreq(N, 1 / sample_freq)
	pos_mark = xf > 0
	xf = xf[pos_mark]
	yf = yf[pos_mark]
    
	dominant_freq = parabolic_interpolation(xf, yf)
	mean_freq = np.sum(xf * yf) / np.sum(yf)
	bandwidth = np.sqrt(np.sum(((xf - mean_freq) ** 2) * yf) / np.sum(yf))
    
	keypress_data.append([peak_index, frame_array[peak_index], dominant_freq, bandwidth])

print("Estimated number of keypresses:", len(peaks))
print("\nDetected peaks:")
for i, (index, amplitude, dom_freq, dist_freq) in enumerate(keypress_data):
	print(f"Peak {i+1}: Index {index}, Time {(index/sample_freq)/2:.3f}s")
#	print(f"  Amplitude: {amplitude}")
	print(f"  Dominant Frequency: {dom_freq:.2f} Hz")
	print(f"  Frequency Distribution: {dist_freq:.2f} Hz\n")

png_filename = wav_filename.replace(".wav", ".png")
plt.figure(figsize=(15, 5))
plt.plot(times, frame_array)
plt.title("Audio Signal")
plt.ylabel("Signal Wave")
plt.xlabel("Time")
plt.xlim(0, audio_length)
plt.savefig(png_filename)
print("Plot saved as", png_filename)

print()
map_keypresses(keypress_data)
