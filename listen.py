#!/usr/bin/env python3

import wave
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq



def highpass_filter(signal, cutoff=150, fs=44100, order=5):
	nyquist = 0.5 * fs
	normal_cutoff = cutoff / nyquist
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	return filtfilt(b,a,signal)

def lowpass_filter(signal, cutoff=2000, fs=44100, order=5):
	nyquist = 0.5 * fs
	normal_cutoff = cutoff / nyquist
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return filtfilt(b,a,signal)


if (len(sys.argv) == 1):

	wav_filename = "re_1031.wav"

if (len(sys.argv) == 2):

	wav_filename = sys.argv[1]

if (len(sys.argv) > 2):

	print("Usage python3 example.py <filename>")
	exit()

obj = wave.open(wav_filename, "rb")


#print("Number of channels: ", obj.getnchannels())
#print("Sample width: ", obj.getsampwidth())
#print("Framerate: ", obj.getframerate())
#print("Number of frames", obj.getnframes())
#print("Parameters: ", obj.getparams())


sample_freq = obj.getframerate()
n_samples = obj.getnframes()
audio_length = n_samples / sample_freq

print("Length: ", audio_length, "seconds")

frames = obj.readframes(-1)

frame_array = np.frombuffer(frames, dtype=np.int16)

frame_array = highpass_filter(frame_array)
frame_array = lowpass_filter(frame_array)


if frame_array.size == 0:
	print("Error, frame_array is empty")
	exit()



times = np.linspace(0, audio_length, num=len(frame_array))

# threshold to be counted as a peak
peak_height_thres = 10000

# dont count another peak for n number of samples
peak_dist_thres = 18000

# find keypresses (peaks, ignore release peak)
peaks, properties = find_peaks(frame_array, height=peak_height_thres, distance=peak_dist_thres)



domfreq = []
distfreq = []

window_size = 6000
first_window = 2000
last_window = 3000

for i in range(len(peaks)):

	peak_index = peaks[i]

	start = max(0, peak_index - first_window)
	end = min(len(frame_array), peak_index + last_window)

	press_signal = frame_array[start:end]

	window = np.hamming(len(press_signal))
	press_signal = press_signal * window

	N = len(press_signal)
	yf = np.abs(fft(press_signal))
	xf = fftfreq(N, 1 / sample_freq)

	pos_mark = xf > 0
	xf = xf[pos_mark]
	yf = yf[pos_mark]

	dominant_freq = xf[np.argmax(yf)]
	domfreq.append(dominant_freq)

	mean_freq = np.sum(xf * yf) / np.sum(yf)
	bandwidth = np.sqrt(np.sum(((xf - mean_freq) ** 2) * yf) / np.sum(yf))
	distfreq.append(bandwidth)




print("Estimated number of keypresses:", len(peaks))

print("\nDetected peaks:")
for i in range(len(peaks)):



	print("Peak:", i+1)
	print("Index:", peaks[i])
	print("ToO:", (peaks[i]/sample_freq)/2)

#	if (i > 0):
#		difference = ((peaks[i]/sample_freq)/2) - ((peaks[i-1]/sample_freq)/2)
#		print("TFL:", difference)

#	print("Height:", frame_array[peaks[i]])
	print("dominant frequency", domfreq[i], "Hz")
	print("frequency distribution:", distfreq[i], "Hz")
	print()

#print("Detected peak amplitudes:", frame_array[peaks])
#print("Detected peak frequencies:", domfreq)
#print("Detected peak distfrequencies", distfreq)
#print("Average peak height:", np.mean(frame_array[peaks]))
#print("Max peak height:", np.max(frame_array[peaks]))
#print("Min peak height:", np.min(frame_array[peaks]))


png_filename = wav_filename.replace(".wav", ".png")

plt.figure(figsize=(15, 5))
plt.plot(times, frame_array)
plt.title("Audio signal")
plt.ylabel("Signal wave")
plt.xlabel("Time")
plt.xlim(0, audio_length)
plt.savefig(png_filename)
print("Plot saved as ", png_filename)
