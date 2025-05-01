#!/usr/bin/env python3

# given file, extract x data and dump into csv file

# import for wav file reading
import wave

# import numpy for matrices
import numpy as np

# import sys for argument inputs
import sys

# import signal fro wav file reading
from scipy.signal import find_peaks, butter, filtfilt

# impor fft for data extraction
from scipy.fft import fft, fftfreq

from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit

# import csv and os for csv file creation
import csv
import os

import matplotlib.pyplot as plt

# define global variables for fine tuning
#WINDOW_LEFT = 1700
#WINDOW_RIGHT = 2500
BNT = 350
WIN = 300

PEAK_THRES = 10000
PEAK_DIST = 5000

# read wav file function
# take a filename string
def read_wav(filename):
	
	# read the file (read binary)
	with wave.open(filename, 'rb') as wf:

		# grabe data using wave library methods
		n_channels = wf.getnchannels()
		n_frames = wf.getnframes()
		sample_freq = wf.getframerate()
		audio_length = n_frames / sample_freq
		audio_data = wf.readframes(-1)
		
		# grab the amplitues of the wav file
		signal = np.frombuffer(audio_data, dtype=np.int16)
		signal = signal.astype(np.float32)

		# stereo mics record two amplitudes per sample, only need one for data reading
		if n_channels == 2:

			# skip every other sample (the duplicate)
			signal = signal[::2]

		# make an array of times in seconds instead of samples
		times = np.linspace(0, audio_length, num=len(signal))

	# return necessary file data
	return signal, times, sample_freq

# function to find the location in the signal array of all the peaks in that array
# make a list of array indices
def detect_clicks(signal, peak_thres, peak_dist):
	
	# use the find_peaks method from scipy.signal to find all the peaks
	peaks, properties = find_peaks(signal, height=peak_thres, distance=peak_dist)

	# return that list
	return peaks


def find_dynamic_window(signal, center_idx, window_size):

	left = center_idx
	right = center_idx
	signal_length = len(signal)

	def avg_abs_amplitude(start, end):
		start = max(0, start)
		end = min(signal_length, end)
		return np.mean(np.abs(signal[start:end]))
	

	while left > 0 and avg_abs_amplitude(left - window_size, left) > BNT:
		left -= 5

	while right < signal_length - 1 and avg_abs_amplitude(right, right + window_size) > BNT:
		right += 5

	if right - left < 10:
		left = max(0, center_idx - 500)
		right = min(len(signal), center_idx +500)

	return signal[left:right], left, right



# function to extract the features from peak
# this is the data used to build a vector for the ml model
# the data the model will learn to relate to the label
def extract_features(window, sample_freq):

	window = window.astype(np.float32)

	peak_amp = np.max(np.abs(window))
	mean_amp = np.mean(window)
	std_amp = np.std(window)
	rms_amp = np.sqrt(np.maximum(0, np.mean(window ** 2)))
	energy = np.sum(window ** 2)
	window_size = len(window) / sample_freq

	fft_result = np.fft.fft(window)
	magnitude = np.abs(fft_result)
	freqs = np.fft.fftfreq(len(window), d=1 / sample_freq)
	pos_freqs = freqs[:len(freqs) // 2]
	pos_mag = magnitude[:len(magnitude) // 2]

	if np.sum(pos_mag) == 0:
		spectral_centroid = 0
		bandwidth = 0
		dominant_freq = 0
		power = 0
	else:
		spectral_centroid = np.sum(pos_freqs * pos_mag) / np.sum(pos_mag)
		bandwidth = np.sqrt(np.sum((pos_freqs - spectral_centroid) ** 2 * pos_mag) / np.sum(pos_mag))
		dominant_freq = pos_freqs[np.argmax(pos_mag)]
		power = np.sum(pos_mag ** 2)

	peak_idx = np.argmax(np.abs(window))
	decay_window = window[peak_idx:]
	decay_time = np.linspace(0, len(decay_window) / sample_freq, len(decay_window))

	if len(decay_window) > 5:
		try:
		
			popt, _ = curve_fit(exp_decay, decay_time, np.abs(decay_window), p0=(decay_window[0], 10))
			decay_constant = popt[1]
		except:
			decay_constant = 0
	else:
		decay_constant = 0

	energy_after_peak = np.sum(decay_window ** 2)
	peak_to_end_energy_ratio = energy_after_peak / (energy + 1e-8)

	zero_crossings = np.sum(np.diff(np.sign(window)) != 0)
	zcr = zero_crossings / len(window)

	window_skewness = skew(window)
	window_kurtosis = kurtosis(window)

	if peak_idx > 0:
		sharpness = (np.abs(window[peak_idx]) - np.abs(window[peak_idx - 1])) * sample_freq
	else:
		sharpness = 0

	return [
		peak_amp, mean_amp, std_amp, rms_amp, energy,
		dominant_freq, spectral_centroid, bandwidth, power,
		window_size, decay_constant, peak_to_end_energy_ratio, zcr,
		window_skewness, window_kurtosis, sharpness
	]


# function to build the feature matrix
def build_feature_matrix(signal, peaks, sample_freq):
	
	# init a blank feature list
	features = []

	# pair the key presses
	keypress_pairs = [(peaks[i], peaks[i+1]) for i in range(0, len(peaks) - 1, 2)]

	# init the order to zero (for csv alignment with labels)
	order = 0

	# grab pairs of peaks 
	for press_idx, release_idx in keypress_pairs:

		press_window, _ , _ = find_dynamic_window(signal, press_idx, WIN)
		release_window, _ , _ = find_dynamic_window(signal, release_idx, WIN)

		press_features = extract_features(press_window, sample_freq)
		release_features = extract_features(release_window, sample_freq)

		feature_vector = press_features + release_features + [order]
		features.append(feature_vector)

		order += 1

	# convert the list to a matrix and return it
	return np.array(features)


# function to append the matrix to csv file
def append_features_to_file(features, filename):

	file_exists = os.path.isfile(filename)

	with open(filename, 'a', newline='') as csvfile:
		writer = csv.writer(csvfile)

	        # Only write header if file doesn't exist yet
		if not file_exists:

			# header for the csv file
			header = ['press_peak', 'press_mean', 'press_std', 'press_rms', 'press_energy', 'press_dom_freq', 'press_centroid', 'press_bandwidth', 'press_power', 'release_peak', 'release_mean', 'release_std', 'release_rms', 'release_energy', 'release_dom_freq', 'release_centroid', 'release_bandwidth', 'release_power']
	
			# write the header
			writer.writerow(header)

	# for every vector in the feautures matric
		for row in features:
	
			# write it to the file
			writer.writerow(row)


# function to append a string to a csv file
def append_keys_to_file(labels, filename):

	# make a list to store the keys
	data = []

	# check if the file exists
	file_exists = os.path.isfile(filename)

	# open the given file
	with open(filename, 'a', newline='') as csvfile:

		# initialize a csv writer
		writer = csv.writer(csvfile)
		
		# initialize the order to zero
		order=0

		# if the file is new, add a header
		if not file_exists:

			# label (character for the parallel feature lsit), order number for alignment
			header = ['label', 'check']

			# write the header
			writer.writerow(header)

		# for every character in the given string
		for key in labels:
	
			# create a vecetor for the matrix (list with two elements)
			vector = [key, order]

			# append it to the data matrix
			data.append(vector)

			# increment the order each vector
			order+=1

		# make the data list a numpy array
		np.array(data)

		# for every row (list/vector) in the data matrix
		for row in data:

			# add it to the csv file
			writer.writerow(row)


# main function
def clean_filename(filename):
	if filename.endswith(".wav"):
		filename = filename[:-4]
	return filename.replace("_"," ")


# plot the wav file, highlight and border the key presses
def plot_click_windows(signal, times, peaks, sample_freq, png_filename):

	# init a plot
	plt.figure(figsize=(20,5))

	# plot the time and signal (amplitudes)
	plt.plot(times, signal)

	# for every location of a peak
	for peak in peaks:

		window_signal, left, right = find_dynamic_window(signal, peak, WIN)
		
		# convert its index from samples to seconds
		left_time = left / sample_freq
		right_time = right / sample_freq
		peak_time = peak / sample_freq

		# border and highlight it on the plot
		plt.axvline(x=peak_time, color='red',linestyle='--', linewidth=1)
		plt.axvspan(left_time, right_time, color='orange', alpha=0.3)

	# add title and labels for the plot
	plt.title('Key click events detected')
	plt.xlabel('Time')
	plt.ylabel('Amplitude')

	# save the plot to a png file
	plt.savefig(png_filename)

def grade(signal, peaks):

	count = 1
	clean = True
	for one, two  in zip(peaks, peaks[1:]):

		window_one, l1, r1 = find_dynamic_window(signal, one, WIN)
		window_two, l2, r2 = find_dynamic_window(signal, two, WIN)


		if abs(l1 - l2) <= 20 or abs(r1 - r2) <= 20:
			print("COMPLETE OVERLAP with peaks", count, "and ", count+1)
			clean = False

		elif (l2 < r1):
			print("PARTIAL OVERLAP with peaks: ", count, " and ", count+1)
			clean = False	

		count += 1

	if (clean):
		print("NO overlap, signal is clean")


def evenSteven(arr):
	if len(arr) % 2 == 0:
		return True
	else:
		return False
