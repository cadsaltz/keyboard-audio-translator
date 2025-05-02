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

# import skey, kurtosis and curve fit for new dynamic window feature extraction
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit

# import csv and os for csv file creation
import csv
import os

# import plot to visualize the wav file
import matplotlib.pyplot as plt

# define global variables for fine tuning
#WINDOW_LEFT = 1700 # fixed windoe for feature extraction
#WINDOW_RIGHT = 2500 # fixed window for feature extraction

BNT = 350 # Background Noise Threshold
WIN = 300 # Window of samples to average for finding the dynamic window boundary 

PEAK_THRES = 10000 # the amplitude threshold to be considered a click event
PEAK_DIST = 5000 # minimum distance between peaks (ignores the next _ samples before looking for another peak)

STEP = 5 # step for the dynamic window 
# trade speed for precision with smaller/larger step

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

# function to find dynamically sized windows 
# each click event has a slightly different length and ratio
# fixed windows either cut out some data or included extra background noise
def find_dynamic_window(signal, center_idx, window_size):

	# start at center (location of highest amplitude)
	left = center_idx
	right = center_idx
	signal_length = len(signal)

	# internal function to find the average amplitude in a window (smaller list of amplitudes)
	def avg_abs_amplitude(start, end):

		# start of the window is farthest location to the right, from the left
		start = max(0, start)

		# end of the window is the farthest location to the left, from the right
		end = min(signal_length, end)

		# return the average positive amplitude of all the samples in the window
		return np.mean(np.abs(signal[start:end]))
	
	# while the location is within the bounds of the signal, and the average amplitude is greater than the threshold
	while left > 0 and avg_abs_amplitude(left - window_size, left) > BNT:

		# traverse left with the step
		left -= STEP

	# while the location is within the bounds of the signal, and the average amplitude is greater than the threshold
	while right < signal_length - 1 and avg_abs_amplitude(right, right + window_size) > BNT:

		# traverse right with the step
		right += STEP

	# if the left and right bounds are really close,
	if right - left < 15:

		# give it the fixed window size as a last resort
		left = max(0, center_idx - WINDOW_LEFT)
		right = min(len(signal), center_idx + WINDOW_RIGHT)

	# return the window, the left bound and the right bound
	return signal[left:right], left, right



# function to extract the features from peak
# this is the data used to build a vector for the ml model
# the data the model will learn to relate to the label
def extract_features(window, sample_freq):

	# convert it to float
	window = window.astype(np.float32)

	# largest amplitude in the window
	peak_amp = np.max(np.abs(window))

	# average amplitude in the window
	mean_amp = np.mean(window)
	
	# standard deviation of the window
	std_amp = np.std(window)

	# rms amplitude 
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

	# grab the peak (location of highest amplitude)
	peak_idx = np.argmax(np.abs(window))

	# only consider the window after the peak (the decay period)
	decay_window = window[peak_idx:]

	# 
	decay_time = np.linspace(0, len(decay_window) / sample_freq, len(decay_window))

	# if the decay window is long enough to fit a curve
	if len(decay_window) > 5:

		# try block because exception possibility
		try:
		
			# fit an exponential decay function to the decay window
			popt, _ = curve_fit(exp_decay, decay_time, np.abs(decay_window), p0=(decay_window[0], 10))

			# grab the decay constant
			decay_constant = popt[1]

		# if there is an exception
		except:
			
			# set the decay const to zero 
			decay_constant = 0

	else:

		# if the window is too short, set the decay const to zero
		decay_constant = 0

	# grab the energy after the peak (sum of positive amplitudes)
	energy_after_peak = np.sum(decay_window ** 2)

	# compare the energy of the window to the energy after the peak (+ a really small number to avoid div/0)
	peak_to_end_energy_ratio = energy_after_peak / (energy + 1e-8)

	# count the number of times the oscillator crosses zero
	zero_crossings = np.sum(np.diff(np.sign(window)) != 0)
	zcr = zero_crossings / len(window)

	# find the skewness of the event
	window_skewness = skew(window)

	# find the kurtosis of the event
	window_kurtosis = kurtosis(window)

	# grab the sharpness of the event, only if the index is positive
	if peak_idx > 0:

		# sharpness is
		sharpness = (np.abs(window[peak_idx]) - np.abs(window[peak_idx - 1])) * sample_freq # peak_idx - 1 to avoid seg fault, zero based indexing

	# if the peak_idx is negative
	else:

		# set the sharpness to zero
		sharpness = 0

	# return the vector of all the features
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

		# find the window of the press and release (dont need the location of the bounds, just the window)
		press_window, _ , _ = find_dynamic_window(signal, press_idx, WIN)
		release_window, _ , _ = find_dynamic_window(signal, release_idx, WIN)

		# grab the feature vectors for each window
		press_features = extract_features(press_window, sample_freq)
		release_features = extract_features(release_window, sample_freq)

		# join the vector and add the alignment index
		feature_vector = press_features + release_features + [order]

		# append the vector to the matrix
		features.append(feature_vector)

		# increment the alignment index
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


# function to grade the validity/consistency/usability of the wav file
# Problem with the dynamic windows: 
# 	if the events are too close to eachother, the windows will overlap
# 	overlaping windows means two events will share the same window for feature extraction
#	the features will include data from two events, and likely wont be valid
#		the average amplitude will be much heigher
# 		the decay const wont work
#		the frequency will be off
#		etc
def grade(signal, peaks):

	# count to remember which peak is being graded
	count = 1

	# start true, find falsities
	clean = True

	# grab tangent events
	for one, two  in zip(peaks, peaks[1:]):

		# get the dynamic windows of each event
		_, l1, r1 = find_dynamic_window(signal, one, WIN)
		_, l2, r2 = find_dynamic_window(signal, two, WIN)

		# if the left and right bounds are in the same location
		if abs(l1 - l2) <= 20 or abs(r1 - r2) <= 20:

			# two events share the same window, complete overlap
			#print("COMPLETE OVERLAP with peaks", count, "and ", count+1)

			# file is not clean
			clean = False
		
		# if the left bound of the second event is before the right bound of the first event
		elif (l2 < r1):

			# the two event windows partially overlap, if not completely
			#print("PARTIAL OVERLAP with peaks: ", count, " and ", count+1)

			# file is not clean
			clean = False	

		# increment the count to keep track 
		count += 1
	
	# if every event window has no overlap whatsoever
	if (clean):

		# the file is clean and usable
		print("CLEAN, no overlap detected")

	# if the file has at least one overlap
	else :
		# the file shouldnt be used 
		print("FILE UNUSABLE, overlapping windows")

# function to find check if there is an even number of items in a list
def evenSteven(arr):

	# if the remainder is a multiple of 2, its even
	if len(arr) % 2 == 0:

		# return true
		return True

	# anything else
	else:

		# the number is odd
		return False


