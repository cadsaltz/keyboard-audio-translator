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

# import csv and os for csv file creation
import csv
import os

# define global variables for fine tuning
WINDOW_LEFT = 1700
WINDOW_RIGHT = 2500

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

# function to extract the features from peak
# this is the data used to build a vector for the ml model
# the data the model will learn to relate to the label
def extract_features(window, sample_freq):

	# grab the window's max amplitude
	peak_amp = np.max(np.abs(window))

	# grab the average amplitude over the window
	mean_amp = np.mean(window)

	# grab the standard deviation of the amplitudes
	std_amp = np.std(window)

	# grab the rms amplitude
	rms_amp = np.sqrt(np.maximum(0, np.mean(window ** 2)))

	# grab the energy of the window (sum of all the amplitudes squared)
	energy = np.sum(window ** 2)

# recomment as neccessary
	# grab the frequency of the window
	fft = np.fft.fft(window)
	# grab the magnitude of the frequency
	magnitude = np.abs(fft)
	# grab the frequency of the window
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
	
	# return a list of all the 18 data fields of the window
	# vector per peak
	return [
		peak_amp, mean_amp, std_amp, rms_amp, energy, dominant_freq, spectral_centroid, bandwidth, power
	]

# function to build the feature matrix
def build_feature_matrix(signal, peaks, sample_freq, window_left, window_right):
	
	# init a blank feature list
	features = []

	# pair the key presses
	keypress_pairs = [(peaks[i], peaks[i+1]) for i in range(0, len(peaks) - 1, 2)]

	# init the order to zero (for csv alignment with labels)
	order = 0

	# grab pairs of peaks 
	for press_idx, release_idx in keypress_pairs:

		# make sure its within bounds
		if (press_idx - window_left < 0 or press_idx + window_right >= len(signal) or release_idx - window_left < 0 or release_idx + window_right >= len(signal)):

			# bad data, inform and exit
			print("Peak out of bounds ")
			exit()

			#continue

		# separate the press peak and the release peak
		press_window = signal[press_idx - window_left : press_idx + window_right]
		release_window = signal[release_idx - window_left : release_idx + window_right]
		# get all the features from the press
		press_features = extract_features(press_window, sample_freq)

		# get all the features from the release
		release_features = extract_features(release_window, sample_freq)

		# create a vector
		feature_vector = press_features + release_features + [order]
		
		# append it to the feature list
		features.append(feature_vector)

		# increment order for alignment
		order+=1

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
		header = [
		# Press
		'press_peak', 'press_mean', 'press_std', 'press_rms', 'press_energy',
		'press_dom_freq', 'press_centroid', 'press_bandwidth', 'press_power',
		# Release
		'release_peak', 'release_mean', 'release_std', 'release_rms', 'release_energy',
		'release_dom_freq', 'release_centroid', 'release_bandwidth', 'release_power'
		]

		# write the header
		writer.writerow(header)

	# for every vector in the feautures matric
	for row in features:

		# write it to the file
		writer.writerow(row)


# main function for training data
def main():

	# parse system argumnts
	if len(sys.argv) == 2:

		# grab their input grab their input and use as filename to open and read
		filename = sys.argv[1]
	
	# anything else is unacceptable 
	else:
		# inform user and exit
		print("Usage: python3 trainx.py <filename.wav>")
		exit()

	# grab necessary data from wav file
	signal, times, freq = read_wav(filename)

	# find the peaks (clicks)
	peaks = detect_clicks(signal, PEAK_THRES, PEAK_DIST)

	# build the data matrix
	x_train = build_feature_matrix(signal, peaks, freq, WINDOW_LEFT, WINDOW_RIGHT)
	
	# append matrix to csv file
	append_features_to_file(x_train, "keyclick_data.csv")


if __name__ == "__main__":
	main()




















