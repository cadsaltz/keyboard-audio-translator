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

from extractFeatures import read_wav, detect_clicks, build_feature_matrix, append_features_to_file, append_keys_to_file, clean_filename

# main function for training data
def main():

	# parse system argumnts
	if len(sys.argv) == 2:

		# grab their input grab their input and use as filename to open and read
		filename = sys.argv[1]
	
	# anything else is unacceptable 
	else:
		# inform user and exit
		print("Usage: train <filename.wav>")
		print("Extracts features from wav file and appends them to the csv files")
		exit()

	# grab necessary data from wav file
	signal, times, freq = read_wav(filename)

	# find the peaks (clicks)
	peaks = detect_clicks(signal, PEAK_THRES, PEAK_DIST)

	# build the data matrix
	x_train = build_feature_matrix(signal, peaks, freq)
	
	# append matrix to csv file
	append_features_to_file(x_train, "keyclick_data.csv")

	keys = clean_filename(filename)
#	print("keys after cleaning: ", keys)

	append_keys_to_file(keys, "labels.csv")

if __name__ == "__main__":
	main()




















