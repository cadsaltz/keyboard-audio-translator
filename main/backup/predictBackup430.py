
#!/usr/bin/env python3

# import model from file
# take wav file as argument input
# extract data from wav file

# use model and data to predict characters

# import wave for wav file reading
import wave

# import numpy for data manipulation
import numpy as np

# import sys for argument inputs
import sys

# import signal and fft for audio reading
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq

# import csv and os for csv import/export
import csv
import os

# import functions from x data training program
from train import read_wav, detect_clicks, extract_features, build_feature_matrix, PEAK_THRES, PEAK_DIST, find_dynamic_window

# improt pickle for model exporting/importing
import pickle

# import decode function from learning
from learn import decode_labels

# function to predict key presses
def predict(features, model_filename):

	# open the model file
	with open(model_filename, 'rb') as f:

		# import the model from the file
		model = pickle.load(f)

	# predict the string using the model
	predictions = model.predict(features)

	# return it
	return predictions

# function to prune the features
# the build features matrix function will always attach the order to the end and assumes its training data
def prune(features):
	
	# create empty list to store each vector
	pruned_features = []
	
	# for all the vectors in features
	for i in range(len(features)):
		
		# grab the vector
		feature_row = features[i]

		# grab everything except the order (last data field)
		pruned_features.append(feature_row[:-1])
	
	# return the pruned matrix
	return np.array(pruned_features)

# main function
def main():

	# parse input arguemnts
	# needs model to use and wav file to read
	if len(sys.argv) == 3:

		wav_filename = sys.argv[2]
		model_filename = sys.argv[1]

	# anything else is not acceptable
	else:

		# inform user and exit
		print("Usage: python3 predict.py <model.pkl> <filename.wav>")
		exit()

	# grab necessary data from the wav file
	signal, times, freq = read_wav(wav_filename)

	# find the peaks
	peaks = detect_clicks(signal, PEAK_THRES, PEAK_DIST)

	# prune and build feature matrix	
	data = prune(build_feature_matrix(signal, peaks, freq))

	# use model to predict output
	guessed_output = predict(data, model_filename)
	
	#print(guessed_output)

	# decode output
	message = decode_labels(guessed_output)

	# print string in plaintext
	print(''.join(message))


if __name__ == "__main__":
	main()
