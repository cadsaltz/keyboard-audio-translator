#!/usr/bin/env python3

# given file, extract x data and dump into csv file


# import sys for argument inputs
import sys

# import all necessary functions from extract features
from extractFeatures import read_wav, detect_clicks, build_feature_matrix, append_features_to_file, append_keys_to_file, clean_filename, PEAK_THRES, PEAK_DIST

# main function for training data
# this version uses the filename as the labels and populates both csv files in parallel
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

	# use the wav filename as the labels
	# clean the filename (replace underscores and remove .wav)
	keys = clean_filename(filename)
	#print("keys after cleaning: ", keys)

	# add the labels to the labels file
	append_keys_to_file(keys, "labels.csv")

if __name__ == "__main__":
	main()




















