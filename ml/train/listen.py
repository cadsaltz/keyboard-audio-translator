#!/usr/bin/env python3

# insect wav file and check if its valid
# highlight and border key presses
# inform user of number of keys and validity

# import wave for wav file reading
import wave

# import matplot for visualizing the wav file
import matplotlib.pyplot as plt

# import numpy for matrices
import numpy as np

# import sys for argument inputs
import sys

# import scipy for wav manipulation
from scipy.signal import find_peaks, butter, filtfilt

# import wav reading, detecting clicks, and var defs from training function
from trainx import read_wav, detect_clicks, WINDOW_LEFT, WINDOW_RIGHT, PEAK_THRES, PEAK_DIST

# plot the wav file, highlight and border the key presses
def plot_click_windows(signal, times, peaks, window_left, window_right, sample_freq, png_filename):

	# init a plot
	plt.figure(figsize=(20,5))

	# plot the time and signal (amplitudes)
	plt.plot(times, signal)

	# convert the window into seconds instead of number of samples
	window_left_sec = window_left / sample_freq
	window_right_sec = window_right / sample_freq

	# for every location of a peak
	for peak in peaks:

		# convert its index from samples to seconds
		peak_time = peak / sample_freq

		# border and highlight it on the plot
		plt.axvline(x=peak_time, color='red',linestyle='--', linewidth=1)
		plt.axvspan(peak_time-window_left_sec, peak_time+window_right_sec, color='orange', alpha=0.3)

	# add title and labels for the plot
	plt.title('Key click events detected')
	plt.xlabel('Time')
	plt.ylabel('Amplitude')

	# save the plot to a png file
	plt.savefig(png_filename)


# main function
# handle argument inputs, read the file, plot it, and check validity
def main():

	# argument inputs 
	if len(sys.argv) == 2:

		# take the first argument as the filename
		filename = sys.argv[1]

	# anything else is unacceptable
	else:

		# prompt user and exit
		print("Usage: python3 listen.py <filename.wav>")
		exit()
	
	# get the necessary data from the wav file
	signal, times, freq = read_wav(filename)

	# find all the peaks in the wav file
	peaks = detect_clicks(signal, PEAK_THRES, PEAK_DIST)

	# save the png as the same name as the given wav file
	png_filename = filename.replace(".wav", ".png")

	# plot the given file
	plot_click_windows(signal, times, peaks, WINDOW_LEFT, WINDOW_RIGHT, freq, png_filename)

	# check validity of the wav file
	# there should be an even number of peaks (each click has a press and release peak)
	if (len(peaks) % 2 == 0):

		# inform user
		print("Likely valid")

	# else (odd number of peaks)
	else:

		# inform user
		print("Invalid")

	# also tell the user how many keys were detected (pairs of peaks)
	print("Number of keypresses detected: ", (len(peaks) / 2))


if __name__ == "__main__":
	main()





















