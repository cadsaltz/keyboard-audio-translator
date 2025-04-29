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
from train import read_wav, detect_clicks, find_dynamic_window, PEAK_THRES, PEAK_DIST, BNT, WIN

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
	plot_click_windows(signal, times, peaks, freq, png_filename)

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





















