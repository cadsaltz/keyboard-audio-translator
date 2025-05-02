#!/usr/bin/env python3

# insect wav file and check if its valid
# highlight and border key presses
# inform user of number of keys and validity

# import sys for argument inputs
import sys

# import wav reading, detecting clicks, and var defs from training function
from extractFeatures import read_wav, detect_clicks, PEAK_THRES, PEAK_DIST, evenSteven, grade, plot_click_windows

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
		print("Usage: listen <filename.wav>")
		print("Takes a wav file, makes a plot, counts number of clicks, and grades each window")
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
	if evenSteven(peaks):

		# even number of peaks (press and release pairs)
		print("likely valid")

	else:
		# odd number of peaks (press and release pairs)
		print("invalid")

	# also tell the user how many keys were detected (pairs of peaks)
	print("Number of keypresses detected: ", (len(peaks) / 2))

	# grade the wav file, display the validity of the contained data
	grade(signal, peaks)
	print(" ")


if __name__ == "__main__":
	main()





















