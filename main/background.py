# filename: find_noise_threshold.py

from extractFeatures import read_wav
import numpy as np
import sys

# main function to find the average amplitude of a wav file
# useful for determining background noise level in different environments
def main():
	
	# take argument input as filename 
	if len(sys.argv) == 2:

		filename = sys.argv[1]

	else:
		# anything else is unacceptable
		print("Usage: background <filename.wav>")
		print("Reads a wav file and returns the average amplitude")
		exit()

	# Read the wav file
	signal, sample_rate, _ = read_wav(filename)

	# Calculate average absolute amplitude
	avg_amplitude = np.mean(np.abs(signal))

	# print the average
	print("Average background noise amplitude: ", avg_amplitude)


if __name__ == "__main__":
	main()

