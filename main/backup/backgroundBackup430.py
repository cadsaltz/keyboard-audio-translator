# filename: find_noise_threshold.py

from train import read_wav
import numpy as np
import sys

def main():
	
	if len(sys.argv) == 2:
		filename = sys.argv[1]

	else:
		print("Usage: python3 background.py <filename.wav>")

	# Read the wav file
	signal, sample_rate, _ = read_wav(filename)

	# Calculate average absolute amplitude
	avg_amplitude = np.mean(np.abs(signal))

	print("Average background noise amplitude: ", avg_amplitude)

if __name__ == "__main__":
	main()

