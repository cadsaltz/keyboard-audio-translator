
#!/usr/bin/env python3

# import model from file
# take wav file as argument input
# extract data from wav file

# use model and data to predict characters


# import sys for argument inputs
import sys

# import functions from data training file
from extractFeatures import read_wav, detect_clicks

# import functions from model file
from model import predict, prune, decode_labels

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
		print("Usage: predict <model.pkl> <filename.wav>")
		print("Using the given model, it reads the wav file and predicts the keys")
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
