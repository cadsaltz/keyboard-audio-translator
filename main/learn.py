
#!/usr/bin/env python3


# load csv file x data
# load csv file y data

# train model from data

# dump model into file

# import sys for argument inputs
import sys

# from the model file, import all necessary functions
from model import load_feature_matrix, load_labels_matrix, prune_align, encode_labels, learn

# main function
def main():

	# parse system arguments
	if len(sys.argv) == 2:

		# grab their input and use it as the model file to import
		filename = sys.argv[1]

	# anything else is unacceptable
	else:

		# inform user and exit
		print("Usage: learn <filename.pkl>")
		print("Takes the two csv files, fits a model to them, and dumps the model into the given file")
		exit()

	# ingest both csv files
	feature_matrix = load_feature_matrix("keyclick_data.csv")
	labels = load_labels_matrix("labels.csv")

	# check alignemnt and remove alignemnt value
	cleaned_features, cleaned_labels = prune_align(feature_matrix, labels)
	
	# encode the labels (convert charaters to numbers)
	encoded = encode_labels(cleaned_labels)

	# train the model
	learn(cleaned_features, encoded, filename)


if __name__ == "__main__":
	main()
