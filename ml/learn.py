
#!/usr/bin/env python3


# load csv file x data
# load csv file y data

# train model from data

# dump model into file

# import sys for argument inputs
import sys

# import os and csv for file ingestion
import os
import csv

# import numpy for data manipulation
import numpy as np

#import pickle for model dumping/importing
import pickle

# import ml library
from sklearn.ensemble import RandomForestClassifier

# function to ingest feature data from csv file
def load_feature_matrix(filename):

	# open the file
	with open(filename, 'r') as csvfile:

		# read the file
		reader = csv.reader(csvfile)

		# skip the first line (header)
		next(reader)

		# ingest each line as a list and build the feature matrix
		data = [list(map(float, row)) for row in reader]
	
	# return a np matrix (features + order)
	return np.array(data)

# function to ingest lables from csv file
def load_labels_matrix(filename):

	# open the file
	with open(filename, 'r') as csvfile:

		# read the file
		reader = csv.reader(csvfile)

		# skip the first line (header)
		next(reader)

		# ingest each line as a list and build the label matrix
		labels = [row for row in reader]

	# return a np matrix (label + order)
	return np.array(labels)

# function to validate alignment between csv files
# this is important because the model needs to learn from consistent data and aligned matrices
def prune_align(features, labels):

	# if the length of the files is different, print error and return empty matrices. 
	# label and features need to be 1:1
	if len(features) != len(labels):
		print("Feature and label matrices have different lengths")
		return [], []

	# create empty matrices to hold the pruned versions
	pruned_features = []
	pruned_labels = []

	# for every line in the csv files
	for i in range(len(features)):

		# grab the row (list)
		feature_row = features[i]
		label_row = labels[i]

		# grab the order parameter
		feature_order = feature_row[-1]
		label_order = float(label_row[-1]) if len(label_row) > 1 else None

		# if the orders are missaliged
		if feature_order != label_order:

			# print error and return empty matrices
			# labels and features need to have consistent relation
			print("Order mismatched")
			return [], []

		# add everything from the list except the order value 
		#the model would rely on the order instead of the acutal feature data
		pruned_features.append(feature_row[:-1])
		pruned_labels.append(label_row[0])

	# return the pruned matrices
	return np.array(pruned_features), np.array(pruned_labels)

# function to encode the labels 
# the model expects numbers, not characters
# im only taking lower case letters and the space bar as keys to consider
def encode_labels(labels):

	# so just map the letters from 0-26 including space
	mapping = {' ':0}

	# map all the letters
	for idx, letter in enumerate('abcdefghijklmnopqrstuvwxyz', start=1):
		mapping[letter] = idx

	# create the encoded list
	encoded = [mapping[label] for label in labels]

	# return the numeric version of the labels
	return np.array(encoded)

# function to decode the numeric version of the labels
def decode_labels(encoded_labels):
	
	# create the mapping
	mapping = {0:' '}

	# map all the numbers
	for idx, letter in enumerate('abcdefghijklmnopqrstuvwxyz', start=1):
		mapping[idx] = letter

	# create the decoded list
	decoded = [mapping[num] for num in encoded_labels]
	
	# return the alphabetic version of the labels 
	# it can just be a list, doesnt need to be a np.array
	return decoded

# function to create the model
def learn(features, labels, model_filename):
	
	
	if len(labels.shape) > 1:
		labels = labels.ravel()

	# init the model
	model = RandomForestClassifier(n_estimators=100)

	# fit the model to the data and labels
	model.fit(features, labels)

	# open a file to save the model to
	with open(model_filename, 'wb') as f:

		# dump the model to the file
		pickle.dump(model, f)

	# inform user
	print("Model saved to", model_filename)

# main function
def main():

	# parse system arguments
	if len(sys.argv) == 2:

		# grab their input and use it as the model file to import
		filename = sys.argv[1]

	# anything else is unacceptable
	else:

		# inform user and exit
		print("Usage: python3 predict.py <filename.pkl>")
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
