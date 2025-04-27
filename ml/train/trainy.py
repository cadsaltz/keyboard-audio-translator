#!/usr/bin/env python3

# take runtime input and dump into parallel csv file with y data

# import csv and os for csv file importing/exporting
import csv
import sys
import os

# import numpy for data manipulation
import numpy as np

# function to append a string to a csv file
def append_keys_to_file(labels, filename):

	# make a list to store the keys
	data = []

	# check if the file exists
	file_exists = os.path.isfile(filename)

	# open the given file
	with open(filename, 'a', newline='') as csvfile:

		# initialize a csv writer
		writer = csv.writer(csvfile)
		
		# initialize the order to zero
		order=0

		# if the file is new, add a header
		if not file_exists:

			# label (character for the parallel feature lsit), order number for alignment
			header = ['label', 'check']

			# write the header
			writer.writerow(header)

		# for every character in the given string
		for key in labels:
	
			# create a vecetor for the matrix (list with two elements)
			vector = [key, order]

			# append it to the data matrix
			data.append(vector)

			# increment the order each vector
			order+=1

		# make the data list a numpy array
		np.array(data)

		# for every row (list/vector) in the data matrix
		for row in data:

			# add it to the csv file
			writer.writerow(row)

# main function
def main():

	if len(sys.argv) == 2:
		keys = sys.argv[1]
	else:
		print("Usage: python3 trainy.py <word>")
		exit()

	# and add them to the csv file
	append_keys_to_file(keys, "labels.csv")


if __name__ == "__main__":
	main()


