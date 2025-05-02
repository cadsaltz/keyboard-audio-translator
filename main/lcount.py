
from collections import Counter
import string
import sys

# function for count all the letters using Counter
def count_letters(file_path):

	# initialize the counter
	letter_counter = Counter()

	# open the passed file
	with open(file_path, 'r', encoding='utf-8') as file:

		# for every line in the file
		for line in file:

			# clean it
			cleaned_line = ''.join(char for char in line.lower() if char in string.ascii_lowercase)

			# update the counter with the cleaned line 
			letter_counter.update(cleaned_line)

	# count all the different letters in the line
	full_counter = {letter: letter_counter.get(letter, 0) for letter in string.ascii_lowercase}

	# and spaces
	full_counter["space"] = letter_counter.get(' ', 0)

	# return the counter 
	return full_counter

# main function
def main():
	
	# take the given argument as the filename
	if len(sys.argv) == 2:

		filename = sys.argv[1]

	else:
		# anything else is unacceptable
		print("Usage: lcount <filename>")
		print("Counts the frequency of each letter in the given file")
		exit()

	# count the letters in the file
	counts = count_letters(filename)

	# print the distribution
	for letter, count in counts.items():
		print(letter, ": ", count)


if __name__ == "__main__":
	main()
