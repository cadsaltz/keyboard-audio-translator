
from collections import Counter
import string
import sys

def count_letters(file_path):
	letter_counter = Counter()

	with open(file_path, 'r', encoding='utf-8') as file:
		for line in file:
			cleaned_line = ''.join(char for char in line.lower() if char in string.ascii_lowercase)

			letter_counter.update(cleaned_line)

	full_counter = {letter: letter_counter.get(letter, 0) for letter in string.ascii_lowercase}

	return full_counter

def main():
	
	if len(sys.argv) == 2:
		filename = sys.argv[1]

	else:
		print("Usage: python3 lcount.py <filename>")
		exit()

	counts = count_letters(filename)

	for letter, count in counts.items():
		print(letter, ": ", count)

if __name__ == "__main__":
	main()
