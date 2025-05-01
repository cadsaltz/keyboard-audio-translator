
import sys

print(len(sys.argv))

if (len(sys.argv) == 1):

	filename = "filename.txt"

if (len(sys.argv) == 2):

	filename = sys.argv[1]

if (len(sys.argv) > 2):

	print("Usage: exe <filename.wav>")
	exit()


print(filename)
