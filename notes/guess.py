#!/usr/bin/env python3
# Python 3
# 
#
#
#
# PSUEDO
#
# Two Parts:
# 	Get Data
#	Interpret Data
#
# Get Data:
# Use argument input to read MP3 file
# If no input is given, it should start recording. Once the recording stops, then take the input from the MP3 file it just recorded
#
# Then clean, filter, enhance, isolate sound - this will be the meat of the project
# loads of data manipulation
#
# 
# Interpret Data:
# Either use ML or manual mapping to map/associate the sound of the keys to the respected character
# 	Map common or distinct letters first: map by process of elimination. if we are certain that we have a successful mapping, we can eliminate it from the idk pile
# 		spacebar, double letters, ngrams 
#  with the mapping and clean audio, translate the sounds to the respected characters
# 
# 
# Add a counter for debugging and certainty. count the number of keys the file contains and print that value along with the translation
# Add some statistics like words per second, characters per second, certainty of accuracy
#
#
# Extra points:
# 	actively learn and map sounds of the keys to characters based on frequency, prediction, and text editing. 
# 	Consider the following keys and possibly ignore them: ctr, shift and capital letters, backspace/delete, arrow keys, tab, caps lock
#
# 	frequency: use the standard frequencies of all characters to map. - this will need a very large sample size to get good mapping (uses averages)
# 		should be fast to map space bar. has a significantly different sound than other keys and is extremely frequent. 
#		keys that are typed after spacebar are the first letter of a word. I can use a different frequency list to map the first letter of a word. 
#
# 	prediction: use ngrams and rules to map certain letters. 
# 		th uses the same two keys and will generate the same sound every time. 
#		q is always followed by u so if we can map q, we can map u
#		use double letters to map them faster. characters like oo and ll make a significantly different sound than a spread of letters 
#			they have a shorter period between sounds compared to a spread of keys.
#
#	text editing: use something like grammarly to edit the translated text and help with mapping
# 	
#	Mobility: 
# 		Make the program mobile and accessable to learn other keyboards and other people's typing style.
#
# Machine Learning
#
#	ML needs to learn, then perform. it needs to be trained with data and a result (sounds and text). it will try to figure out how to go from input to output. it will refine this with repitition and fitness. save the function once the model has found a consistent and mobile function that maps sounds to text. 
# use the function it generated to try and guess text from the sounds the text made. 
#
#
# 	sample data > prototype function > sample solution
#
# 	actual data > refined function > attempted solution
#
#
#

import sys



# use a separate program to train the ML model
# import from program or compile together, something like that


def translate():

	# use the mapping to associate the data from the file to the characters that were typed.
	print()

def main():
	

	if (len(sys.argv) > 2):
		
		print("Too many Arguments. Usage: \nguess <> (To record)\nguess <file.mp3> (To read)")
		return 1

	elif (len(sys.argv) == 2):
		
		print("Read from file")
	
		fileName=sys.argv[1]

	elif (len(sys.argv) == 1):
	# if they didnt input anything, start recording and use the recording to read from
	
		# debugging print
		print("Record from device/stdin")
		
		# RECORD HERE

		# set file to read from as the file that was just created
#		fileName=recordedFile


	# then attempt to translate the file
	#translate(fileName)	
	
	# then print translation
	print("Translated Text: Mm thats was some good eats")
	
# main function
if __name__ == "__main__":
	main()
	
