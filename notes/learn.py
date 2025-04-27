# Python 3
# 
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



def mapChars():

	# sample data + sample solution > prototype function
	# actual data + refined function > attempted solution
	print()

def main():
	

	if (len(sys.argv) > 3):
		
		print("Too many Arguments. Usage: \nmapping <file.mp3> <file.txt>")
		return 1

	elif (len(sys.argv) == 3):
		
		print("learning from data")
	
		soundFileName=sys.argv[1]
		solutionFileName=sys.argv[2]

	else:
	# if they didnt input anything, invalid usage
	
		print("Too few Arguments. Usage: \nmapping <file.mp3> <file.txt>")
		return 1


	# then save the learned function toa file to be used by the guess function
	print("learning complete. Mapping saved to <function.py>")
	
# main function
if __name__ == "__main__":
	main()
	
