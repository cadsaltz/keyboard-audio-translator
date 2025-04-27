from pynput import keyboard
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import threading

typed_chars = []

def on_press(key):
	try:
		# Try to get the character
		typed_chars.append(key.char)
	except AttributeError:
		if key == keyboard.Key.space:
			typed_chars.append(' ')
		elif key == keyboard.Key.enter:
			# Maybe stop on enter?
			pass
		# Handle special keys if needed

def record_audio(duration, fs):
	print("Recording audio...")
	audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
	sd.wait()  # Wait until recording is finished
	return audio

def main():
	fs = 44100  # Sample rate
	duration = 10  # seconds (or however long you want)

	# Start key listener in a separate thread
	listener = keyboard.Listener(on_press=on_press)
	listener.start()

	# Record audio
	audio_data = record_audio(duration, fs)

	listener.stop()

	# Save audio
	wavfile.write('output.wav', fs, audio_data)

	# Save typed characters
	with open('output.txt', 'w') as f:
		f.write(''.join(typed_chars))

	print("Recording complete. Saved output.wav and output.txt")

if __name__ == "__main__":
	main()

