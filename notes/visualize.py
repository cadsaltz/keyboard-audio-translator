import numpy as np
import matplotlib.pyplot as plt

def visualize_audio(file_path):
    """Visualize the waveform of a WAV file."""
    with wave.open(file_path, 'rb') as wav_file:
        # Extract audio parameters
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration = n_frames / sample_rate

        # Read the audio frames
        audio_data = wav_file.readframes(n_frames)
        # Convert byte data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # If stereo, split into two channels (left and right)
        if channels == 2:
            audio_array = audio_array.reshape(-1, 2)
            audio_array = audio_array.mean(axis=1)  # Convert to mono by averaging channels

        # Create a time array for the x-axis
        time = np.linspace(0, duration, num=len(audio_array))

        # Plot the waveform
        plt.figure(figsize=(10, 6))
        plt.plot(time, audio_array, color='blue')
        plt.title("Audio Waveform")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process or record audio.")
    parser.add_argument("file", nargs="?", help="Path to WAV file")
    args = parser.parse_args()

    if args.file:
        # If a file is provided, read and visualize it
        read_wav(args.file)
        visualize_audio(args.file)
    else:
        # If no file is provided, record audio and visualize it
        output_path = "recording.wav"
        record_audio(output_path, duration=10)
        visualize_audio(output_path)

