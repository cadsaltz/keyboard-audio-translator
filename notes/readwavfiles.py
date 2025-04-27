import argparse
import wave
import pyaudio

def read_wav(file_path):
    """Read a WAV file and print basic information."""
    with wave.open(file_path, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        bit_depth = wav_file.getsampwidth() * 8
        duration = wav_file.getnframes() / sample_rate

        print(f"Channels: {channels}")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Bit Depth: {bit_depth}-bit")
        print(f"Duration: {duration:.2f} seconds")

def record_audio(output_path, duration=10):
    """Record audio and save to a WAV file."""
    chunk = 1024  # Record in chunks of 1024 samples
    format = pyaudio.paInt16  # 16-bit audio
    channels = 1  # Mono audio
    sample_rate = 44100  # Sample rate in Hz

    audio = pyaudio.PyAudio()

    print("Recording...")
    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

    frames = []

    try:
        for _ in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
    except KeyboardInterrupt:
        print("\nRecording stopped manually.")

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording to a WAV file
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(audio.get_sample_size(format))
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b''.join(frames))

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process or record audio.")
    parser.add_argument("file", nargs="?", help="Path to WAV file")
    args = parser.parse_args()

    if args.file:
        # If a file is provided, read it
        read_wav(args.file)
    else:
        # If no file is provided, record audio
        record_audio("recording.wav", duration=10)






"""
How the Code Works
Command-Line Arguments:
Accepts an optional file argument to specify the WAV file.
Recording:
Records a 10-second audio clip (adjustable) if no file is provided.
Saves the recording as recording.wav.
Reading WAV Files:
Reads the WAV file and prints its properties (channels, sample rate, bit depth, duration).
"""
