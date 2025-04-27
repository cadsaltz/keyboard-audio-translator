import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load a key sound
y, sr = librosa.load("key_A.wav", sr=None)

# Plot the spectrogram
spectrogram = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram of Key A")
plt.show()

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print("MFCCs of Key A:", mfccs.mean(axis=1))

