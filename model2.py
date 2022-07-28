# CNN for my voice classification
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from scipy.io import wavfile
from pydub import AudioSegment
from scipy import signal
import numpy as np
import random
import glob
import wave
import os

# Parameters
num_classes = 2

### DATASET 
data = []
labels = []

audio_files = [f for f in glob.glob(os.path.abspath(r"Data")+"/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(audio_files)

# My voice data
for path in audio_files:

    filename = str(path)

    samplerate, data_array = wavfile.read(filename)
    # Covert audio into image spectogram
    frecuencies, times, spectrogram = signal.spectrogram(data_array, samplerate)

    #print(spectogram) is alreay a np array
    data.append(spectrogram)

    label = path.split(os.path.sep)[-2]
    if label == "me":
        label = 1
    else:
        label = 0

    labels.append([label])


# ADAPTING THE DATA FOR THE MODEL
X = np.array(data) # all voices data
y_ = np.array(labels)

y = to_categorical(y_, num_classes=num_classes)


# CNN MODEL
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(129, 1071, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
print(model.summary())

model.fit(X, y, batch_size=1, epochs=10, verbose=1)

model.save('D:/Machine Learning MODELS/DOT/voiceclassify2.model')


