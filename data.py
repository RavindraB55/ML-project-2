# Machine Learning Architecture Project 2
# Ravindra Bisram, Matthew Chan, Kevin Chow, Miho Takeuchi
import os
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import sounddevice as sd
import soundfile as sf
import sys
from playsound import playsound

# Importing the Keras packages that can be used to structure the CNN
import keras
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam

# Print out full array (useful for debugging)
np.set_printoptions(threshold=sys.maxsize)

# Get the name of the working directory where all the pictures are stored
dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
print()
print("Program beginning!")
# print(cwd)

# Script to convert the PDF files to JPGs (the files were originally sent in PDF form)
# Requires poppler to be installed before hand
def convert_PDF_to_JPG():
    for subdir, dirs, files in os.walk(cwd):
        for file in files:
            # This is just the name of the file (letters/numbers) without path or extension
            name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[-1].lower()
            if ext == '.pdf':
                full_path = os.path.join(subdir, file)
                pdf_list.append(full_path)
                converting_pdf = convert_from_path(full_path)
                new_filename = cwd+'/rooms/JPG/'+name+'.jpg'
                converting_pdf[0].save(new_filename, 'JPEG')
                print(name, ' uploaded')


# Script to iterate through the JPGs and make sure there is a corresponding WAV file
# Confirmation that we only use data that has a corresponding pair
# Note, some path names here were hardcoded - would need to edit in a personal deployment
def confirm_JPG_WAV(image_array, soundfile_array):
    for subdir, dirs, files in os.walk(cwd+'/rooms/JPG'):
        for file in files:
            name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[-1].lower()
            if ext == '.jpg':
                image_name = 'C:/Python38/CNN/rooms/JPG/' + file
                soundfile_name = 'C:/Python38/CNN/IRs/' + name + '.WAV'
                if os.path.exists(soundfile_name):
                    # print("We got a match!: ", image_name, soundfile_name)
                    image_array.append(image_name)
                    soundfile_array.append(soundfile_name)
                else:
                    # print("No match: ", image_name)
                    continue
    return image_array, soundfile_array


# Create the data arrays (features and labels)
def create_data(image_array, soundfile_array, func_features, func_labels, func_sampling_rates):
    for i in range(len(image_array)):
        # print(i)
        func_features.append(np.array(Image.open(image_array[i]).convert('L').resize((50, 60))))

        data, fs = sf.read(soundfile_array[i], dtype='float32')
        func_labels.append(data)
        func_sampling_rates.append(fs)

    return func_features, func_labels, func_sampling_rates


# Some wav files were randomly in stereo form, this function converts them to mono (a vector)
def stereoToMono(audio_data):
    newaudiodata = []
    for i in range(len(audio_data)):
        d = ((audio_data[i][0])/2 + (audio_data[i][1])/2)
        newaudiodata.append(d)

    return np.array(newaudiodata, dtype='float32')


# Make sure all the label arrays are the same dimensions
# This ensures all labels are the exact same shape, mono vectors of the correct length, padding is used
def fixing_labels(original_label_array, new_label_array):
    size_array = []
    temp_array = []
    for item in original_label_array:
        if len(item.shape) == 2:
            new_item = stereoToMono(item)
            temp_array.append(new_item)
        else:
            temp_array.append(item)

        size_array.append(item.shape[0])
    print(max(size_array))

    # The longest vector was used as the reference for all others to match, but this is an arbitrary choice
    for item in temp_array:
        if len(item) != max(size_array):
            new_label_array.append(np.pad(item, (0, max(size_array)-len(item)), mode='constant'))
        else:
            new_label_array.append(item)

    return new_label_array


# Uncomment this function if needed to run
# convert_PDF_to_JPG()

# Preallocate the relevant list names
images = []
soundfiles = []
features = []
labels = []
sampling_rates = []

images, soundfiles = confirm_JPG_WAV(images, soundfiles)
# print(len(images), len(soundfiles))

features, labels, sampling_rates = create_data(images, soundfiles, features, labels, sampling_rates)

# Sanity check on the sampling rates, see if they are all the same
if sampling_rates.count(sampling_rates[0]) == len(sampling_rates):
    print("All the sampling rates are the same (thank god)")

new_labels = []
new_labels = fixing_labels(labels, new_labels)

# Reformatting Data to be compliant with the necessary keras form
feat_arr = np.array(features)
label_arr = np.array(new_labels)

train_X = feat_arr.reshape(-1, 60, 50, 1)
print(train_X.shape)
train_X = train_X.astype('float32')
train_X = train_X/255.
train_Y = label_arr

# Divide into testing, training, and validation data
train_X, temp_X, train_label, temp_label = train_test_split(train_X, train_Y, test_size=0.1, random_state=13)
valid_X, test_X, valid_label, test_label = train_test_split(temp_X, temp_label, test_size=0.5, random_state=13)

# parameters for training the model
batch_size = 64
epochs = 30
num_classes = 90698
print('num classes = ', num_classes)

# Compile using the Adam optimization algorithm
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

room_model = Sequential()
room_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(60, 50, 1)))
room_model.add(LeakyReLU(alpha=0.1))
room_model.add(MaxPooling2D((2, 2), padding='same'))
room_model.add(Dropout(0.25))
room_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
room_model.add(LeakyReLU(alpha=0.1))
room_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
room_model.add(Dropout(0.25))
room_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
room_model.add(LeakyReLU(alpha=0.1))
room_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
room_model.add(Dropout(0.4))
room_model.add(Flatten())
room_model.add(Dense(128, activation='linear'))
room_model.add(LeakyReLU(alpha=0.1))
room_model.add(Dropout(0.3))
room_model.add(Dense(num_classes, activation='softmax'))

room_model.summary()

room_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print("----------------------------------")
print(train_X.shape)
print(train_label.shape)
print("----------------------------------")

room_train_dropout = room_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))

print(test_X.shape)

# test_eval = fashion_model.evaluate(test_X, test_label, verbose=1)
# print('Test loss:', test_eval[0])
# print('Test accuracy:', test_eval[1])

# Test the model out, but the real testing is done in a separate file
predicted_classes = fashion_model.predict(test_X)
# print(predicted_classes)

# Export the model so it can be used without having to retrain every time
fashion_model.save("model.h5")
print("Saved model to disk")

# Play the output impulse response
sample_sound = predicted_classes[0]
print(sample_sound.shape)
# print(sample_sound)
print(type(sample_sound))
sd.play(sample_sound, 44140)
sf.write('new_file.WAV', sample_sound, 44140)

status = sd.wait()

# This code can be used to plot the validation vs accuracy of the model
# Was not helpful in this case since the model performed so poorly with a huge amount of labels
"""
accuracy = fashion_train_dropout.history['acc']
val_accuracy = fashion_train_dropout.history['val_acc']
loss = fashion_train_dropout.history['loss']
val_loss = fashion_train_dropout.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
"""

