from numpy import loadtxt
from keras.models import load_model
import PIL
from PIL import Image
import numpy as np
import sounddevice as sd
import soundfile as sf
from playsound import playsound
import sys

# load model
model = load_model('model.h5')
# summarize model.
# model.summary()
np.set_printoptions(threshold=sys.maxsize)

testing_input = np.array(Image.open('C:/Python38/CNN/rooms/JPG/acB1.jpg').convert('L').resize((50, 60)))
test_X = testing_input.reshape(-1, 60, 50, 1)
test_X = test_X.astype('float32')
test_X = test_X/255.

testing_output = model.predict(test_X)

print(testing_output.shape)
result = testing_output[0]
print(result.shape)
#print(result)
print(type(result))

print('-------------------')
filename = 'IRs/bcB22.WAV'
data, fs = sf.read(filename, dtype='float32')
print(data.shape)
print(type(data))
print(data)

print(type(result[0]))
print(type(data[0]))

sf.write('new_file.WAV', result, 44140)
