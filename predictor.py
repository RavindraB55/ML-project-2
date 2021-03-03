from numpy import loadtxt
from keras.models import load_model
import PIL
from PIL import Image
import numpy as np
import sounddevice as sd
import soundfile as sf
from playsound import playsound
# load model
model = load_model('model.h5')
# summarize model.
model.summary()

testing_input = np.array(Image.open('C:/Python38/CNN/rooms/JPG/acB1.jpg').convert('L').resize((50, 60)))
test_X = testing_input.reshape(-1, 60, 50, 1)
test_X = test_X.astype('float32')
test_X = test_X/255.

testing_output = model.predict(test_X)

result = testing_output[0]
print(result.shape)
print(result)

main_sound = 'C:/Python38/CNN/apple_iphone_6.WAV'
data, fs = sf.read(main_sound, dtype='float32')
print(fs)
iphone_mono = []

for i in range(len(data)):
    d = ((data[i][0]) / 2 + (data[i][1]) / 2)
    iphone_mono.append(d)

iphone_mono = np.array(iphone_mono, dtype='float32')
print(iphone_mono.shape)

new_signal = np.convolve(iphone_mono, result, mode='full')
print(new_signal.shape)
sd.play(new_signal, fs)
status = sd.wait()
