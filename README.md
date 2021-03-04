# Doodle Room Acoustics
**Team Members:** Ravindra Bisram, Matthew Chan, Kevin Chow, Miho Takeuchi

A fun exploration into using Convolutional Neural Networks to transform 2D images of room layouts into impulse responses. 

This project consists primarily of three python files: 
- data.py: contains functions to load in data, convert to proper file types, prep the features and labels, build and train the keras model, and export it into a file if desired.
- predictor.py: accepts a trained model which can then be used to predict the impulse response of a given image (doodle)
- conv.py: contains relevant functions to convolve two sound files together to produce the echo effect we are trying to replicate

## Generating Training Set
Using Grasshopper and Rhino, several two-dimensional line drawings and extruded versions of those drawings were generated. These inputs were fed into CATT Acoustics, which was then used to generate an impulse response to simulate what a certain sound would sound like in that environment.	


