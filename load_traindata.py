import numpy as np
import pandas as pd
import os
import pickle


from keras.preprocessing import image

X_train = []
y_train = []

input_size = (96, 96)

folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/TrainingSet/Undistorted/'

# load image arrays
for filename in os.listdir(folderpath):
    imagepath = folderpath + filename
    img = image.load_img(imagepath, color_mode="grayscale", target_size = input_size)
    X_train.append((1/255)*np.asarray(img))
    y_train.append(0)
print("Trainset: Undistorted loaded...")

folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred/'

# load image arrays
for filename in os.listdir(folderpath):
    imagepath = folderpath + filename
    img = image.load_img(imagepath, color_mode="grayscale", target_size=input_size)
    X_train.append((1/255)*np.asarray(img))
    y_train.append(1)
print("Trainset: Artificially Blurred loaded...")


folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred/'

# load image arrays
for filename in os.listdir(folderpath):
    imagepath = folderpath + filename
    img = image.load_img(imagepath, color_mode="grayscale", target_size=input_size)
    X_train.append((1/255)*np.asarray(img))
    y_train.append(1)
print("Trainset: Naturally Blurred loaded...")

# Pickle the train files

with open('X_train.pkl', 'wb') as picklefile:
    pickle.dump(X_train, picklefile)

with open('y_train.pkl', 'wb') as picklefile:
    pickle.dump(y_train, picklefile)
