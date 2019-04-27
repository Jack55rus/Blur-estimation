import numpy as np
import pandas as pd
import os
import pickle

from keras.preprocessing import image
input_size = (96, 96)
X_test = []
y_test = []
dgbset = pd.read_excel('D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
nbset = pd.read_excel('D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')
dgbset['MyDigital Blur'] = dgbset['MyDigital Blur'].apply(lambda x : x.strip())
dgbset = dgbset.rename(index=str, columns={"Unnamed: 1": "Blur Label"})
nbset['Image Name'] = nbset['Image Name'].apply(lambda x : x.strip())
folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'

for filename in os.listdir(folderpath):
    imagepath = folderpath + filename
    img = image.load_img(imagepath, color_mode="grayscale", target_size=input_size)
    X_test.append((1/255)*np.asarray(img))
    blur = dgbset[dgbset['MyDigital Blur'] == filename].iloc[0]['Blur Label']
    if blur == 1:
        y_test.append(1)
    else:
        y_test.append(0)

print("Testset: Artificially Blurred loaded...")

folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'
for filename in os.listdir(folderpath):
    imagepath = folderpath + filename
    img = image.load_img(imagepath, color_mode="grayscale", target_size=input_size)
    X_test.append((1/255)*np.asarray(img))
    blur = nbset[nbset['Image Name'] == filename.split('.')[0]].iloc[0]['Blur Label']
    if blur == 1:
        y_test.append(1)
    else:
        y_test.append(0)

print("Trainset: Naturally Blurred loaded...")
with open('X_test.pkl', 'wb') as picklefile:
    pickle.dump(X_test, picklefile)

with open('y_test.pkl', 'wb') as picklefile:
    pickle.dump(y_test, picklefile)
