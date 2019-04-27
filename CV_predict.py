import cv2
import numpy as np
# noinspection PyUnresolvedReferences
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from imutils import paths
import pandas as pd
import time
import os


def lapl_var_gray(im_2g):
    im_2g = cv2.cvtColor(im_2g, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(im_2g, cv2.CV_32F).var()


threshold = 50  # defined empirically
start = time.time()
'''
#The following commented block is used to evaluate accuracy of the method
dg_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx'
nat_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx'
dgbset = pd.read_excel(dg_folderpath)
natbset = pd.read_excel(nat_folderpath)
digeval = {}
nateval = {}
dg_im_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'
nat_im_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'
Sum_d = 0
Sum_n = 0
for i in range(0, 1000):
    filename = natbset['Image Name'][i]
    imagepath = nat_im_folderpath + filename + '.jpg'
    image = cv2.imread(imagepath)
    if lapl_var_gray(image) > threshold:
        nateval[filename] = -1
    else:
        nateval[filename] = 1
    if nateval[filename] == natbset['Blur Label'][i]:
        Sum_n += 1

for i in range(0, 480):
    filename = dgbset['MyDigital Blur'][i]
    imagepath = dg_im_folderpath + filename
    image = cv2.imread(imagepath)
    if lapl_var_gray(image) > threshold:
        digeval[filename] = -1
    else:
        digeval[filename] = 1
    if digeval[filename] == dgbset['Unnamed: 1'][i]:
        Sum_d += 1
# Accuracy calculation:
print("Digital blur accuracy: ", Sum_d/480*100)
print("Natural blur accuracy: ", Sum_n/1000*100)
print("Total accuracy of the method: ", (Sum_d+Sum_n)/1480*100)
'''
# this block is used to check how the algorithm works on interesting images and measure the elapsed time
test_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/AdditionalDataset/'
for filename in os.listdir(test_folderpath):
    impath = test_folderpath + filename
    image = cv2.imread(impath)
    # debugging printing
    if lapl_var_gray(image) > threshold:
        print(filename + " was defined as a sharp one ", lapl_var_gray(image))
    else:
        print(filename + " was defined as a blurry one ", lapl_var_gray(image))
end = time.time()
print("Elapsed time: ", end - start)

'''
# this part was used to write data to the excel file
dval = []
nval = []
for key in digeval:
    dval.append(digeval[key])

dval = [digeval[key] for key in digeval]
dval = [nateval[key] for key in nateval]

for i in range(0, len(dval)):
    df1 = pd.DataFrame(dval,
                       # index=range(0, 5),
                       columns=['label'])
df1.to_excel("output_nat.xlsx") 
'''
