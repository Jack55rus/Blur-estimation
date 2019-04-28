import cv2
import numpy as np
# noinspection PyUnresolvedReferences
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from imutils import paths
import pandas as pd
import time
import os


def isBlurred(img, th):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(img, cv2.CV_32F).var() > th:
        return False
    else:
        return True

threshold = 50  # defined empirically, might be changed

# The following commented block is used to evaluate accuracy of the method
dg_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx'
nat_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx'
dgbset = pd.read_excel(dg_folderpath)
natbset = pd.read_excel(nat_folderpath)
digeval = {}
nateval = {}
dg_im_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'
nat_im_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'
tn = 0  # true negative
tp = 0  # true positive
fn = 0  # false negative
fp = 0  # false positive
start = time.time()
for i in range(0, 1000):
    filename = natbset['Image Name'][i]
    imagepath = nat_im_folderpath + filename + '.jpg'
    image = cv2.imread(imagepath)
    if not (isBlurred(image, threshold)) and natbset['Blur Label'][i] == -1:
        tn += 1
    elif not (isBlurred(image, threshold)) and natbset['Blur Label'][i] == 1:
        fn += 1
    elif (isBlurred(image, threshold)) and natbset['Blur Label'][i] == -1:
        fp += 1
    elif (isBlurred(image, threshold)) and natbset['Blur Label'][i] == 1:
        tp += 1
    if i % 200 == 0:
        print(i/1000*100)

for i in range(0, 480):
    filename = dgbset['MyDigital Blur'][i]
    imagepath = dg_im_folderpath + filename
    image = cv2.imread(imagepath)
    if not (isBlurred(image, threshold)) and dgbset['Unnamed: 1'][i] == -1:
        tn += 1
    elif not (isBlurred(image, threshold)) and dgbset['Unnamed: 1'][i] == 1:
        fn += 1
    elif (isBlurred(image, threshold)) and dgbset['Unnamed: 1'][i] == -1:
        fp += 1
    elif (isBlurred(image, threshold)) and dgbset['Unnamed: 1'][i] == 1:
        tp += 1
end = time.time()
# Metrics calculation:
print("Elapsed time: ", end - start)
Precision = tp/(tp+fp)
Recall = tp/(tp+fn)
F = 2*Precision*Recall/(Precision + Recall)
MCC = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
print("Precision = ", Precision)
print("Recall = ", Recall)
print("F-measure = ", F)
print("MCC = ", MCC)

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

'''
# this part was used to write data to the excel file, not relevant anymore
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
