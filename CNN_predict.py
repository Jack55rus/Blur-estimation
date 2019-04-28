from keras.models import load_model
import os
from keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import numpy as np
import time
import pandas as pd
import pickle

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('model_1.h5')
print("Model loading is over")
input_size = (96, 96)
start = time.time()
with open('X_test.pkl', 'rb') as picklefile:
    X_test = pickle.load(picklefile)
with open('y_test.pkl', 'rb') as picklefile:
    y_test = pickle.load(picklefile)
testdata = np.stack(X_test)
testdata = testdata.reshape(1480, 96, 96, 1)
preds = model.predict(testdata, verbose=1)
# identifying other metrics
tn = 0  # true negative
tp = 0  # true positive
fn = 0  # false negative
fp = 0  # false positive
for i in range(0, 1480):
    if (preds[i][0] > preds[i][1] and y_test[i] == 0):
        tn += 1
    elif (preds[i][0] > preds[i][1] and y_test[i] == 1):
        fn += 1
    elif (preds[i][0] < preds[i][1] and y_test[i] == 0):
        fp += 1
    elif (preds[i][0] < preds[i][1] and y_test[i] == 1):
        tp += 1
end = time.time()
print("Elapsed time: ", end - start)
Precision = tp/(tp+fp)
Recall = tp/(tp+fn)
F = 2*Precision*Recall/(Precision + Recall)
MCC = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
print("Precision = ", Precision)
print("Recall = ", Recall)
print("F-measure = ", F)
print("MCC = ", MCC)
