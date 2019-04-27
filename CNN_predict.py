from keras.models import load_model
import os
from keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope
from keras.preprocessing import image
import numpy as np
import time
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('model_1.h5')
print("Model loading is over")
input_size = (96, 96)
test_folderpath = 'D:/Hearts of Iron III/dowmloads/CERTH_ImageBlurDataset/AdditionalDataset/'
start = time.time()
# check method efficiency on the same stuff as the classic CV method
for filename in os.listdir(test_folderpath):
    impath = test_folderpath + filename
    img = image.load_img(impath, color_mode="grayscale", target_size=input_size)
    img = (1 / 255) * np.asarray(img)
    img = img.reshape(1, 96, 96, 1)
    pred = model.predict(img, verbose=1)
    # debugging printing
    print(pred)
    if pred[0][0] > pred[0][1]:
        print(filename + " was defined as a sharp one")
    else:
        print(filename + " was defined as a blurry one")
end = time.time()
print("Elapsed time: ", end-start)
