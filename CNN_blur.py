import numpy as np
import pickle

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical

input_size = (96, 96)

with open('X_train.pkl', 'rb') as picklefile:
    X_train = pickle.load(picklefile)

with open('y_train.pkl', 'rb') as picklefile:
    y_train = pickle.load(picklefile)

with open('X_test.pkl', 'rb') as picklefile:
    X_test = pickle.load(picklefile)

with open('y_test.pkl', 'rb') as picklefile:
    y_test = pickle.load(picklefile)
model = Sequential()
# Layer 1
model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1), border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Layer 2
model.add(Conv2D(64, (3, 3), border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Layer 3
model.add(Conv2D(128, (3, 3), border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# Layer 4
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.2))
# Layer 5
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.2))
# Layer 6
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
# Layer 7
model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

traindata = np.stack(X_train)
testdata = np.stack(X_test)
trainlabel = to_categorical(y_train)
testlabel = to_categorical(y_test)

traindata = traindata.reshape((1142, 96, 96, 1))
testdata = testdata.reshape((1480, 96, 96, 1))

model.fit(traindata, trainlabel, batch_size=32, epochs=10, validation_split=0.1, verbose=1)
print("Model training complete...")
(loss, accuracy) = model.evaluate(testdata, testlabel, batch_size=32, verbose=1)
print("accuracy: {:.2f}%".format(accuracy * 100))
print(model.summary())

model.save('model_1.h5')
print("The model is saved")
