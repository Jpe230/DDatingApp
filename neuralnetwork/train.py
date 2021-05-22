import tensorflow as tf
import keras.backend as K
import pickle
import numpy as np

from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

# User-defined const
import const

# Needs to be here if using NVIDIA GPU, otherwise model wouldnt fit
tf.logging.set_verbosity(tf.logging.ERROR)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Use ResNet50 as our base NN
resnet = ResNet50(include_top=False, pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.layers[0].trainable = False

# Print our model
print(model.summary())

# Define and Compile our model
sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='kld', optimizer=sgd, metrics=['accuracy'])

# Load our training data
label_dist = pickle.load(open(const.TRAINING_FILE, 'rb'))

# Separate label + imgs
train_X = np.array([x[1]
                   for x in label_dist[0:len(label_dist)]])
train_Y = np.array([x[2]
                   for x in label_dist[0:len(label_dist)]])

# Add early stopping, in general it stops around 50 epochs
earlyStopping = EarlyStopping(
    monitor='val_loss', patience=15, verbose=0, mode='auto')

# Train NN for 100 epochs
history = model.fit(x=train_X, y=train_Y, batch_size=32, callbacks=[
                    earlyStopping], epochs=100, verbose=1, validation_split=0.1)

# Save our trained weights
model.save_weights(const.TRAINEDMODEL_FILE)
