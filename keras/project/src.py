import keras
from keras.datasets import mnist # mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K

import random
import numpy as np

def create_bags(input_data, labels, max_batch_size):

    # output probabilities
    # the_y_probs = keras.utils.to_categorical(y_train, num_classes)
    the_y_probs = None

    # get next `max_batch_size' pieces of the `input_data'
    input_data_length = len(input_data)
    for lower_bound in range(0, input_data_length, max_batch_size):

        # Check the top limit
        if lower_bound + max_batch_size >= input_data_length:
            upper_bound = input_data_length
        else:
            upper_bound = lower_bound + max_batch_size

        the_batch = input_data[lower_bound : upper_bound]

        # Find the probability of each class
        the_labels = labels[lower_bound : upper_bound]
        the_probs = sum(the_labels) / max_batch_size # an array [y0, y1, ...]
        #print(the_probs)
        the_probs = the_probs.reshape(1, -1)
        #print("reshape:", the_probs)
        the_probs = np.repeat(the_probs, upper_bound - lower_bound, axis=0)
        #print("repeat:", the_probs)
        # the_y_probs.extend(the_probs)
        if the_y_probs is None:
            the_y_probs = the_probs
        else:
            the_y_probs = np.append(the_y_probs, the_probs, axis=0)
            # raise Exception

    return the_y_probs


batch_size = 16
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train = create_bags(x_train, y_train, batch_size)
y_test = create_bags(x_test, y_test, batch_size)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# model.add(BatchNormalization()) # not so good for this dataset; drops accuracy from 0.98 to 0.92

model.compile(loss=keras.losses.kullback_leibler_divergence, # using KL-divergence
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# for epoch in epochs:



score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
