from keras import backend as K
from keras.engine.topology import Layer
# from keras.layers.core import Lambda

import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        x = K.print_tensor(x, message="the x now is: ")
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class BatchAverager(Layer):
    """BatchAverager Layer

    Idea by Ehsan Mohammady Ardehaly & Aron Culotta
    https://arxiv.org/abs/1709.04108

    Layer is built from scratch with
    https://keras.io/layers/writing-your-own-keras-layers/ Over the
    Lambda Layer

    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        # super(BatchAverager, self).build()
        super(BatchAverager, self).__init__(**kwargs)

    def build(self, input_shape):
    # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(BatchAverager, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        shape = K.int_shape(x)
        # print(shape)
        # the_probs = K.mean(x)
        # the_probs = K.print_tensor(the_probs, message="the probs now are: ")
        # the_probs = np.repeat(the_probs, x_len)

        # backup = x

        dolog = False

        if dolog:
            x = K.print_tensor(x, message="[0] the x now is: ")
        x = K.mean(x, axis=[0], keepdims=True)
        if dolog:
            x = K.print_tensor(x, message="[1] the x now is: ")
        x = K.repeat_elements(x, 1, axis=0)
        if dolog:
            x = K.print_tensor(x, message="[2] the x now is: ")
        x = K.reshape(x, shape=(-1, self.output_dim))

        # x = K.permute_dimensions(x, (1, self.output_dim))
        if dolog:
            x = K.print_tensor(x, message="[3] the x now is: ")

        # raise Exception
        # x = backup

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)




"""Averages the input batch probs, and output the same label for all
sample in batch

"""
def average_fn(x):
    # print(x.get_shape()[1])

    shape = x.get_shape()
    # try:
    #     print(K.get_value(K.mean(x)))
    # except: # Exception(x):
    #     print("get_value raised an exc")
    #     #print(x)

    # try:
    #     print(K.batch_get_value(x))
    # except: #Exception(x):
    #     print("batch_get_value raised an exc")
    #     #print(x)


    # raise Exception
    # x_length = len(x)
    # the_probs = sum(x) / x_length
    # the_probs = the_probs.reshape(1, -1)
    # the_probs = np.repeat(the_probs, x_length, axis=0)
    # return the_probs

    the_probs = K.mean(x)
    the_probs = K.print_tensor(the_probs, message="the probs now are: ")
    # the_probs = np.repeat(the_probs, x_len)

    backup = x

    x = K.print_tensor(x, message="the x now is: ")
    x = K.mean(x, axis=[1])
    # x = K.repeat_elements(x, shape[1], axis=0)
    x = K.reshape(x, shape=(self.input_shape, shape[1]))
    x = K.print_tensor(x, message="the x now is: ")

    x = backup

    return x
