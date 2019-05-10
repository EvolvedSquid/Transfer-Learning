"""
Transfer Learning with MNIST and Letters \n

Takes a convolutional model pretrained on the MNIST dataset, removes final dense and activation layer and creates
a new dense layer with 26 neurons.  Then the final layer is trained to recognize 28x28 handwritten letters, with only
THREE training examples per letter. \n

The model achieves approximately a 73% accuracy.  This number is expected to improve dramatically if more than 3 training
examples per letter is supplied. \n

Can be used with other pretrained neural networks to get high accuracy on unseen data, without a large amount of training data.
"""
from keras.datasets import fashion_mnist
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Conv2DTranspose, Lambda, Reshape
from keras.callbacks import LambdaCallback
from keras.backend import tf # as ktf
import keras.backend as K
import numpy as np
import random, os, math
import matplotlib.pyplot as plt
import numpy as np

import to_mnist

model = load_model('mnist_pred.hdf5')
model.summary()

print(model.layers[-1])
print(model.layers[-2])

# Create Transfer Learner from base model
def transfer_learner(base_model, new_output_shape, cut_off=1):
    for layer in base_model.layers:
        layer.trainable = False
    predictions = Dense(new_output_shape, activation='softmax', name='predictions')(base_model.layers[-cut_off - 1].output)
    model = Model(base_model.layers[0].input, predictions)
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])

    return model

model = transfer_learner(model, 26, 2)
model.summary()

# Create file pathes
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
t_alphabet = 'AAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLLMMMNNNOOOPPPQQQRRRSSSTTTUUUVVVWWWXXXYYYZZZ'

names = []
for letter in alphabet:
    for i in range(3):
        names.append('letter' + letter + str(i+1))

print(len(alphabet), len(t_alphabet), len(names))

# Define training data
x_train = to_mnist.convert_mass('Images/Raw/', '.png', names, (28, 28), True)
y_train = to_mnist.one_hot_classes([letter for letter in alphabet], [letter for letter in t_alphabet])

print(x_train); print(y_train)
print(type(x_train), type(y_train))
print(x_train.shape, y_train.shape)

# Fit model
model.fit(x_train, y_train,
            epochs=30,
            batch_size=None,
            steps_per_epoch=15)

# Evaluate model
for letter in alphabet:
    test = to_mnist.to_mnist(f'Images/Raw/letter{letter}t.png', True).reshape((1, 28, 28, 1))
    print(alphabet[model.predict(test).argmax()], np.amax(model.predict(test)), letter)
