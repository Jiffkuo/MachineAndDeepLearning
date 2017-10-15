# Implement deep feed forward network
# activation f: ReLU
# output h: Softmax
# input layer: 10 nodes
# Hidden layer: 50 nodes
# output layer: 3 nodes
import numpy as np
import random as ra

# initialize weight with uniform distribution [0, 0.1]
def _init_weight(inputs):
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            inputs[i][j] = ra.uniform(0,0.1)

# initialize bias with uniform distribution [0, 0.1]
def _init_bias(inputs):
    for i in range(len(inputs)):
        inputs[i] = ra.uniform(0,0.1)

# activation function: ReLu
def _active_relu(input):
    return np.maximum(0, input)

# output function: softmax
def _output_softmax(input):
    e_input = np.exp(input - np.max(input))
    return e_input / e_input.sum()

# only input -wi/bi-> hidden and hidden -wo/bo-> output, one layer only
# note: weights needs to switch i, j to j, i for matrix product purpose
wi=np.empty(shape=(50,10))
wo=np.empty(shape=(3,50))
bi=np.empty(shape=(1,50))
bo=np.empty(shape=(1,3))

# initialization
_init_weight(wi)
_init_weight(wo)
_init_bias(bi)
_init_bias(bo)
x = np.array([0.5,0.6,0.1,0.25,0.33,0.9,0.88,0.76,0.69,0.96])

# forward pass: input - hidden layer
firstpass = x
for w, b in zip(wi, bi):
    firstpass = np.dot(w, firstpass) + b
    firstpass = _active_relu(firstpass)
#print len(firstpass)
#print firstpass

# forward pass: hidden layer - output
secondpass = firstpass
for w, b in zip(wo, bo):
    secondpass = np.dot(w, secondpass) + b
    secondpass = _output_softmax(secondpass)
print len(secondpass)
print secondpass