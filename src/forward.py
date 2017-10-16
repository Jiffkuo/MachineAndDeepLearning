# Implement deep feed forward network
# activation f: ReLU
# output h: Softmax
# input layer: 10 nodes
# Hidden layer: 50 nodes
# output layer: 3 nodes
import numpy as np

# initialize weight with uniform distribution [0, 0.1]
def _init_weight(inputs):
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            inputs[i][j] = np.random.uniform(0,0.1)

# initialize bias with uniform distribution [0, 0.1]
def _init_bias(inputs):
    for i in range(len(inputs)):
        inputs[i] = np.random.uniform(0,0.1)

# activation function: ReLu
def _active_relu(input, derivative=False):
    if not derivative:
        return np.maximum(0, input)
    else:
        if input <= 0:
            return 0
        else:
            return 1

# output function: softmax
def _output_softmax(input):
    #scoreMatexp = np.exp(np.asarray(input))
    #return scoreMatexp / scoreMatexp.sum(0)
    return np.exp(input) / np.exp(input).sum(axis=0)

# only input -[wi/bi]-> hidden and hidden -[wo/bo]-> output, one layer only
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
print "input X =",x

outputs = []
# forward pass: input - hidden layer
firstpass = x
for w, b in zip(wi, bi):
    firstpass = np.dot(w, firstpass) + b
    outputs.append(firstpass)
    firstpass = _active_relu(firstpass)
#print len(firstpass)
#print firstpass

# forward pass: hidden layer - output
secondpass = firstpass
for w, b in zip(wo, bo):
    secondpass = np.dot(w, secondpass) + b
    secondpass = _output_softmax(secondpass)
    outputs.append(secondpass)
print "output Y =",secondpass

# calculate sample loss function
T = np.array([1, 0, 0])
L = 0
for t, y in zip(T, secondpass):
    L = L + t * np.log10(y)
print L

# backward pass: calculate delta
deltas = []
# output layer delta
delta_output = np.array(outputs[1]) - T
print "Output layer delta ="
for delta in delta_output.tolist():
    print delta
    deltas.append(delta)
# hidden layer delta
print "Hidden layer delta ="
for i in range(len(wo.T)):
    delta = np.multiply(
        np.dot(wo.T[i], delta_output.T),
        _active_relu(outputs[0][i], derivative=True)
    )
    print delta
    deltas.append(delta)
