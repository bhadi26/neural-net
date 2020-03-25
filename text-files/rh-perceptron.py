# rebecca hadi 
# module 5 assignment 
 
import numpy as np 

# variables
weights = [.24,.88]
inputs = [.8,.9]
output = .15
bias = 0
eta = 5.0


# activity function 
def calc_activity(inputs, weights, bias):
     activity = np.dot(inputs,weights) + bias
     return activity 

# activation function 
def calc_activation(activity): 
    x = activity
    activation = 1/(1+np.exp(-1*x))
    return activation

# calc delta
def set_delta_weights(activation, output):
    e = output - activation 
    delta = activation*(1-activation)*e
    return delta 

# use delta to update weights
def update_weights(inputs, output, weights, delta):
    weights[0] = weights[0] + (eta*delta*inputs[0])
    weights[1] = weights[1] + (eta*delta*inputs[1])


# loop 
for i in range(31):
    activity = calc_activity(inputs,weights,bias)
    activation = calc_activation(activity)
    delta = set_delta_weights(activation, output)
    update_weights(inputs, output, weights, delta)
    print(f'{i}th iteration weights are:\n w1 = {weights[0]}  and  \n w2 = {weights[1]} \n activation {activation}')


