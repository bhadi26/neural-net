# rebecca hadi 
# module 5 assignment 

import pandas as pd 
import numpy as np 

# variables
weights = [.2,.7]
inputs = [.75206,.75206]
output = .95
bias = 0
eta = .1
layer = [.8,.5,.1,.2]



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
#for i in range(31):
activity = calc_activity(inputs,weights,bias)
activation = calc_activation(activity)
delta = set_delta_weights(activation, output)
update_weights(inputs, output, weights, delta)



