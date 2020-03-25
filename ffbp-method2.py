'''
HW Assignment #7 
Rebecca Hadi 
''' 

import numpy as np 
import NodeLayerClass as nl 


# input output paids 
input = np.array([[1.0,1.0],[-1,-1]], float)
desiredOut = np.array([[.9],[0.05]], float)
eta = 1.0 

weights_hidden = [0.3, 0.3, 0.3, 0.3]
weights_output = [0.8, 0.8]
bias = 0.0 

# initialize network 
hiddenlayer =  nl.NodeLayer(False,2,2,weights_hidden, bias)
outputlayer = nl.NodeLayer(True,2,1,weights_output, bias)


# method 2 
for pair in range(0,2):
    for iter in range(0,15): 
        # feed forward 
        hidden_out = hiddenlayer.get_layer_output_vector(input[pair,:])
        output_out = outputlayer.get_layer_output_vector(hidden_out) 
        Error = outputlayer.get_error_vector(desiredOut[pair,:])
        BigE = 0.5*Error**2 
    
        # backpropagation 
        outputlayer.set_output_layer_delta_values(desiredOut[pair,:])
        hiddenlayer.set_hidden_layer_weighted_delta_values(outputlayer)
    
        # calculate weights (don't update yet)
        outputlayer.calc_delta_weights(eta, hidden_out)
        hiddenlayer.calc_delta_weights(eta, input[pair,:])
        outputlayer.calc_delta_bias(eta)
        hiddenlayer.calc_delta_bias(eta)
    
        # update all the weights at the same time 
        outputlayer.update_layer_weights()
        hiddenlayer.update_layer_weights()
        outputlayer.update_layer_bias()
        hiddenlayer.update_layer_bias()
    
# training complete  
# run once more to get output 
hidden_out = hiddenlayer.get_layer_output_vector(input[1,:])
output_out = outputlayer.get_layer_output_vector(hidden_out)
Error = outputlayer.get_error_vector(desiredOut[1,:])
BigE = 0.5*Error**2
print(f'BigE is {BigE}')
print(f'Output Value is: {output_out}')