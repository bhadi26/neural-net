'''
Node Layer Class 
Naming conventions inspired by Office Hours + Coding suggestion handout 

Rebecca Hadi
''' 

import numpy as np 

# perceptron object (ended up not using in layer class)
class Perceptron: 
    def __init__(self,inputs,weights,bias): 
        self.inputs = inputs 
        self.weights = weights 
        self.bias = bias     
         
    def calc_output(self): 
        x = np.dot(inputs,weights) + bias
        activation = 1/(1+np.exp(-1*x))
        return activation

        

class NodeLayer: 
    def __init__(self,OutputLayerFlag,layer,layer_length,weights,bias):
        self.OutputLayerFlag = OutputLayerFlag
        self.layer = layer 
        self.layer_length = layer_length 
        self.weights = weights  # for bias
        self.output_vector = [0] * layer_length
        self.littleE_vector = [0] * layer_length
        self.delta = [0] * layer_length
        self.delta_weights = [0] * len(weights)
        self.delta_bias = [0] * layer_length 
        self.bias_vector = [bias] * layer_length
        self.bias = bias
    
    def get_error_vector(self,desiredOut): 
        if self.OutputLayerFlag == True: 
            self.littleE_vector = desiredOut - self.output_vector
        return self.littleE_vector 
 
    def get_layer_output_vector(self,inputs): 
        for i in range(self.layer_length): 
            w = np.array_split(self.weights,self.layer_length)[i] 
            x = np.dot(inputs,w) + self.bias
            activation = 1/(1+np.exp(-1*x))
            self.output_vector[i] = activation
        return self.output_vector 
    
    def set_output_layer_delta_values(self,desiredOut):
        for i in range(self.layer_length):
            self.delta[i] = self.littleE_vector * [1 - self.output_vector[i]] * [self.output_vector[i]]

    def set_hidden_layer_weighted_delta_values(self,outputlayer):
        for i in range(self.layer_length): 
            for k in range(outputlayer.layer_length):
                self.delta[i] = (1 - self.output_vector[i])*self.output_vector[i]*np.sum(outputlayer.delta[k]*outputlayer.weights[k])

    def calc_delta_weights(self, eta, input):
        a = 0 
        for i in range(0,self.layer):
            for j in range(0,self.layer_length):
                self.delta_weights[a] = self.weights[i] + eta*(self.delta[j]*input[i])
                a = a+1
    
    def calc_delta_bias(self, eta):
         for j in range(0,self.layer_length):
              self.delta_bias[j] = self.bias_vector[j] + eta*(self.delta[j]*1)


    def update_layer_weights(self): 
        for i in range(len(self.weights)): 
            self.weights[i] = self.delta_weights[i]

    def update_layer_bias(self): 
        for j in range(0,self.layer_length):
            self.bias_vector[j] = self.delta_bias[j]