import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import math
# serialization and deserialization of model
import pickle

import numpy as np
import pandas as pd

'''

###################################################################################

The MIT License (MIT)

Copyright (c) 2018 Lisong Guo <lisong.guo@me.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

###################################################################################
'''

class MNL(nn.Module):
    '''
        The Multinomial Logistic Regression model implemented with Pytorch
    '''
    def __init__(self, feature_list):
        super(MNL, self).__init__()
        
        self.feature_list = feature_list
        input_dim = len(feature_list)
        # a linear layer without bias
        self.linear = torch.nn.Linear(input_dim, 1, bias=False)
        self.softmax = torch.nn.Softmax()
    
    
    def forward(self, x):
        # expect the input to be a session of alternatives
        util_values = self.linear(x)
        
        #!! a trik to prevent the underflow (divide by zero) in the softmax later
        util_values = util_values + 2
        
        # transpose the result vector before the softmax 
        util_values = torch.t(util_values)
        
        # convert the softmax values to binary values
        #max_values, indices = self.softmax(util_values).max()
    
        #results = np.zeros(len(x))
        #results[indices] = 1
        #results = np.transpose(results)
        
        return torch.t(self.softmax(util_values))

    
    def l1_loss(self, l1_weight=0.01):
        '''
            Return: L1 regularization on all the parameters
        '''
        params_list = []
        for param in self.parameters():
            params_list.append(param.view(-1))
        torch_params = torch.cat(params_list)
            
        return l1_weight * (torch.abs(torch_params).sum())


    def l2_loss(self, l2_weight=0.01):
        '''
            Return: L2 regularization on all the parameters
        '''
        params_list = []    
        for param in self.parameters():
            params_list.append(param.view(-1))
        torch_params = torch.cat(params_list)
            
        return l2_weight * (torch.sqrt(torch.pow(torch_params, 2).sum()))


    def train(self, loss, optimizer, x_val, y_val,
              l1_loss_weight = 0,  # when zero, no L1 regularization
              l2_loss_weight = 0,
              gpu=False):
        """
            Train the model with a batch (in our case, also a session) of data
        """
        # expect y_val to be of one_dimension
        y_val = y_val.reshape(len(y_val), 1)

        tensorX = torch.from_numpy(x_val).double()
        tensorY = torch.from_numpy(y_val).double()

        if (gpu):
            dtype = torch.cuda.DoubleTensor
        else:
            dtype = torch.DoubleTensor

        # input variable
        x = Variable(tensorX.type(dtype), requires_grad=False)
        # target variable
        y = Variable(tensorY.type(dtype), requires_grad=False)

        # Forward to calculate the losses
        fx = self.forward(x)
        data_loss = loss.forward(fx, y)

        # optional: add L1 or L2 penalities for regularization
        if (l1_loss_weight != 0):
            l1_loss = self.l1_loss(l1_loss_weight)
            output = data_loss + l1_loss

        elif (l2_loss_weight != 0):
            l2_loss = self.l2_loss(l2_loss_weight)
            output = data_loss + l2_loss

        else:
            output = data_loss

        # Underflow in the loss calculation
        if math.isnan(output.data[0]):
            raise ValueError('NaN detected')
            #return output.data[0]

        if (isinstance(optimizer, torch.optim.LBFGS)):
            def closure():
                optimizer.zero_grad()
                output.backward(retain_graph=True)
                return output

            optimizer.step(closure)
        else:
            # Reset gradient
            optimizer.zero_grad()
            # Backward
            output.backward()
            # Update parameters
            optimizer.step()

        # return the cost
        return output.data[0]


    def predict(self, x_val, binary=False):
        '''
            Give prediction for alternatives within a single session
            x_val: DataFrame, or np.ndarray
            return: numpy
        '''
        is_gpu = self.get_params()[0].is_cuda

        if isinstance(x_val, pd.DataFrame):
            tensorX = torch.from_numpy(x_val.values).double()
        else:
            tensorX = torch.from_numpy(x_val).double()
    
        if (is_gpu):
            x = Variable(tensorX.type(torch.cuda.DoubleTensor), requires_grad=False)
        else:
            x = Variable(tensorX, requires_grad=False)

        output = self.forward(x)
    
        if (is_gpu):
            # get the data from the memory of GPU into CPU
            prob = output.cpu().data.numpy()
        else:
            prob = output.data.numpy()
        
        if (binary):
            # convert the softmax values to binary values
            max_indice = prob.argmax(axis=0)
            ret = np.zeros(len(prob))
            ret[max_indice] = 1
            return ret
        else:
            return prob


    def get_params(self):
        '''
            Return the Variable of the MNL parameters,
              which can be updated manually.
        '''
        for name, param in self.named_parameters():
            if param.requires_grad and name == 'linear.weight':
                return param
        return None

    
    def print_params(self):
        '''
            Print all the parameters within the model
        '''
        params = self.get_params()[0]
        
        if (params.is_cuda):
            values = params.cpu().data.numpy()
        else:
            values = params.data.numpy()
        
        for index, feature in enumerate(self.feature_list):
            print(feature, ':', values[index])
    
    
    def get_feature_weight(self, feature_name):
        ''' Retrieve the weight of the desired feature '''
        params = self.get_params()[0]

        if (params.is_cuda):
            param_values = params.cpu().data.numpy()
        else:
            param_values = params.data.numpy()
        
        for index, feature in enumerate(self.feature_list):
            if (feature_name == feature):
                return param_values[index]
        
        # did not find the specified feature
        return None
    
    
    def get_feature_weights(self):
        ''' get the dictionary of feature weights '''
        params = self.get_params()[0]

        if (params.is_cuda):
            param_values = params.cpu().data.numpy()
        else:
            param_values = params.data.numpy()
        
        feature_weights = {}
        
        for index, feature in enumerate(self.feature_list):
            feature_weights[feature] = param_values[index]
        
        return feature_weights
    
    
    def set_feature_weight(self, feature_name, value):
        ''' Reset the specified feature weight
        '''
        params = self.get_params()[0]
        is_found = False
        
        try:
            for index, feature in enumerate(self.feature_list):
                if (feature_name == feature):
                    is_found = True
                    # override the parameters within the model
                    params[index] = value
        except RuntimeError as e:
            #print('RuntimeError: ', e)
            #print('One can ignore this error, since the parameters are still updated !')
            pass
        
        return is_found


    def save(self, file_name):
        '''
            Serialize the model object into a file
        '''
        with open(file_name, mode='wb') as model_file:
            pickle.dump(self, model_file)
            print('save model to ', file_name)


    def set_train_config(self, train_config):
        '''
            Set the training configs along with the model,
             so that it can be serialized together with the model.
        '''
        self.train_config = train_config


    def get_train_config(self):
        return self.train_config


def load_model(file_name):
    '''
        Load a model from a pickled file
    '''
    with open(file_name, mode='rb') as model_file:
        model = pickle.load(model_file)
        print('load model from ', file_name)
        return model


def build_model(input_dim):
    '''
        Another way to build the model.
    '''
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, 1, bias=False))
    
    # We need the softmax layer here for the binary cross entropy later 
    model.add_module('softmax', torch.nn.Softmax())
    
    return model


