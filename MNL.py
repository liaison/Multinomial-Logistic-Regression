import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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
        '''
        is_gpu = self.get_params()[0].is_cuda

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
            max_indice = prob.argmax(axis=1)
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
        for name, param in model.named_parameters():
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




def test_model(model, df_testing, train_config, features_to_skip = None):
    '''
        Test the model with the given data
        train_config: some parameters are used, i.e. MNL_feature, gpu
        features_to_skip:  a list of features to skip in the validation
        return: the statistic results of testing
    '''
    df_session_groups = df_testing.groupby('session_id')

    if (train_config['verbose']):
        print('Num of testing sessions:', len(df_session_groups))

    MNL_features = train_config['MNL_features']

    # the testing data with the prediction value for each alternative
    ret = []
    
    session_list = list(df_session_groups.groups.keys())
    
    # shuffle the sample list in each epoch
    # Important for the "stochastic" probability of the gradient descent algorithm ?!
    if (train_config.get('shuffle_batch', True)):
        import random
        random.shuffle(session_list)

    for session_id in session_list:
    
        # create a copy of the testing data
        df_session = df_session_groups.get_group(session_id).copy()
    
        if (features_to_skip == None):
            testing_data = df_session[MNL_features]
        else:
            # Set the values of feature-to-skip to be zero, 
            #   i.e. nullify the weights associated with the features to skip
            testing_data = df_session[MNL_features].copy()
            testing_data[features_to_skip] = 0
        
        # predict a single session
        predY = model.predict(testing_data.values, binary=False)
    
        # add the prediction column
        df_session['pred_value'] = predY
        
        ret.append(df_session)
        
    # concatenate the dataframes along the rows
    return pd.concat(ret, axis=0)


