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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
import math

from MNL import *


'''
This module provides a number of auxiliary functions, in addition to the MNL model.

- One can find another loss function called MaxLogLikelihoodLoss. 

- There is a training function with early stopping capability.

- There are some functions to calculate the KPIs for model benchmarking.

'''


class MaxLogLikelihoodLoss(torch.autograd.Function):
    '''
       the negative of the log likelihood of the chosen alternative. 
       Note: this loss function ignores the loss of non-chosen alternatives,
         unlike the BinaryCrossEntropy loss which takes all losses into account.
    
       But while we maximize the log probability of the chosen alternative, 
         we are also minimizing the log probability of the non-chosen ones,
         since we do a softmax over the alternatives within a session.
    '''
    import torch

    def forward(self, input, target):
        # return the negative of the log likelihood of the chosen alternative
        likelihood = torch.dot(torch.t(input).view(-1), target.view(-1))

        # shift the value to the zone [1, 2] to avoid the underflowing
        likelihood = likelihood + 1
        
        # average over the number of samples
        n_samples = target.size()[0]
        return torch.neg(torch.log(likelihood) / n_samples)


def init_model(train_config):
    '''
        build and initialize the MNL model
    '''
    # use the full float type, float64
    torch.set_default_tensor_type('torch.DoubleTensor')
    
    MNL_features = train_config['MNL_features']
    optimizer_method = train_config['optimizer']
    learning_rate = train_config['learning_rate']
    momentum = train_config['momentum']
    weight_decay = train_config.get('weight_decay', 0)
    loss_func = train_config.get('loss', 'BinaryCrossEntropy')
    
    #model = build_model(n_features)
    model = MNL(MNL_features)
    
    # binary cross entropy
    if (loss_func == 'BinaryCrossEntropy'):
        # doc: http://pytorch.org/docs/master/nn.html
        # loss(o,t)=−1/n∑i(t[i]∗log(o[i])+(1−t[i])∗log(1−o[i]))
        loss = torch.nn.BCELoss()
    elif (loss_func == 'MaxLogLikelihood'):
        loss = MaxLogLikelihoodLoss()

    #loss = torch.nn.CrossEntropyLoss(size_average=True)
    
    if (optimizer_method == 'SGD'):
        # e.g. lr = 0.01/ 1e-2
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    
    elif (optimizer_method == 'Adam'):
        # weight_decay:  add L2 regularization to the weights ? 
        # It seems that with MNL any regularization would deteriarate the performance.
        # The Adam optimizer seems to converge faster than SGD
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    elif (optimizer_method == 'Adagrad'):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    elif (optimizer_method == 'RMSprop'):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    elif (optimizer_method == 'LBFGS'):
        # http://cs231n.github.io/neural-networks-3/#sgd
        # TODO: Even we eliminate the memory concerns, a large downside of naive application of L-BFGS is
        #   that it MUST be computed over the entire training set, which could contain millions of examples.
        #   Unlike mini-batch SGD, getting L-BFGS to work on mini-batches is more tricky and an active area
        #   of research.
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    
    return (model, loss, optimizer)


def train_one_epoch(epoch_index, module_tuple, df_session_groups, train_config):
    '''
    '''
    (model, loss, optimizer) = module_tuple
    
    gpu = train_config['gpu']
    verbose = train_config['verbose']
    l1_loss_weight = train_config['l1_loss_weight']
    l2_loss_weight = train_config['l2_loss_weight']
    MNL_features = train_config['MNL_features']
    save_gradients = train_config.get('save_gradients', False)

    total_cost = 0
    if (verbose >= 2):
        print('Num. sessions:', len(df_session_groups))
    
    for session_id in list(df_session_groups.groups.keys()):
    
        df_session = df_session_groups.get_group(session_id)
    
        if (verbose >= 2):
            print('-----------------------')
            print('session_id:', session_id)
            print('No. alternatives:', len(df_session))
    
        try:
            cost = model.train(loss, optimizer,
                     df_session[MNL_features].values,
                     df_session['choice'].values,
                     l1_loss_weight = l1_loss_weight,  # when zeor, no regularization
                     l2_loss_weight = l2_loss_weight,  # when zeor, no regularization
                     gpu=gpu)
        
        except ValueError:
            if (verbose >= 1):
                print('loss underflow in session: ', session_id)
            # skip this session
            continue

        total_cost += cost
    
        # save the gradients if asked
        if (save_gradients):
            new_gradients = get_session_gradients(epoch_index, session_id, model.parameters())
            train_config['session_gradients'].extend(new_gradients)
        
        if (verbose >= 2):
            print('train cost:', cost)
            predY = model.predict(df_session[MNL_features].values)
            print('Real Y-value:', df_session['choice'].values)
            print('Prediction:', predY)

    return total_cost


def train_with_early_stopping(model_tuple, train_data, train_config):
    '''
    '''
    wait = 0
    best_loss = 1e15

    loss_list = []
    
    verbose = train_config['verbose']
    epochs = train_config['epochs']
    patience = train_config['patience']
    early_stop_min_delta = train_config['early_stop_min_delta']
    save_gradients = train_config['save_gradients']
    
    if (save_gradients):
        # a variable that carries over epoches
        train_config['session_gradients'] = []
    
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(epoch, model_tuple, train_data, train_config)
        loss_list.append(epoch_loss)
        
        if (verbose >= 1):
            print('epoch:', epoch, ' loss:', epoch_loss, 'best_loss:', best_loss)

        if (epoch_loss - best_loss) < -early_stop_min_delta:
            # find the new minimal point, reset the clock
            best_loss = epoch_loss
            wait = 1
        else:
            if (wait >= patience):
                print('Early stopping!', ' epoch:', epoch,
                      'min_delta:', early_stop_min_delta, ' patience:', patience)
                break
            wait += 1

    print('Final epoch:', epoch, ' loss:', epoch_loss)
    
    return loss_list


def get_session_gradients(epoch_index, session_id, parameters):
    '''
        retrieve the gradient values from the Parameter of gradient
    '''
    res = []
    for param in parameters:
        if (param.is_cuda):
            gradients = param[0].cpu().data.numpy()
        else:
            gradients = param[0].data.numpy()
        
        res.append({
            'epoch_id': epoch_index,
            'session_id': session_id, 
            'mean_abs_gradients': np.mean(np.abs(gradients)),
            'std_abs_gradients': np.std(np.abs(gradients)),
            'gradients': gradients})
    
    return res


def get_default_MNL_features(df_data):
    '''
        Retrieve all features from the dataframe,
          excluding the auxliary features.
    '''
    # use all the applicable features in the data, excluding session specific features
    return sorted(set(df_data.columns.values) - 
                  set(['session_id', 'alter_id', 'choice']))


def run_training(df_training, train_config, model_tuple=None):
    '''
    '''
    MNL_features = train_config.get('MNL_features', [])
    
    if (len(MNL_features) == 0):
        # use all the applicable features in the data, excluding session specific features
        MNL_features = get_default_MNL_features(df_training)
        
        # set the config for the later use
        train_config['MNL_features'] = MNL_features
    
    n_features = len(MNL_features)
    print('Num features:', n_features)
    print('========================')
    print(train_config)
    print('========================')
    
    if (model_tuple is None):
        # Create a new model, other continue training on the existing model.
        (model, loss, optimizer) = init_model(train_config)
    
        if (train_config['gpu']):
            # run the model in GPU
            model = model.cuda()
            
            #hook = model.get_params().register_hook(lambda grad: print(grad))

        model_tuple = (model, loss, optimizer)
    else:
        print('Continue training...')
    
    # train with early stopping
    df_session_groups = df_training.groupby('session_id')
    
    loss_list = train_with_early_stopping(model_tuple, df_session_groups, train_config)

    return (model_tuple, loss_list)


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
    import pandas as pd
    return pd.concat(ret, axis=0)


def rank(pred_values, real_choice):
    '''
        Get the rank of chosen alternative within the predicted values
    '''
    # first, rank all the values
    rank = pd.DataFrame(pred_values).rank(axis=0, ascending=False)

    # filter out the rank of the chosen alternative, by dot product
    return rank[0].dot(real_choice)


def mean_rank(pred_values, real_choice):
    '''
        In a session with multiple choices,
         get the mean rank values for the chosen choices.
    '''
    return rank(pred_values, real_choice) / real_choice.sum()


def get_chosen_pred_value(pred_values, real_choice):
    return pd.Series(pred_values.reshape(-1)).dot(real_choice)


def mean_chosen_pred_value(pred_values, real_choice):
    '''
        In a session with multiple choices,
          get the mean probability value for the chosen ones.
    '''
    return get_chosen_pred_value(pred_values, real_choice) / real_choice.sum()


def validate(model, df_testing, train_config, features_to_skip = None):
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
    if (len(MNL_features) == 0):
        MNL_features = get_default_MNL_features(df_testing)
    
    session_size = []
    session_num_chosen_choices = [] # the number of chosen choices
    session_rank = []
    session_pred_value = []
    # the maximum probability that is assigned to an alternative within a session.
    session_max_prob = []
    
    for session_id in list(df_session_groups.groups.keys()):
    
        df_session = df_session_groups.get_group(session_id)
    
        if (features_to_skip == None):
            testing_data = df_session[MNL_features]
        else:
            # Set the values of feature-to-skip to be zero, 
            #   i.e. nullify the weights associated with the features to skip
            testing_data = df_session[MNL_features].copy()
            testing_data[features_to_skip] = 0
        
        predY = model.predict(testing_data.values, binary=False)
    
        #print('SessionId:', session_id)
        #print('AlterId:', df_session['alter_id'].values)
        #print('Real Y-value:', df_session['choice'].values)
        #print('Prediction:', predY)

        choice_value = df_session['choice'].values
        session_num_chosen_choices.append(choice_value.sum())
        session_size.append(len(df_session))
        #session_pred_value.append(get_chosen_pred_value(predY, choice_value))
        #session_rank.append(rank(predY, choice_value))
        session_pred_value.append(mean_chosen_pred_value(predY, choice_value))
        session_rank.append(mean_rank(predY, choice_value))
        session_max_prob.append(predY.max())
    
    df_session_KPIs = pd.DataFrame()
    df_session_KPIs['session_id'] = list(df_session_groups.groups.keys())
    df_session_KPIs['session_size'] = session_size
    df_session_KPIs['num_chosen_choices'] = session_num_chosen_choices
    df_session_KPIs['rank_of_chosen_one'] = session_rank
    df_session_KPIs['prob_of_chosen_one'] = session_pred_value
    df_session_KPIs['max_prob'] = session_max_prob
    
    return df_session_KPIs


def summarize_KPIs(df_session_KPIs, n_features):

    from scipy import stats

    KPI_summary = {}
    
    #stats.percentileofscore([1, 2, 3, 3, 4], 3, kind='weak')
    # expected 80.0
    KPI_summary['session_num'] = len(df_session_KPIs)
    KPI_summary['mean_session_size'] = df_session_KPIs['session_size'].mean()

    # calculate the percentile, LESS THAN and EQUAL to the given score
    KPI_summary['top_1_rank_quantile'] = \
          stats.percentileofscore(df_session_KPIs['rank_of_chosen_one'], 1, kind='weak')

    KPI_summary['top_5_rank_quantile'] = \
          stats.percentileofscore(df_session_KPIs['rank_of_chosen_one'], 5, kind='weak')

    KPI_summary['top_10_rank_quantile'] = \
          stats.percentileofscore(df_session_KPIs['rank_of_chosen_one'], 10, kind='weak')
        
    # The ratio between the rank of the chosen alternative and the number of alternatives
    rank_ratio = (df_session_KPIs['rank_of_chosen_one'] / df_session_KPIs['session_size'])
    KPI_summary['mean_rank_ratio'] = rank_ratio.mean()
    KPI_summary['median_rank_ratio'] = rank_ratio.median()
    
    KPI_summary['mean_rank'] = df_session_KPIs['rank_of_chosen_one'].mean()
    KPI_summary['median_rank'] = df_session_KPIs['rank_of_chosen_one'].median()
    
    KPI_summary['mean_probability'] = df_session_KPIs['prob_of_chosen_one'].mean()
    KPI_summary['median_probability'] = df_session_KPIs['prob_of_chosen_one'].median()

    # the difference of probability values between the chosen one and the predicted one
    prob_diff = (df_session_KPIs['prob_of_chosen_one'] - df_session_KPIs['max_prob'])
    KPI_summary['mean_probability_diff'] = prob_diff.mean()
    KPI_summary['median_probability_diff'] = prob_diff.median()
    
    # the log likelihood for each chosen alternative is negative. The higher the probability, 
    #  the closer the log likelihood is to the zero, (i.e. the lower the absolute value)
    KPI_summary['log_likelihood'] = np.log(df_session_KPIs['prob_of_chosen_one']).sum()
    
    KPI_summary['mean_log_likelihood'] = np.log(df_session_KPIs['prob_of_chosen_one']).mean()
    
    # AIC <- 2*length(model$coefficients) - 2*model$loglikelihood
    # Akaike Information Criterion, which estimates the quality of the model, (i.e. the lower, the better)
    '''
      AIC is founded on information theory: it offers an estimate of the relative information lost
        when a given model is used to represent the process that generated the data.
        (In doing so, it deals with the trade-off between the goodness of fit of the model and
         the simplicity of the model.)
    '''
    KPI_summary['AIC'] = 2 * n_features - 2 * KPI_summary['log_likelihood']
    
    return KPI_summary


def plot_loss(loss_list):
    '''
        plot the loss evolution
    '''
    import pandas as pd
    ax = pd.Series(loss_list, name='loss').plot()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    _ = ax.set_title('Loss Evolution during MNL training')
