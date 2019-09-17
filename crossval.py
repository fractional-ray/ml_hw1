"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
import numpy as np


#
# This function will yeild the train and test indices for the k-folds to make sure that it is the same size
#
def k_fold(n_splits, data, examples_per_fold,ideal_length,indices,n):
    count = 0
    for fold in range(n_splits):
        start, stop = count, count + examples_per_fold
        test_indices = indices[start:stop]
        # Creating an array of booleans of the size of the data so we can have a not to flip them all
        test = np.zeros(n,dtype=np.bool)
        test[test_indices] = True 
        train_index = indices[np.logical_not(test)]
        test_index = indices[test]
        yield train_index, test_index
        count = stop



def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """Perform cross validation with random splits.

    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    
    
    scores = np.zeros(folds)
    
    d, n = all_data.shape
    #
    # Creating indecies and getting the number of example per fold to pass the k_fold function
    #
    indices = np.arange(n)
    examples_per_fold = int(np.ceil(n / folds))
    ideal_length = int(examples_per_fold * folds)
    scores = ()
    models = ()

    #
    # Going through the different folds. I created a generator for k_folds so I can iterate over it.
    #
    for train_index, test_index in k_fold(folds, all_data, examples_per_fold,ideal_length,indices,n):
        x_train, x_test = all_data.T[train_index], all_data.T[test_index]
        y_train, y_test = all_labels[train_index], all_labels[test_index]
        model = trainer(x_train.T, y_train,params)
        predictions = predictor(x_test.T, model)
        scores = scores + (np.mean(predictions == y_test),)
        models = models + (models,)
    

    score = np.mean(scores)
    


    return score, models
