"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np
from collections import defaultdict
import pandas as pd 

def prob(data, alpha, d, n):
    total = d * n 
    probabilites = []
    for feature in data:
       counts = np.unique(feature, return_counts = True)
       true_counts = counts[1][1]
       probabilites.append((true_counts + alpha) / (total + 2 * alpha))
    return probabilites


def find_classes(labels):
    probabilites = defaultdict(list)
    for y in range(len(labels)):
        y_classes = labels[y]
        if y_classes in probabilites:
            probabilites[y_classes].append(y)
        else:
            probabilites[y_classes] = [y]
    return probabilites


def feature_probs(data, dict_labels, labels, alpha):
    fp = defaultdict(list)
    tmp_dict = defaultdict(list)
    tmp_arr = []

    #transpose data to access samples
    data = data.T
    for index in range(len(data)):
        tmp_dict[labels[index]].append(data[index])

    for index in range(len(tmp_dict)):
        fp[index] = ((sum(tmp_dict[index])) + alpha)/(len(tmp_dict[index]) + 2*alpha)

        #Calculate the probability of each label and append it to list of all label probabilities
        tmp_arr.append((len(tmp_dict[index]) + alpha)/(len(data)+len(tmp_dict)*alpha))

    #add probabilities of labels to new dictionary key
    length = len(fp)
    fp[length] = tmp_arr
    #dictionary index 20 should now hold the ordered label probabilities

    return fp

def conditional_prob(prob_A,prob_B):
    return (prob_A * prob_B)/(prob_A)

def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include an 'alpha' value
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    alpha = params['alpha']
    
    d, n = train_data.shape
    cond_prob = []
    cond_prob_not = []

    labels = np.unique(train_labels)
    num_classes = labels.size
    probs = prob(train_data, alpha, d, n)


    dict_labels = find_classes(train_labels)
    dict_probs = feature_probs(train_data, dict_labels, train_labels, alpha)
    
    model = {}
    model["conditions"] = cond_prob 
    model["priors"] = cond_prob_not


    return dict_probs


def bayes_prediction(sample, model):

    bayes_prob = []
    tmp_int = 0;
    #for all labels in the model (expect last inde which is label probailities)
    for key in range(len(model) - 1):
        #for all features in each label
        for index in range(len(model[key])):
            #take the log of each probabilty and add them up for total probability of the label
            if(sample[index] == True):
                tmp_int += model[key][index]
            else:
                tmp_int += 1-model[key][index]

        #add probability of label
        tmp_int += model[len(model)-1][key]
        #add total probaility for label to array and clear
        bayes_prob.append(tmp_int)
        tmp_int = 0

    return np.argmax(bayes_prob)



def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """

    prediction = []

    for index in range(len(data.T)):
        sample = data.T[index,:]
        prediction.append(bayes_prediction(sample,model))

    return prediction

