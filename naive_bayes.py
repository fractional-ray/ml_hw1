"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np

def prob(data, alpha, d, n):
    total = d * n 
    probabilites = []
    for feature in data:
       counts = np.unique(feature, return_counts = True)
       true_counts = counts[1][1]
       probabilites.append((true_counts + alpha) / (total + 2 * alpha))
    #obabilites = (count[0] + alpha) / (total + 2 * alpha)
    return probabilites


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
    
    cond_prob = []
    cond_prob_not = []

    labels = np.unique(train_labels)
    #probs = prob(train_labels,alpha)
    d, n = train_data.shape
    num_classes = labels.size
    probs = prob(train_data, alpha, d, n)
    for x in range(d):
        y = (x + 1) % num_classes
        cond_prob.append(conditional_prob(probs[x],probs[y]))
        cond_prob_not.append(conditional_prob(1 - probs[x] , 1 - probs[y]))
    #    cond_prob[1] = conditional_prob(probs[x],probs[index])

    # TODO: INSERT YOUR CODE HERE TO LEARN THE PARAMETERS FOR NAIVE BAYES
    
    model = {}
    model["conditions"] = cond_prob 
    model["priors"] = cond_prob_not 

    return model


def bayes_prediction(sample, model):
    prior = model["priors"]
    conditions = model["conditions"]
    total = 1
    bayes_probility = []
    for index in range(20):
        word = sample[index]
        if word == False:
            value = np.log(prior[index])
        else:
            value = np.log(conditions[index])
        
        total += total *  value

        bayes_probility.append(total)
        

    return np.argmax(bayes_probility)



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

    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA
