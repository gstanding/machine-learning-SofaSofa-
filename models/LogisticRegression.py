# author: viaeou

#sys.path
import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import utils.lr_utils as lr_utils



train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = lr_utils.load_dataset()


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    :param X_train: training set of shape(num_px * num_px * 3, m_train)
    :param Y_train: training labels of shape(1,m_train)
    :param X_test: validation set of shape(num_px * num_px * 3, m_test)
    :param Y_test: validation_set of shape(1, m_test)
    :param num_iterations: hyperparameter representing the number of iterations to optimize the parameters
    :param learning_rate: hyperparameter representing learning rate
    :param print_cost: set true to print cost every 100 iterations
    :return: d: dictionary contain the information of the model
    """
    """
    steps:
    1. intialize the parameters
    2. optimize the model--it contains the loop of forward propagation to calculate the cost function and back propagation to calculate the gradient to the parameters and upgrade the parametsrs 
    3. make the prediction
    """
    # 1. initialize the parameters
    dim = X_train.shape[0]
    w, b = initialize(dim)

    # 2. optimize the model
    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations=num_iterations, learning_rate=learning_rate,
                                 print_cost=True)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters['w']
    b = parameters['b']

    # 3. Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False):
    # the lists contains the values of cost function after every 100 iterations
    costs = []
    for i in range(num_iterations):
        grads, cost = propagation(w, b, X_train, Y_train)  # 1. forward propagation and backforward propagation

        # retrive the gradients
        dw = grads['dw']
        db = grads['db']

        # 2 update the parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # record the costs
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {
        'w': w,
        'b': b
    }

    return params, costs


def propagation(w, b, X, Y):
    m = X.shape[1]

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    cost = -1 / m * np.sum(np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), (1 - Y).T))

    dZ = A - Y
    dw = 1 / m * np.dot(X, dZ.T)
    db = 1 / m * np.sum(dZ)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {
        'dw': dw,
        'db': db
    }

    return grads, cost


def sigmoid(z):
    a = np.divide(1, (1 + np.exp(-z)))

    return a


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
train_set_y = train_set_y_orig
test_set_y = test_set_y_orig


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost= True)
