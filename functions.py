import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt

def objective(X: np.ndarray, Y: np.ndarray, theta: np.ndarray):
    u = 1 / (1 + np.exp(X @ -theta)) # predicted probability
    u = np.clip(u, 1e-8, 1 - 1e-8)
    nll = - (Y * np.log(u) + (1 - Y) * np.log(1 - u)) # negative log likelihood (objective is to minimize this)
    return np.mean(nll) # NLL is low if the prediction is close to the true labels

def gradient(Xi: np.ndarray, Yi: np.ndarray, theta: np.ndarray):
    u = 1 / (1 + np.exp(-theta.T @ Xi)) # predicted probability
    grad = -(Yi - u) * Xi # gradient
    return grad

def plot_nlls(num_epochs: list[int], train_nlls: list[float], val_nlls: list[float], fig_size=(6.4, 4.8)):
    
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    # create a new figure, to avoid duplicating with figure in previous plots
    plt.figure(figsize=fig_size) 

    # add title and labels and title
    plt.title('Average Negative Log Likelihood vs. Number of Epochs')
    plt.xlabel('Number of epochs')
    # plt.xticks(num_epochs)
    plt.ylabel('Average negative log likelihood')
    
    # add train and error plots here: students write the below 2 lines
    plt.plot(num_epochs, train_nlls, label='training')
    plt.plot(num_epochs, val_nlls, label='validation')
    
    # show legends and save figure
    plt.legend() # show legend
    plt.savefig('./figures/nlls.png') # save figure for comparison

def train(X_train: np.ndarray, Y_train: np.ndarray, theta0: np.ndarray, num_epochs: int, lr: float):
    numSamples, numFeatures = X_train.shape
    theta_history = [theta0]
    theta = theta0
    for epoch in range(num_epochs): 
        for i in range(numSamples):
            X_i = X_train[i].reshape((numFeatures, 1)) # convert from shape (M+1,) to shape (M+1, 1)
            Y_i = Y_train[i].reshape((1, 1))            # convert from shape (1,) to shape (1, 1)
            theta = theta - lr * gradient(X_i, Y_i, theta)
        theta_history.append(theta)
    return theta_history

def predict(X: np.ndarray, theta: np.ndarray):
    numSamples, numFeatures = X.shape
    predictions = []
    for i in range(numSamples):
        u = 1/(1 + np.exp(-theta.T @ X[i].reshape((numFeatures, 1)))) # sigmoid function
        Y_i = 1 if u >= 0.5 else 0 # if p >= 0.5 the prediction is a 1
        predictions.append([Y_i])
    return predictions

def error_rate(y: np.ndarray, y_hat: np.ndarray):
    N = y.shape[0]
    misclassified = 0
    for (y_i, y_hat_i) in zip(y, y_hat):
        if y_i != y_hat_i: misclassified += 1
    return misclassified / N

def train_evaluate_model(X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, num_epochs: int, lr: float, visualize_nlls: bool = True):
    theta0 = np.zeros((X_train.shape[1], 1))  # initialize theta to zeros

    # train the model
    theta_history = train(X_train, Y_train, theta0, num_epochs, lr)

    # get the training and validation negative log likelihoods
    train_nlls, val_nlls = [], []
    best_train_nll, best_val_nll, best_theta, best_epoch = None, None, None, None

    for epoch, theta in enumerate(theta_history): 
        # compute nlls
        train_nll = objective(X_train, Y_train, theta)
        val_nll = objective(X_val, Y_val, theta)
        train_nlls.append(train_nll)
        val_nlls.append(val_nll)
        # get best parameters
        if best_val_nll is None or val_nll < best_val_nll:
            best_val_nll = val_nll
            best_train_nll = train_nll
            best_epoch = epoch
            best_theta = theta

    # Get error rate
    Y_train_hat = predict(X_train, best_theta)
    Y_val_hat = predict(X_val, best_theta)
    train_error = error_rate(Y_train, Y_train_hat)
    val_error = error_rate(Y_val, Y_val_hat)

    if visualize_nlls:
        num_epochs = [i for i in range(num_epochs + 1)]
        plot_nlls(num_epochs, train_nlls, val_nlls)

    return {
        'best_theta': best_theta,         # best parameters
        'best_epoch': best_epoch,         # epoch that produces the best parameters
        'best_train_nll': best_train_nll, # best (lowest) average train nll
        'best_val_nll': best_val_nll,     # best (lowest) average val nll
        'train_error': train_error,       # train error of the model with the best parameters
        'val_error': val_error,           # val error of the model with the best parameters
        'theta_history': theta_history    # list of parameters produced at each epoch
    }
