import os
import numpy as np

def objective(X: np.ndarray, Y: np.ndarray, theta: np.ndarray):
    u = 1 / (1 + np.exp(X @ -theta)) # predicted probability
    u = np.clip(u, 1e-8, 1 - 1e-8)
    nll = - (Y * np.log(u) + (1 - Y) * np.log(1 - u)) # negative log likelihood (objective is to minimize this)
    return np.mean(nll) # NLL is low if the prediction is close to the true labels

def gradient(Xi: np.ndarray, Yi: np.ndarray, theta: np.ndarray):
    u = 1 / (1 + np.exp(-theta.T @ Xi)) # predicted probability
    grad = -(Yi - u) * Xi # gradient
    return grad

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

def train_and_val(X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, theta0: np.ndarray, num_epochs: int, lr: float):
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

    return {
        'best_theta': best_theta,         # best parameters
        'best_epoch': best_epoch,         # epoch that produces the best parameters
        'best_train_nll': best_train_nll, # best (lowest) average train nll
        'best_val_nll': best_val_nll,     # best (lowest) average val nll
        'train_error': train_error,       # train error of the model with the best parameters
        'val_error': val_error,           # val error of the model with the best parameters
        'theta_history': theta_history    # list of parameters produced at each epoch
    }

num_epochs = 1000
lr = 0.05

# Example: Student passing grades prediction
# Features: [bias, math_score, english_score]
# Label: 1 = pass, 0 = fail
# Rescale function for feature columns (excluding bias)
def rescale_matrix(X):
    X = np.array(X, dtype=float)
    X_scaled = X.copy()
    # Exclude bias (first column)
    for col in range(1, X.shape[1]):
        min_val = X[:, col].min()
        max_val = X[:, col].max()
        if max_val > min_val:
            X_scaled[:, col] = (X[:, col] - min_val) / (max_val - min_val)
        else:
            X_scaled[:, col] = 0.0
    return X_scaled

# You can now modify these arrays as you wish, then call rescale_matrix()
toy_X_train = np.array([
    [1.0, 40, 60],   # fail (low math)
    [1.0, 55, 45],   # fail (low math)
    [1.0, 65, 50],   # pass (math above 60)
    [1.0, 80, 30],   # pass (high math, low english)
    [1.0, 50, 80],   # fail (math not enough)
    [1.0, 90, 90],   # pass (high both)
])
toy_X_train = rescale_matrix(toy_X_train)
toy_Y_train = np.array([[0], [0], [1], [1], [0], [1]])
toy_X_val = np.array([
    [1.0, 60, 40],   # pass (math just enough)
    [1.0, 45, 85],   # fail (math too low)
])
toy_X_val = rescale_matrix(toy_X_val)
toy_Y_val = np.array([[1], [0]])
toy_X_test = np.array([
    [1.0, 70, 60],   # pass
    [1.0, 55, 90],   # fail
    [1.0, 85, 40],   # pass
    [1.0, 50, 50],   # fail
    [1.0, 75, 85],   # pass
])
toy_X_test = rescale_matrix(toy_X_test)
# Update experimental parameters for new feature size
num_features = toy_X_train.shape[1]
theta0 = np.array([[0.0], [0.0], [0.0]])

# training and validation
output = train_and_val(X_train=toy_X_train, 
                       Y_train=toy_Y_train, 
                       X_val=toy_X_val, 
                       Y_val=toy_Y_val, 
                       theta0=theta0, 
                       num_epochs=num_epochs, 
                       lr=lr)

predictions = predict(toy_X_test, output['best_theta'])
print(predictions)
print(output['best_theta'])