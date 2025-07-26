# NLP Sentiment Analysis Model

A machine learning project implementing logistic regression for binary sentiment analysis on movie reviews using Python and scikit-learn.

## 📋 Overview

This repository contains a complete sentiment analysis pipeline that classifies movie reviews as positive (1) or negative (0) using logistic regression. The model is trained from scratch using gradient descent with custom implementations of the objective function and gradient calculations.

## 🏗️ Project Structure

```
nlp-model/
├── functions.py              # Core ML functions and utilities
├── nlp_train.ipynb          # Main training notebook
├── data/
│   ├── debug/               # Small dataset for testing
│   │   ├── dict.txt        # Vocabulary mapping for debug set
│   │   └── reviews.tsv     # Sample reviews for debugging
│   └── full/               # Complete dataset
│       ├── dict.txt        # Full vocabulary mapping (39,190 words)
│       ├── train_data.tsv  # Training data (1,200 samples)
│       ├── valid_data.tsv  # Validation data
│       └── test_data.tsv   # Test data
├── figures/
│   └── nlls.png           # Training/validation loss visualization
└── __pycache__/           # Python cache files
```

## 🔧 Core Functions (`functions.py`)

### Mathematical Functions

#### `objective(X, Y, theta)`
Computes the negative log-likelihood (NLL) loss function for logistic regression:
- **Input**: Feature matrix X, labels Y, parameters theta
- **Output**: Mean negative log-likelihood
- Uses sigmoid function with numerical stability (clipping to prevent log(0))
- This is the cost function we minimize during training

#### `gradient(Xi, Yi, theta)`
Calculates the gradient of the loss function for a single sample:
- **Input**: Single feature vector Xi, label Yi, parameters theta
- **Output**: Gradient vector for parameter updates
- Used in stochastic gradient descent for weight updates

### Training Functions

#### `train(X_train, Y_train, theta0, num_epochs, lr)`
Implements stochastic gradient descent training:
- **Parameters**:
  - `X_train`: Training feature matrix
  - `Y_train`: Training labels
  - `theta0`: Initial parameters (usually zeros)
  - `num_epochs`: Number of training epochs
  - `lr`: Learning rate
- **Returns**: History of parameter values after each epoch
- Processes one sample at a time for parameter updates

#### `train_and_val(X_train, Y_train, X_val, Y_val, num_epochs, lr, visualize_nlls=True)`
Complete training pipeline with validation:
- Trains the model and tracks performance on both training and validation sets
- Implements early stopping based on validation performance
- **Returns**: Dictionary containing:
  - `best_theta`: Optimal parameters
  - `best_epoch`: Epoch with best validation performance
  - `best_train_nll` & `best_val_nll`: Best loss values
  - `train_error` & `val_error`: Error rates
  - `theta_history`: Complete parameter evolution

### Prediction and Evaluation

#### `predict(X, theta)`
Makes binary predictions using trained parameters:
- Applies sigmoid function to get probabilities
- Uses 0.5 threshold for binary classification
- **Returns**: List of binary predictions (0 or 1)

#### `error_rate(y, y_hat)`
Calculates classification error rate:
- **Input**: True labels y, predicted labels y_hat
- **Output**: Proportion of misclassified samples

### Visualization

#### `plot_nlls(num_epochs, train_nlls, val_nlls, fig_size=(6.4, 4.8))`
Creates training visualization:
- Plots training and validation negative log-likelihood vs epochs
- Saves plot as `./figures/nlls.png`
- Helps identify overfitting and convergence patterns

## 📊 Jupyter Notebook (`nlp_train.ipynb`)

The notebook implements a complete machine learning workflow:

### 1. Data Loading and Preprocessing
```python
# Load datasets
train_data = pd.read_csv('./data/full/train_data.tsv', sep='\t', header=None)
val_data = pd.read_csv('./data/full/valid_data.tsv', sep='\t', header=None)
test_data = pd.read_csv('./data/full/test_data.tsv', sep='\t', header=None)

# Feature extraction using bag-of-words
cvect = CountVectorizer(binary=True, max_features=10000)
X_train = cvect.fit_transform(train_data.iloc[:, 1])  # Text reviews
Y_train = train_data.iloc[:, 0].values.reshape(-1, 1)  # Labels (0/1)
```

### 2. Model Training
```python
# Training parameters
num_epochs = 1000
lr = 0.005

# Train model with validation
output = fn.train_and_val(X_train, Y_train, X_val, Y_val, 
                         num_epochs, lr, visualize_nlls=True)
```

### 3. Model Evaluation
The notebook evaluates model performance on multiple metrics:
- **Training Error Rate**: Performance on training data
- **Validation Error Rate**: Performance on validation data (for model selection)
- **Test Error Rate**: Final performance evaluation
- **Negative Log-Likelihood**: Loss function values

### 4. Prediction Examples
Demonstrates model predictions on individual reviews:
```python
# Example prediction on first test sample
review_prediction = fn.predict(X_test[0], output['best_theta'])
print(f"True sentiment: {Y_test[0]}")
print(f"Predicted sentiment: {review_prediction}")
```

## 📈 Dataset Details

### Data Format
- **TSV files**: Tab-separated values with two columns
  - Column 1: Label (0 = negative, 1 = positive)
  - Column 2: Review text
- **Dictionary files**: Word-to-index mappings for vocabulary

### Dataset Sizes
- **Training**: 1,200 movie reviews
- **Validation**: Used for hyperparameter tuning and early stopping
- **Test**: Final evaluation set
- **Vocabulary**: 39,190 unique words in full dataset

### Preprocessing
- **Binary bag-of-words**: Each word's presence/absence (not frequency)
- **Max features**: Limited to 10,000 most frequent words
- **Bias term**: Added automatically during training

## 🚀 Usage

1. **Environment Setup**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

2. **Run Training**:
   Open `nlp_train.ipynb` in Jupyter and execute cells sequentially

3. **Custom Training**:
   ```python
   import functions as fn
   
   # Train with custom parameters
   output = fn.train_and_val(X_train, Y_train, X_val, Y_val, 
                            num_epochs=500, lr=0.01)
   
   # Make predictions
   predictions = fn.predict(X_test, output['best_theta'])
   ```

## 📊 Model Performance

Based on the notebook output:
- **Best Epoch**: 11 (early stopping prevents overfitting)
- **Training NLL**: 0.0664
- **Validation NLL**: 0.3216
- **Training Error**: 0.00% (perfect fit on training data)
- **Validation Error**: 13.50%
- **Test Error**: 14.25%
- **Training Time**: ~113 seconds

The model shows signs of overfitting (perfect training accuracy vs 86.75% validation accuracy), which is common in text classification with limited data.

## 🔍 Key Features

- **From-scratch Implementation**: Custom logistic regression without external ML libraries
- **Stochastic Gradient Descent**: Sample-by-sample parameter updates
- **Early Stopping**: Prevents overfitting using validation performance
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Modular Design**: Reusable functions for different datasets
- **Educational Focus**: Clear separation of concerns for learning purposes

## 🎯 Learning Objectives

This project demonstrates:
- Binary classification with logistic regression
- Gradient descent optimization
- Text preprocessing and feature extraction
- Model evaluation and validation techniques
- Overfitting detection and mitigation
- Scientific computing with NumPy and pandas

## 📋 Requirements

- Python 3.x
- NumPy
- pandas
- scikit-learn
- matplotlib
- Jupyter Notebook

## 📝 Notes

- The model uses binary bag-of-words features (word presence, not frequency)
- Numerical stability is ensured through probability clipping in the objective function
- The visualization helps identify convergence and overfitting patterns
- This is an educational implementation; production systems would use optimized libraries like scikit-learn
