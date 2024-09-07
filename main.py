import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from random import randint

#warning, very volatile to change performance
DAYS_TRACKING = 10

def z_score_normalize(data):
    sum = 0
    data = data.astype(np.float64)
    for i in range(0, data.shape[1]):
        sum += data[0][i]
    mean = sum / data.shape[1]
    newData = (data - mean) / mean
    return newData, mean

def unnormalize_data(normalized_data, mean):
    original_data = (normalized_data * mean) + mean
    return original_data
file_path = "portfolio_data.csv" 
df = pd.read_csv(file_path)
data = np.array(df)
dates = pd.to_datetime(df['Date'])
values = df['AMZN']
train_dataX = data[:, :2]
train_dataX = train_dataX[:, 1:]
train_dataX = train_dataX.T
test_dataX = train_dataX[:, :400]
train_dataX = train_dataX[:, 400:]
train_dataX, meanTr = z_score_normalize(train_dataX)
train_dataX = train_dataX.astype(np.float64)
test_dataX, meanTe = z_score_normalize(test_dataX)
test_dataX = test_dataX.astype(np.float64)


def init_params():
    W1 = random.uniform(-0.01, 0.01) 
    b1 = random.uniform(-0.01, 0.01) 
    W2 = random.uniform(-0.01, 0.01) 
    b2 = random.uniform(-0.01, 0.01) 
    W3 = random.uniform(-0.01, 0.01) 
    return W1, b1, W2, b2, W3

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def compute_cost(val, Y): #MSE
    cost = np.sum((val - Y) ** 2)
    return cost

def forward_prop(W1, b1, W2, b2, W3, X, time, sum, step, max_step, Aarray, inputArray):
    if time == 0:
        Aarray = []  # Store activations
        inputArray = []  # Store inputs
        sum = 0
    if step <= max_step: 
        Z1 = W1 * X[0][step] + b1 + sum
        sum = 0
        A1 = ReLU(Z1)
        sum += W3 * A1
        Aarray.append(A1)
        inputArray.append(X[0][step])
        return (forward_prop(W1, b1, W2, b2, W3, X, time + 1, sum, step + 1, max_step, Aarray, inputArray))
    else:
        Z2 = W2 * X[0][step] + b2 + sum
        A2 = Z2
        return X[0][step], A2, Aarray, inputArray

def backward_prop(A1, A2, W1, W2, W3, b2, X, index, Aarray, inputArray):
    Y = X[0][index]
    dCdA2 = 2 * (A2 - Y)
    dCdW2 = dCdA2 * A1
    dCdb2 = dCdA2 
    dCdW1 = calculate_chain(A1, A2, Y, W1, W2, W3, Aarray, inputArray, Wnum=1)
    dCdW3 = calculate_chain(A1, A2, Y, W1, W2, W3, Aarray, inputArray, Wnum=3)
    return dCdW1, dCdW2, dCdW3, dCdb2

def calculate_chain(A1, A2, Y, W1, W2, W3, Aarray, inputArray, Wnum):
    if Wnum == 1:
        dCdA2 = 2 * (A2 - Y)
        init = W2 * dCdA2 * inputArray[-1]
        sum = 0
        for i in range(0, len(inputArray)):
            sum =  W2 * dCdA2
            for j in range(len(inputArray) - 1, -1 + i, -1):
                sum = sum * ReLU_deriv(Aarray[j]) * W3 * inputArray[j]  
            init += sum
        return init
    elif Wnum == 3:
        dCdA2 = 2 * (A2 - Y)
        init = W2 * dCdA2 * inputArray[-1]
        sum = 0
        for i in range(0, len(Aarray)):
            sum =  W2 * dCdA2
            for j in range(len(Aarray) - 1, -1 + i, -1):
                sum = sum * ReLU_deriv(Aarray[j]) * W3 * Aarray[j]   
            init += sum
        return init

def get_ratio(val, Y):
    if abs(val / Y) > 1:
        return abs(Y / val)
    else: return abs(val / Y)


def gradient_descent(X, learning_rate, iterations):
    W1, b1, W2, b2, W3 = init_params()
    for j in range(iterations):
        for i in range(train_dataX.shape[1] - DAYS_TRACKING - 2):
            A1, A2, Aarray, inputArray = forward_prop(W1, b1, W2, b2, W3, X, 0, 0, i, i + DAYS_TRACKING, [], [])
            dW1, dW2, dW3, db2 = backward_prop(A1, A2, W1, W2, W3, b2, X, i+DAYS_TRACKING + 1, Aarray, inputArray)
            W1 -= learning_rate * dW1
            W2 -= learning_rate * dW2
            W3 -= learning_rate * dW3
            b2 -= learning_rate * db2
        print("done")
    return W1, b1, W2, b2, W3
            
W1, b1, W2, b2, W3 = gradient_descent(train_dataX, 0.001, 1000)

def make_predictions(W1, b1, W2, b2, W3, X, time, sum, Aarray, inputArray):
    sum = 0
    for i in range (X.shape[1] - DAYS_TRACKING - 2):
        _, A2, _, _, = forward_prop(W1, b1, W2, b2, W3, X, time, sum, i, i + DAYS_TRACKING, Aarray, inputArray)
        sum += get_ratio(A2, X[0][i+4])
    sum = abs(sum / (X.shape[1] - DAYS_TRACKING - 2))
    print("%Off total price Avg. Accuracy: " + str(sum))
    
make_predictions(W1, b1, W2, b2, W3, test_dataX, 0, 0, [], [])
make_predictions(W1, b1, W2, b2, W3, train_dataX, 0, 0, [], [])

def get_specific_examples(W1, b1, W2, b2, W3, X, time, sum, Aarray, inputArray):
    num = randint(20, 100)
    _, A2, _, _, = forward_prop(W1, b1, W2, b2, W3, X, time, sum, num, num + DAYS_TRACKING, Aarray, inputArray)
    Y = X[0][num+DAYS_TRACKING+1]
    A2 = unnormalize_data(A2, meanTe)
    Y = unnormalize_data(Y, meanTe)
    print("Expected Stock closing price on day: " + str(num + DAYS_TRACKING + 1) + " with data starting from day: " + str(num) + " value: " + str(A2))
    print("Actual Stock closing price on day: " + str(num + DAYS_TRACKING + 1) + " with data starting from day: " + str(num) + " value: " + str(Y))

print("Examples:")
get_specific_examples(W1, b1, W2, b2, W3, test_dataX, 0, 0, [], [])
get_specific_examples(W1, b1, W2, b2, W3, test_dataX, 0, 0, [], [])
get_specific_examples(W1, b1, W2, b2, W3, test_dataX, 0, 0, [], [])
get_specific_examples(W1, b1, W2, b2, W3, test_dataX, 0, 0, [], [])