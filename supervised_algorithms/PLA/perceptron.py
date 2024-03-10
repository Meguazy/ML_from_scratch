import numpy as np
import matplotlib
import pandas as pd

import matplotlib.pyplot as plt
from numpy import random

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        print("Initializating Perceptron...")
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_function = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        print("Starting the fit of the Perceptron...")
        # Extracting samples number and feature set size
        n_samples, n_features = X.shape

        # Setting the starting weights and bias
        self.weights = random.rand(n_features)
        self.bias = self.weights[0]
        print(self.weights)
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                lin_combination = self.bias + np.dot(x_i, self.weights)                
                y_pred = self.activation_function(lin_combination)

                if y_pred != y[idx]:
                    delta_w = self.lr * (y[idx] - y_pred)

                    self.weights += delta_w * x_i
                    self.bias += delta_w

    def predict(self, X):
        print("Starting the prediction...")
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)

def my_formula(x, w):
    return x**3+2*x-4

if __name__ == "__main__":
    pla_df = pd.read_csv("Single Layer Perceptron Dataset.csv", header=0)
    features = pla_df[["Feature2", "Feature3"]]
    labels = pla_df["Class_Label"]

    pla = Perceptron(learning_rate=0.1, n_iters=1000)
    pla.fit(features.values, labels.values)

    print(pla.predict(np.array([1,1])))
    print(pla.predict(np.array([0.4,0.2])))

    fig = plt.figure(figsize=(8,8))

    x = [-pla.bias/pla.weights[0], 0]
    y = [0, -pla.bias/pla.weights[1]]
    plt.plot(x,y)

    colors = ["red", "green"]
    plt.scatter(features.Feature2, features.Feature3, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    
    plt.show()