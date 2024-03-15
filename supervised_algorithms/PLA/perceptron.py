import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import pandas as pd

from sklearn.decomposition import PCA
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

if __name__ == "__main__":
    pla = Perceptron(n_iters=10000)
    
    iris = datasets.load_iris()
    iris_df = pd.DataFrame( iris.data, columns=iris.feature_names)
    
    iris_df['target'] = iris.target
    
    # Map targets to target names
    target_names = {
        0:'setosa',
        1:'versicolor',
        2:'virginica'
    }
    iris_df['target_names'] = iris_df['target'].map(target_names)
    
    df_no_setosa = iris_df[iris_df['target'] != 1]
    
    X_reduced = PCA(n_components=3).fit_transform(df_no_setosa.iloc[:, 0: 4])
    
    df_no_setosa['PC1'] = X_reduced[:, 0]
    df_no_setosa['PC2'] = X_reduced[:, 1]
    df_no_setosa['PC3'] = X_reduced[:, 2]
    
    features = df_no_setosa[["PC1", "PC2", "PC3"]]
    target = df_no_setosa["target"]
    pla.fit(features.values, target.values)
    
    w = pla.weights
    print(pla.weights)
    print(pla.bias)
    
    a,b,c,d = w[0],w[1],w[2],pla.bias
    
    x = np.linspace(-1,1,10)
    y = np.linspace(-1,1,10)
    
    X,Y = np.meshgrid(x,y)
    Z = (d - a*X - b*Y) / c
    
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    
    surf = ax.plot_surface(X, Y, Z)
    
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=df_no_setosa.target,
        s=40,
    )
    
    ax.set_title("First three PCA dimensions")
    ax.set_xlabel("1st Eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    ax.zaxis.set_ticklabels([])
