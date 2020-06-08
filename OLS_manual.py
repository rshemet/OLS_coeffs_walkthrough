import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

class Model():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self, intercept = False):
        self.intercept = intercept

        x = np.hstack([np.ones(len(self.x))[:, np.newaxis], self.x]) if self.intercept else self.x

        y = self.y
        self.betas = np.linalg.inv(x.T @ x) @ x.T @ self.y

    def predict(self):
        
        x = np.hstack([np.ones(len(self.x))[:, np.newaxis], self.x])\
            if self.intercept else self.x
        
        self.y_hat = x @ self.betas

        # If you want to see the predictions, you can 'return self.y_hat' here

    def plot_predictions(self):
        plt.scatter(self.x, self.y, c='orange', label = 'Observed values')
        plt.plot(self.x, self.y_hat, label = 'Fitted values')
        plt.legend()
        title_label = 'Fitted OLS Regression: intercept={}, slope={}'\
            .format(np.round(self.betas[0][0],2), np.round(self.betas[1][0],2))
        plt.title(label=title_label)
        plt.show()
        
if __name__ == '__main__':
    x = np.arange(50)[:, np.newaxis]
    y = np.array([i + np.random.rand() for i in range(50)])[:, np.newaxis]
    ols_test = Model(x, y)
    ols_test.fit(intercept=True)
    ols_test.predict()
    ols_test.plot_predictions()