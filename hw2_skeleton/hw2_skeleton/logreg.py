import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.cost = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        N, d = x.shape
        x = np.hstack((np.ones((N, 1)), x))
        self.w = np.zeros((d + 1, 1))
        self.cost = np.zeros((self.num_iterations, 1))

        for i in range(1, self.num_iterations):
            y_predict = self.sigmoid(np.dot(x, self.w))
            self.cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1 - y, np.log(1 - y_predict)))
            self.w = self.w - self.learning_rate * np.dot(x.T, y_predict - y)

    def predict(self, x):
        if self.w is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return (self.sigmoid(np.dot(x, self.w)) >= 0.5).astype(int)