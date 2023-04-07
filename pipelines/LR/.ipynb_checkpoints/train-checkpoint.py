import argparse
import subprocess
subprocess.call(['pip', 'install', 'dill'])
import numpy as np
import pandas as pd
import logging
import pathlib
import os
import pickle
import dill
import json
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
 
class SimpleLinearRegression:
    def __init__(self, iterations=15000, lr=0.1):
        self.iterations = iterations # number of iterations the fit method will be called
        self.lr = lr # The learning rate
        self.losses = [] # A list to hold the history of the calculated losses
        self.W, self.b = None, None # the slope and the intercept of the model

    def __loss(self, y, y_hat):
        """

        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: the sum of squared error

        """
        #ToDO calculate the loss. use the sum of squared error formula for simplicity
        # loss = np.sum( (y - y_hat)**2 )

        # vectorize the formulas.
        delta = y - y_hat
        loss = delta.T.dot(delta)

        self.losses.append(loss)
        return loss

    def __init_weights(self, X):
        """

        :param X: The training set
        """
        weights = np.random.normal(size=X.shape[1] + 1)
        self.W = weights[:X.shape[1]].reshape(-1, X.shape[1])
        self.b = weights[-1]

    def __sgd(self, X, y, y_hat):
        """

        :param X: The training set
        :param y: The actual output on the training set
        :param y_hat: The predicted output on the training set
        :return:
            sets updated W and b to the instance Object (self)
        """
        # ToDo calculate dW & db.
        # dW = -2 * np.mean(X*(y - y_hat))
        # db = -2 * np.mean(y - y_hat)
        # vectorize the formulas.
        # print( (y - y_hat).shape,X.shape)
        delta = y - y_hat
        n = X.shape[0]
        dW = -2 * delta.T.dot(X) / n
        db = -2 * np.mean(delta)
        #  ToDO update the self.W and self.b using the learning rate and the values for dW and db
        self.W -= self.lr * dW
        self.b -= self.lr * db


    def fit(self, X, y):
        """

        :param X: The training set
        :param y: The true output of the training set
        :return:
        """
        self.__init_weights(X)
        y_hat = self.predict(X)
        loss = self.__loss(y, y_hat)
        print(f"Initial Loss: {loss}")
        for i in range(self.iterations + 1):
            self.__sgd(X, y, y_hat)
            y_hat = self.predict(X)
            loss = self.__loss(y, y_hat)
            if not i % 100:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X):
        """

        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        #ToDO calculate the predicted output y_hat. remember the function of a line is defined as y = WX + b
        y_hat = X.dot(self.W) + self.b
        return y_hat
    

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--iterations', type=float, default=1.5e4)
    # parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    
    args, _ = parser.parse_known_args()
    
    
    model_dir=args.model_dir
    training_dir = args.training_dir
    iterations = args.iterations
    lr = args.lr
    model = SimpleLinearRegression()
    logger.info("Loading training data.")
    X_train = pd.read_csv(os.path.join(training_dir,"diabetes_X_train.csv")).values.reshape(-1,1)
    y_train = pd.read_csv(os.path.join(training_dir,"diabetes_y_train.csv")).values.reshape(-1,1)
    # df = pd.read_csv(os.path.join(training_dir,"train.csv"))
    # X_train = df.values[:,0].reshape(-1,1) 
    # y_train = df.values[:,1].reshape(-1,1)
    logger.info(f"shape of diabetes_X_train: {X_train.shape}")
    logger.info(f"shape of diabetes_y_train: {y_train.shape}")
    model.fit(X_train,y_train)
    
    
    with open(os.path.join(model_dir,'model.pkl'), 'wb') as s:
        dill.dump(model, s)