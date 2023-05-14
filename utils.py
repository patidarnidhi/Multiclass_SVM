#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt




def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test





def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    for i in range(X_train.shape[1]):
        minval = np.min(X_train[:,i])
        maxval = np.max(X_train[:,i])
        data_range = maxval - minval

        if data_range==0:
            X_train[:,i] = 0.
        else:
            X_train[:,i] = (2.*(X_train[:,i]-minval)/data_range)-1
    

    for i in range(X_test.shape[1]):
        minval = np.min(X_test[:,i])
        maxval = np.max(X_test[:,i])
        data_range = maxval - minval

        if data_range==0:
            X_test[:,i] = 0.
        else:
            X_test[:,i] = (2.*(X_test[:,i]-minval)/data_range)-1
        
    return X_train , X_test



def plot_metrics(metrics) -> None:
    # plot and save the results

    metricsDF = pd.DataFrame(metrics, columns=["K", "Accuracy", "Precision", "Recall", "F1 score"])

    # Create a bar chart using the plot method of the DataFrame
    metricsDF.plot(x="K", y=["Accuracy", "Precision", "Recall", "F1 score"], kind="bar", figsize=(10,5))

    # Show the plot
    plt.show()
    
    

