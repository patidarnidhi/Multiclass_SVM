# #!/usr/bin/env python
# # coding: utf-8




from utils import get_data, plot_metrics, normalize
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from tqdm import tqdm
from sklearn.utils import shuffle





class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # fit the PCA model
        
        #subtracting each data point by mean
        new_X = X - np.mean(X,axis=0)
        
        #covariance matrix
        cov_mat = np.cov(new_X.T)
        
        #eigen values, eigen vectors of covariance matrix & sort them to find principle component
        e_val , e_vec = LA.eig(cov_mat)
        
        indices = e_val.argsort()[::-1]
        # e_val = e_val[indices]
        e_vec = e_vec[:,indices]
        
        #vector of given principle component
        self.components = e_vec[: , :self.n_components]
        
    
    def transform(self, X) -> np.ndarray:
        # transform the data
        
        reduced_data = np.dot(X , self.components)
        return reduced_data
    

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)





class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        self.w=np.random.uniform(0,1,size=len(X[0]))*0.01
        self.b=0.1




    def fit(
            self, X, Y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        
        
        self._initialize(X)
        

        
        # fit the SVM model using stochastic gradient descent
        shuffled = list(range(0,len(X)))

        
        for j in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            np.random.shuffle(shuffled)

            for i in shuffled:
                
                distance = 1 - (Y[i] * (np.dot(X[i], self.w)+self.b))
                distance = np.array([distance])
                dw = np.zeros(len(self.w))
                di=0
                b=0
                
                if max(0, distance) == 0:
                    di = self.w
                   
                else:
                    di = self.w - (C * Y[i] * X[i])
                    b = -1*C * Y[i]
        
                
                self.w = self.w - (learning_rate * di)
                self.b = self.b - (learning_rate * b)
            
        
        
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data

        y_train_predicted = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_train_predicted[i] = np.dot(X[i], self.w)
        y_train_predicted += self.b
        
        return y_train_predicted

    
    
    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        
        return np.mean((self.predict(X)>0) == (y>0))




class MultiClassSVM:
    
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    
    
    
    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        # then train the 10 SVM models using the preprocessed data for each class
    
        
        for label in np.unique(y):
            y_copy = np.where(y == label, 1, -1) 
            self.models[label].fit(X , y_copy , **kwargs)
        
    

    def predict(self, X ,y) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        score = np.zeros((X.shape[0], self.num_classes))

        for label in np.unique(y):
            score[:,label] = self.models[label].predict(X)
            
            
        y_predict = np.argmax(score , axis=1)
        return y_predict
        
        
        

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X,y) == y)
    
    
    def precision_score(self, X, y) -> float:
        y_pred = self.predict(X,y)
        precision_scores = []
        for label in np.unique(y):
            tp = np.sum((y == label) & (y_pred == label))
            fp = np.sum((y != label) & (y_pred == label))
            precision = tp / (tp + fp)
            precision_scores.append(precision)
        
        return np.mean(precision_scores)
        
    
    def recall_score(self, X, y) -> float:
        y_pred = self.predict(X,y)
        recall_scores = []
        for label in np.unique(y):
            tp = np.sum((y == label) & (y_pred == label))
            fn = np.sum((y == label) & (y_pred != label))
            recall = tp / (tp + fn)
            recall_scores.append(recall)
        
        return np.mean(recall_scores)
    
    
    def f1_score(self, X, y) -> float:
        ps = self.precision_score(X,y)
        rs = self.recall_score(X,y)
        f1_score = (2*ps*rs)/(ps+rs)
        return f1_score






