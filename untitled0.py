# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:32:35 2017

@author: dezhou
"""

import numpy as np
def softmax(x):
    print(np.exp(x).sum(axis=0))
    return np.exp(x) / np.exp(x).sum(axis=0)
def dsoftmax(x):
    
    return (np.exp(x).sum(axis=0) - np.exp(x)) / np.exp(x).sum(axis=0) / np.exp(x).sum(axis=0)     
scores = np.array([1.0, 2.0, 3.0])
print(softmax(scores))
print(dsoftmax(scores))
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
print(softmax(scores))
print(dsoftmax(scores))
def ReLu(X):
    shape = X.shape
    #print(shape)
    X_dim = 1
    if X.ndim == 2 :
        X = X.reshape((X.shape[0] * X.shape[1],1))
        X_dim = 2
    x = np.zeros_like(X)
    i = 0
    for value in X :
        if value < 0 :
            x[i] = 0
        else :
            x[i] = value
        i = i + 1
    if X_dim == 2:
        x = x.reshape((shape[0],shape[1]))
    #print ('x',x.shape)
    return x
    
def dReLu(X):
    shape = X.shape
    #print(shape)
    X_dim = 1
    if X.ndim == 2 :
        X = X.reshape((X.shape[0] * X.shape[1],1))
        X_dim = 2
    x = np.zeros_like(X)
    i = 0
    for value in X :
        print(value)
        if value < 0 :
            x[i] = 0
        else :
            x[i] = 1
        i = i + 1
    if X_dim == 2 :
        x = x.reshape((shape[0],shape[1]))
    #print('dx',x.shape)
    return x

def relu(x) :  
    #return np.maximum(0,x)  
    x[x<0] = 0.1*x[x<0]
    x[x>0] = x[x>0]
    return x   
    
def drelu(x) :
    x[x<0]=0
    x[x>0]=1    
    return x
    
A = np.random.random((5,4)) -0.5
D = np.random.random(10) -0.5
B = ReLu(A)
C = dReLu(A)
print (A ,'\n',B,'\n',C,'\n',D,'\n')   

def softmax1(x):
    """Compute softmax values for each sets of scores in x."""
    pass  # TODO: Compute and return softmax(x)
    x = np.array(x)
    x = np.exp(x)
    x.astype('float32')
    if x.ndim == 1:
        sumcol = sum(x)
        for i in range(x.size):
            x[i] = x[i]/float(sumcol)
    if x.ndim > 1:
        sumcol = x.sum(axis = 0)
        for row in x:
            print(row.shape)
            for i in range(row.size):
                row[i] = row[i]/float(sumcol[i])
    return x
#测试结果
print (softmax1(A))
print(softmax(A))
print(relu(A))
print(drelu(A))
print(relu(D))
print(drelu(D))

E = np.random.randint(0,10,size=[10])
print(E)
print((E==0) + 0)