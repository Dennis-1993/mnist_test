# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:43:32 2017

@author: dezhou
"""



from __future__ import print_function
import numpy as np
from keras.datasets import mnist

dot = np.dot

batch_size = 64
num_classes = 10
epochs = 500

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = np.c_[np.ones(x_train.shape[0]),x_train]
x_test = np.c_[np.ones(x_test.shape[0]),x_test]

theta1 = np.random.random((511,785))*np.sqrt(2.0/(785+511)) - np.sqrt(2.0/(785+211))/2
theta2 = np.random.random((127,512))*np.sqrt(2.0/(512+127)) - np.sqrt(2.0/(512+127))/2
theta3 = np.random.random((10,128))*np.sqrt(2.0/(128+10)) - np.sqrt(2.0/(128+10))/2

#b1 = np.ones((batch_size,500))
#b2 = np.ones((batch_size,100))
#b3 = np.ones((batch_size,10))  

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
    x[x<0]=0.1
    x[x>0]=1    
    return x
    
def active_val(theta,X,b=0):
    Z = X.dot(theta.T) + b
    return Z
    
def sigmoid(z):
    g = 1.0/(1.0+np.e**-z)
    return g
    
def dsoftmax(x):
    
    return (np.exp(x).sum(axis=0) - np.exp(x)) / np.exp(x).sum(axis=0) / np.exp(x).sum(axis=0) 
    
def softmax(x):
    
    return np.exp(x) / np.exp(x).sum(axis=0)
    
def costFunction(theta1,theta2,theta3,X,y):
    m = y.size
    tmp_y =  np.zeros((m,num_classes),int)
#    for k in range(0,m):
#        tmp_y[k,y[k]] = 1
    for k in range(0,num_classes):
        tmp_y[:,k] = ((y==k)+0)
#    print('y :',tmp_y.shape)
    a1 = X
    #print('a1',a1.shape)
    Z2 = active_val(theta1,a1)
    #print ('Z2',Z2.shape)
    a2 = np.c_[np.ones(m),relu(Z2)]
    #print('a2',a2.shape)           
    Z3 = active_val(theta2,a2)
    #print('Z3',Z3.shape)
    a3 = np.c_[np.ones(m),relu(Z3)]
    #print('a3',a3.shape)
    Z4 = active_val(theta3,a3)
    #print('Z4',Z4.shape) 
    a4 = sigmoid(Z4)
    #print('a4',a4)
    h = a4
#    print('h :',h.shape)
    J = 1.0/m*(-tmp_y*(np.log(h))-(np.ones((m,num_classes),int)-tmp_y)*(np.log(np.ones((m,num_classes),int)-h)))
    loss = np.sum(J)
    print('loss :' ,loss)
    if np.isnan(loss):
        return np.inf
    return loss  
    
def test_fun(data,label,w1,w2,w3):
    sum_num = label.shape[0]
    right_num = 0
    i=0
    for value in data :

        Z2 = active_val(w1,value)
        #print ('C2',ReLu(Z2).shape)
        a2 = np.r_[np.ones(1),relu(Z2)]
        #print(a2)
        Z3 = active_val(w2,a2)
        #print('Z3',Z3.shape)
        a3 = np.r_[np.ones(1),relu(Z3)]
        Z4 = active_val(w3,a3)
        #print('Z4',Z4.shape) 
        y = sigmoid(Z4)
        #print(y)
        last_num = np.argmax(y)
        if last_num == label[i]:
            right_num +=1 
        i+=1
    print("accuracy",right_num/sum_num)

for z in range (0,epochs):
    print ('epoch :',z+1)

    for i in range (0,x_train.shape[0]-batch_size,batch_size) :
        theta1_d = np.zeros_like(theta1) 
        theta2_d = np.zeros_like(theta2)
        theta3_d = np.zeros_like(theta3)    
#        b1_d = np.zeros((batch_size,500))
#        b2_d = np.zeros((batch_size,100))
#        b3_d = np.zeros((batch_size,10))        
        a1 = x_train[i:i+batch_size]
        #print('a1',a1.shape)
        Z2 = active_val(theta1,a1)
        #print ('Z2',Z2.shape)
        a2 = np.c_[np.ones(batch_size),relu(Z2)]
        #print('a2',a2)           
        Z3 = active_val(theta2,a2)
        #print('Z3',Z3.shape)
        a3 = np.c_[np.ones(batch_size),relu(Z3)]
        #print('a3',a3.shape)
        Z4 = active_val(theta3,a3)
        #print('Z4',Z4.shape) 
        a4 = sigmoid(Z4)
        #print('a4',a4)
        
        y = y_train[i:i+batch_size]
        tmp_y =  np.zeros((batch_size,num_classes),int)
#        for k in range(0,batch_size):
#            tmp_y[k,y[k]] = 1
        for k in range(0,num_classes):
            tmp_y[:,k] = ((y==k)+0)
        delta4 = (a4-tmp_y)#*dsoftmax(Z4)
#        print ('delta4',delta4.shape)
#        print(dReLu(Z3).shape)
        delta3 = dot(delta4,theta3[:,1:])*drelu(Z3)#*a3[:,1:]*(1-a3[:,1:])
        #print ('delta3',delta3.shape)
        delta2 = dot(delta3,theta2[:,1:])*drelu(Z2)#*a2[:,1:]*(1-a2[:,1:])
        #print ('delta2',delta2.shape)
        
        theta1_d = theta1_d + dot(delta2.T,a1)/batch_size 
        theta2_d = theta2_d + dot(delta3.T,a2)/batch_size
        theta3_d = theta3_d + dot(delta4.T,a3)/batch_size
        theta1_d[:,1:] += 0.01*theta1[:,1:]/batch_size
        theta2_d[:,1:] += 0.01*theta2[:,1:]/batch_size
        theta3_d[:,1:] += 0.01*theta3[:,1:]/batch_size
#        b1_d += delta2
#        b2_d += delta3
#        b3_d += delta4
        theta1 = theta1 - 0.1/(np.sqrt(z)+1)*theta1_d 
        theta2 = theta2 - 0.05/(np.sqrt(z)+1)*theta2_d
        theta3 = theta3 - 0.03/(np.sqrt(z)+1)*theta3_d
#        b1 = b1 - 0.000005*b1_d/x_train.shape[0]
#        b2 = b2 - 0.000005*b2_d/x_train.shape[0]
#        b3 = b3 - 0.000005*b3_d/x_train.shape[0]
    if z%5 == 0 :
        costFunction(theta1,theta2,theta3,x_train,y_train)
        test_fun(x_train,y_train,theta1,theta2,theta3)        
    if z%10 == 0 :
        test_fun(x_test,y_test,theta1,theta2,theta3)  
        costFunction(theta1,theta2,theta3,x_test,y_test)

test_fun(x_test,y_test,theta1,theta2,theta3)  
costFunction(theta1,theta2,theta3,x_test,y_test)    

