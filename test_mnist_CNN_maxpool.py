# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:23:25 2018

@author: dezhou
"""



from __future__ import print_function
import numpy as np
import skimage.measure
import time
from scipy import signal
from keras.datasets import mnist

dot = np.dot

batch_size = 64
num_classes = 10
epochs = 50

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

Filter1 = np.random.random((32,3,3))*np.sqrt(2/(28*28+26*26)) - np.sqrt(2/(28*28+26*26))/2
Filter2 = np.random.random((64,32,3,3))*np.sqrt(2/(26*26+24*24)) - np.sqrt(2/(26*26+24*24))/2

theta1 = np.random.random((128,9216))*np.sqrt(2/(128+9216)) - np.sqrt(2/(128+9216))/2
theta2 = np.random.random((10,128))*np.sqrt(2/(10+128)) - np.sqrt(2/(10+128))/2

def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))
    
def conv(A,Filter,s=1,zp=0,bais = 0) :
    
    Width = int((A.shape[1] - Filter.shape[1] + 2*zp)/s + 1)
    High = int((A.shape[2] - Filter.shape[2] + 2*zp)/s + 1)
    CC = np.zeros((A.shape[0],Filter.shape[0],Width,High))
    for k in range(CC.shape[0]):
        for d in range(CC.shape[1]):
            for i in range(CC.shape[2]) :
                for j in range(CC.shape[3]) :
                    CC[k,d,i,j] =np.sum(A[k,i:i+Filter.shape[1],j:j+Filter.shape[2]] * Filter[d])
                    #print (A.shape,'\n',Filter.shape,'\n',CC.shape)
    CC += bais        
    return CC
    
def Conv(A,Filter,s=1,zp=0,bais = 0) :
    
#    tmp_F = np.zeros_like(Filter)
    if Filter.ndim == 3 :   
        tmp_F = Filter.reshape((Filter.shape[0],Filter.shape[1]*Filter.shape[2]))[:,::-1].reshape((Filter.shape[0],Filter.shape[1],Filter.shape[2]))
#        for d in range(Filter.shape[0]):
#            tmp_F[d] = FZ(Filter[d])
        Width = int((A.shape[1] - Filter.shape[1] + 2*zp)/s + 1)
        High = int((A.shape[2] - Filter.shape[2] + 2*zp)/s + 1)
        Depth = Filter.shape[0]
        CC = np.zeros((A.shape[0],Depth,Width,High))
        for k in range(CC.shape[0]):
            for d in range(CC.shape[1]):
                CC[k,d,:,:]=signal.convolve2d(A[k],tmp_F[d],'valid')
                #print (A.shape,'\n',Filter.shape,'\n',CC.shape)
                
    if Filter.ndim == 4 :
        tmp_F = Filter.reshape((Filter.shape[0],Filter.shape[1],Filter.shape[2]*Filter.shape[3]))[:,:,::-1].reshape((Filter.shape[0],Filter.shape[1],Filter.shape[2],Filter.shape[3]))
#        for d in range(Filter.shape[0]):
#            for n in range(Filter.shape[1]):                
#                tmp_F[d,n,:,:] = FZ(Filter[d,n,:,:])
        Width = int((A.shape[2] - Filter.shape[2] + 2*zp)/s + 1)
        High = int((A.shape[3] - Filter.shape[3] + 2*zp)/s + 1)
        Depth = Filter.shape[0]
        #print (Depth,Width,High)
        CC = np.zeros((A.shape[0],Depth,Width,High))
        for k in range(CC.shape[0]):
            for d in range(CC.shape[1]):
                tmp = np.zeros((Width,High))
                #print ("tmp",tmp.shape)
                for n in range(Filter.shape[1]):
                    tmp += signal.convolve2d(A[k,n,:,:],tmp_F[d,n,:,:],'valid')        
                CC[k,d,:,:]= tmp
                #print (A.shape,'\n',Filter.shape,'\n',CC.shape)    
    CC += bais        
    return CC
    
def Conv_back(A,Filter,s=1,zp=0,bais = 0) :
    if Filter.ndim == 3 :   
        Width = int((A.shape[2] - Filter.shape[1] + 2*zp)/s + 1)
        High = int((A.shape[3] - Filter.shape[2] + 2*zp)/s + 1)
        CC = np.zeros((A.shape[0],Width,High))
        #print (CC.shape)
        for k in range(CC.shape[0]):
            tmp = np.zeros((Width,High))
            for d in range(Filter.shape[0]):
                   tmp += signal.convolve2d(A[k,d,:,:],Filter[d,:,:],'full')
            CC[k,:,:]= tmp
        #print (A.shape,'\n',Filter.shape,'\n',CC.shape)
                
    if Filter.ndim == 4 :
        Width = int((A.shape[2] - Filter.shape[2] + 2*zp)/s + 1)
        High = int((A.shape[3] - Filter.shape[3] + 2*zp)/s + 1)
        Depth = Filter.shape[1]
        #print (Depth,Width,High)
        CC = np.zeros((A.shape[0],Depth,Width,High))
        for k in range(CC.shape[0]):
            for d in range(CC.shape[1]):
                tmp = np.zeros((Width,High))
                #print ("tmp",tmp.shape)
                for n in range(Filter.shape[0]):
                    tmp += signal.convolve2d(A[k,n,:,:],Filter[n,d,:,:],'full')        
                CC[k,d,:,:]= tmp
                #print (A.shape,'\n',Filter.shape,'\n',CC.shape)    
    CC += bais        
    return CC 
    
def Conv_grad(A,Filter,s=1,zp=0,bais = 0) :
    
#    tmp_F = np.zeros_like(Filter)
    if A.ndim == 3 :   
        tmp_F = Filter.reshape((Filter.shape[0],Filter.shape[1],Filter.shape[2]*Filter.shape[3]))[:,:,::-1].reshape((Filter.shape[0],Filter.shape[1],Filter.shape[2],Filter.shape[3]))
#        for d in range(Filter.shape[0]):
#            tmp_F[d] = FZ(Filter[d])
        Width = int((A.shape[1] - Filter.shape[2] + 2*zp)/s + 1)
        High = int((A.shape[2] - Filter.shape[3] + 2*zp)/s + 1)
        Depth = Filter.shape[1]
        CC = np.zeros((Depth,Width,High))
        for d in range(CC.shape[0]):
            tmp = np.zeros((Width,High))
            for k in range(A.shape[0]):
                tmp += signal.convolve2d(A[k],tmp_F[k,d],'valid')
            CC[d,:,:]=tmp
            #print (A.shape,'\n',Filter.shape,'\n',CC.shape)
                
    if A.ndim == 4 :
        tmp_F = Filter.reshape((Filter.shape[0],Filter.shape[1],Filter.shape[2]*Filter.shape[3]))[:,:,::-1].reshape((Filter.shape[0],Filter.shape[1],Filter.shape[2],Filter.shape[3]))
#        for d in range(Filter.shape[0]):
#            for n in range(Filter.shape[1]):                
#                tmp_F[d,n,:,:] = FZ(Filter[d,n,:,:])
        Width = int((A.shape[2] - Filter.shape[2] + 2*zp)/s + 1)
        High = int((A.shape[3] - Filter.shape[3] + 2*zp)/s + 1)
        Depth = Filter.shape[1]
        #print (Depth,Width,High)
        CC = np.zeros((Depth,A.shape[1],Width,High))
        #print(CC.shape)
        for d in range(CC.shape[0]):
            for n in range(CC.shape[1]):
                tmp = np.zeros((Width,High))
                #print ("tmp",tmp.shape)
                for k in range(A.shape[0]):
                    tmp += signal.convolve2d(A[k,n,:,:],tmp_F[k,d,:,:],'valid')        
                CC[d,n,:,:]= tmp
                #print (A.shape,'\n',Filter.shape,'\n',CC.shape)    
    CC += bais
    #print('CC',CC.shape)       
    return CC  
    
def mean_pool_fun(A,n=2):
    z = skimage.measure.block_reduce(A,(1,1,n,n),np.mean)
    return z
    
def mean_pool_back(A,n=2):
    N = np.ones((n,n))/n/n
    Z = np.kron(A,N)
    return Z

def max_pool_fun(A,n=2):
    z = skimage.measure.block_reduce(A,(1,1,n,n),np.max)
    N = np.ones((n,n))
    Z = np.kron(z,N)     
    P = A - Z
    P[P==0] = 1
    P[P<0] = 0
    return z,P

def max_pool_back(A,P,n=2):
    N = np.ones((n,n))
    Z = np.kron(A,N)*P
    return Z      
    
def relu(x) :
    #return np.maximum(0,x)  
    x[x<0] = 0
    return x
    
def drelu(x) :
    x[x<0]=0
    x[x>0]=1    
    return x
    
def active_val(theta,X,b=0):
    Z = X.dot(theta.T) + b
    return Z  

def sigmoid(z):
    h = 1.0/(1.0+np.e**-z)
    return h
    
def dsigmoid(z):
    h = 1.0/(1.0+np.e**-z)
    g = h*(1-h)
    return g

def Conv_forword(X,Filter,s=1,zp=0,bais=0,activation = relu):
    Z = Conv(X,Filter,s,zp,bais)
    a = activation(Z)
    return a
    
def Conn_forword(X,theta,b=0,activation = sigmoid):
    Z = active_val(theta,X,b)
    a = activation(Z)
    return a

#Conv_forword(x_train[0:batch_size],Filter1)
def test_fun(data,label,Filter1,Filter2,theta1,theta2):
    sum_num = label.shape[0]
    right_num = 0
    
    a1 = data
    #print(a1.shape)
    Z2 = Conv(a1,Filter1)
    #print('Z2',Z2.shape)
    a2 = relu(Z2)
    #a2 = max_pool_fun(a2)
    Z3 = Conv(a2,Filter2)
    #print('Z3',Z3.shape)
    a3 = relu(Z3)
    a3,P1 = max_pool_fun(a3)

    A1 = a3.reshape((a3.shape[0],a3.shape[1]*a3.shape[2]*a3.shape[3]))
    ZC2 = active_val(theta1,A1)
    A2  = relu(ZC2)
    ZC3 = active_val(theta2,A2) 
    y = sigmoid(ZC3)        
    for k in range (y.shape[0]) :     
        last_num = np.argmax(y[k])
        if last_num == label[k]:
            right_num +=1 
    print("accuracy",right_num/sum_num)
    
def costFunction(X,y,Filter1,Filter2,theta1,theta2):
    m = y.size
    tmp_y =  np.zeros((m,num_classes),int)
#    for k in range(0,m):
#        tmp_y[k,y[k]] = 1
    for k in range(0,num_classes):
        tmp_y[:,k] = ((y==k)+0)
#    print('y :',tmp_y.shape)
    a1 = X
    #print(a1.shape)
    Z2 = Conv(a1,Filter1)
    #print('Z2',Z2.shape)
    a2 = relu(Z2)
    #a2 = max_pool_fun(a2)
    Z3 = Conv(a2,Filter2)
    #print('Z3',Z3.shape)
    a3 = relu(Z3)
    a3,P1 = max_pool_fun(a3)

    A1 = a3.reshape((a3.shape[0],a3.shape[1]*a3.shape[2]*a3.shape[3]))
    ZC2 = active_val(theta1,A1)
    A2  = relu(ZC2)
    ZC3 = active_val(theta2,A2) 
    A3 = sigmoid(ZC3)
    h = A3
#    print('h :',h.shape)
    J = 1.0/m*(-tmp_y*(np.log(h))-(np.ones((m,num_classes),int)-tmp_y)*(np.log(np.ones((m,num_classes),int)-h)))
    loss = np.sum(J)
    print('loss :' ,loss)
    if np.isnan(loss):
        return np.inf
    return loss  
    
for z in range (0,epochs):
    print ('epoch :',z+1)

    for i in range (0,2560-batch_size,batch_size) :
#   convolve 
        s_time = int(time.time())
        
        a1 = x_train[i:i+batch_size]
        #print(a1.shape)
        Z2 = Conv(a1,Filter1)
        #print('Z2',Z2.shape)
        a2 = relu(Z2)
        #a2 = max_pool_fun(a2)
        Z3 = Conv(a2,Filter2)
        #print('Z3',Z3.shape)
        a3 = relu(Z3)
        a3,P1 = max_pool_fun(a3)

        A1 = a3.reshape((a3.shape[0],a3.shape[1]*a3.shape[2]*a3.shape[3]))
        
        ZC2 = active_val(theta1,A1)
        #print('ZC2',ZC2.shape)
        A2  = relu(ZC2)
        #print('A2',A2.shape)
        ZC3 = active_val(theta2,A2)
        #print('ZC3',ZC3.shape) 
        A3  = sigmoid(ZC3)
        #print('A3',A3.shape)
        
        
        y = y_train[i:i+batch_size]
        tmp_y =  np.zeros((batch_size,num_classes),int)
#        for k in range(0,batch_size):
#            tmp_y[k,y[k]] = 1
        for k in range(0,num_classes):
            tmp_y[:,k] = ((y==k)+0)
        Delta3 = (A3-tmp_y)
        Delta2 = dot(Delta3,theta2)*drelu(ZC2)#*A2*(1-A2)
        #print ('Delta2',Delta2.shape)
        Delta1 = dot(Delta2,theta1)
        #print('Delta1',Delta1.shape)
        
        Delta1 = Delta1.reshape((a3.shape))
        
        delta3 = max_pool_back(Delta1,P1)*drelu(Z3)
        #print ('delta3',delta3.shape)        
        delta2 = Conv_back(delta3,Filter2,zp=2)*drelu(Z2)
        #print ('delta2',delta2.shape)        
        #delta1 = Conv_back(delta2,Filter1,zp=2)*drelu(a1)
        
        Filter1_d = Conv_grad(a1,delta2)/batch_size
        Filter2_d = Conv_grad(a2,delta3)/batch_size           
        Filter1 -= 0.025/(np.sqrt(z)+1)*Filter1_d
        Filter2 -= 0.02/(np.sqrt(z)+1)*Filter2_d

        theta1_d = dot(Delta2.T,A1)/batch_size 
        theta2_d = dot(Delta3.T,A2)/batch_size
#        theta1_d += 0.01*theta1/batch_size
#        theta2_d += 0.01*theta2/batch_size
        theta1 -= 0.02/(np.sqrt(z)+1)*theta1_d 
        theta2 -= 0.01/(np.sqrt(z)+1)*theta2_d
        
        e_time = int(time.time())    
        print("%02d:%02d:%02d" %((e_time-s_time)/3600,(e_time-s_time)%3600/60,(e_time-s_time)%60))
    costFunction(x_train[0:1000],y_train[0:1000],Filter1,Filter2,theta1,theta2)
    test_fun(x_test[0:1000],y_test[0:1000],Filter1,Filter2,theta1,theta2)

        


    
def dsoftmax(x):
    
    return (np.exp(x).sum(axis=0) - np.exp(x)) / np.exp(x).sum(axis=0) / np.exp(x).sum(axis=0) 
    
def softmax(x):
    
    return np.exp(x) / np.exp(x).sum(axis=0)
    


    
    
   

