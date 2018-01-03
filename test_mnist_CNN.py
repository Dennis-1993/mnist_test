

from __future__ import print_function
import numpy as np
import skimage.measure
import time
from scipy import signal
from keras.datasets import mnist

dot = np.dot

batch_size = 128
num_classes = 10
epochs = 500

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print (x_train.shape)
print (x_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

Filter1 = np.random.random((32,3,3))
Filter2 = np.random.random((64,32,3,3))

theta1 = np.random.random((128,9216))
theta2 = np.random.random((10,128))

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
    
def max_pool_fun(A,n=2):
    z = skimage.measure.block_reduce(A,(1,1,n,n),np.mean)
    return z
    
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
    h = 1.0/(1.0+np.e**-z)
    return h
    
def dsigmoid(z):
    h = 1.0/(1.0+np.e**-z)
    g = h*(1-h)
    return g

def Conv_forword(X,Filter,activation = relu):
    Z = Conv(X,Filter)
    a = activation(Z)
    return a
    
def Conn_forword(X,theta,activation = sigmoid):
    Z = active_val(theta,X)
    a = activation(Z)
    return a

Conv_forword(x_train[0:batch_size],Filter1)
    
for z in range (0,epochs):
    print ('epoch :',z+1)

    for i in range (0,x_train.shape[0]-batch_size,batch_size) :
#   convolve 
        s_time = int(time.time())
        a1 = x_train[i:i+batch_size]
        #print(a1.shape)
        Z2 = Conv(a1,Filter1)
        print('Z2',Z2.shape)
        a2 = relu(Z2)
        #a2 = max_pool_fun(a2)
        Z3 = Conv(a2,Filter2)
        print('Z3',Z3.shape)
        a3 = relu(Z3)
        a3 = max_pool_fun(a3)

        a3 = a3.reshape((a3.shape[0],a3.shape[1]*a3.shape[2]*a3.shape[3]))

        Z4 = active_val(theta1,a3)
        #print('Z3',Z3.shape)
        a4 = sigmoid(Z4)
        #print('a3',a3.shape)
        Z5 = active_val(theta2,a4)
        #print('Z4',Z4.shape) 
        a5 = sigmoid(Z5)
        print('a5',a5.shape)
        e_time = int(time.time())    
        print("%02d:%02d:%02d" %((e_time-s_time)/3600,(e_time-s_time)%3600/60,(e_time-s_time)%60))
        
#b1 = np.ones((batch_size,500))
#b2 = np.ones((batch_size,100))
#b3 = np.ones((batch_size,10))  


    
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


    
    
    if z%5 == 0 :
        costFunction(theta1,theta2,theta3,x_train,y_train)
        test_fun(x_train,y_train,theta1,theta2,theta3)        
    if z%10 == 0 :
        test_fun(x_test,y_test,theta1,theta2,theta3)  
        costFunction(theta1,theta2,theta3,x_test,y_test)

test_fun(x_test,y_test,theta1,theta2,theta3)  
costFunction(theta1,theta2,theta3,x_test,y_test)    

