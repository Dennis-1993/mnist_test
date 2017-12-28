
from __future__ import print_function
import numpy as np
from keras.datasets import mnist

dot = np.dot

batch_size = 50
num_classes = 10
epochs = 50
isbias = 1

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if isbias == 1:
    x_train = np.c_[np.ones(60000),x_train]
    x_test = np.c_[np.ones(10000),x_test]
    theta = np.random.random((785,num_classes))/1000
else :
    theta = np.random.random((784,num_classes))/1000


def gradient(theta,X,y):
    m = y.size
    tmp_y =  np.zeros((m,num_classes),int)
    for k in range(0,m):
        tmp_y[k,y[k]] = 1
#    print ('tmp_y',tmp_y,k)
    tmp_theta = theta.reshape(X.shape[1],num_classes).copy()
#    print ('tmp_theta :' , tmp_theta)
    z = dot(X,tmp_theta)
#    print ('Z:',z)
    h = 1.0/(1.0+np.e**-z)
#    print ('H:',h)
    grad = 1.0/m*X.T.dot(h-tmp_y)
#    grad = grad.flatten
    return grad 

def hypothesis(X,theta):
    z = X.dot(theta)
    g = 1.0/(1.0+np.e**-z)
#    print(z)
    return g

def test_fun(data,label,w):
    sum_num = label.shape[0]
    right_num = 0
    i=0
    for value in data :
#        y = hypothesis(value,w)
        y = value.dot(w)
        last_num = np.argmax(y)
        if last_num == label[i]:
            right_num +=1 
        i+=1
    print("accuracy",right_num/sum_num)

def costFunction(theta,X,y):
    tmp_J = np.zeros((num_classes),float)
    m = y.size
    tmp_y =  np.zeros((m,num_classes),int)
    for k in range(0,m):
        tmp_y[k,y[k]] = 1
#    print('y :',tmp_y.shape)
    z = np.array(dot(X,theta))
#    print ('z :',z.shape)
    h = 1.0/(1.0+np.e**-z)
#    print('h :',h.shape)
    for k in range (0,num_classes) :
        J = 1.0/m*(-tmp_y.T[k].dot(np.log(h[:,k]))-(np.ones((m,num_classes),int)-tmp_y).T[k].dot(np.log(np.ones((m),int)-h[:,k])))
        #print('y :' ,tmp_y.T[k].shape)
        #print('J : ',np.log(h[:,k]).shape)
        tmp_J = tmp_J + J
    loss = np.sum(tmp_J)
    if np.isnan(loss):
        return np.inf
    return loss
    
for z in range (0,epochs):
    print ('epoch :',z+1)
    for i in range (0,x_train.shape[0]-batch_size,batch_size) :
        X = x_train[i:i+batch_size]
        y = y_train[i:i+batch_size]
        grad = gradient(theta,X,y)
#        print ('grad',grad)
        theta = theta - 0.5/(1+i/1000)*grad
    test_fun(x_train,y_train,theta)
    loss = costFunction(theta,x_train,y_train)
    print ('loss',loss)

test_fun(x_test,y_test,theta)
loss = costFunction(theta,x_test,y_test)
print ('test loss',loss)


    

