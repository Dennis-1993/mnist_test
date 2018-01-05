# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:02:41 2018

@author: dezhou
"""
import numpy as np
from scipy import signal

AA = [1,2,3,4,5,6,7,8,9]
BB = [5,6,7,8]
CC = np.convolve(AA,BB,'valid')
print (AA,'\n',BB,'\n',CC)

#AA = np.reshape(AA,(4,4))
#BB = np.reshape(BB,(3,3))
#CC = np.random.random((2,2))
AA = np.arange(16).reshape((4,4))
BB = np.arange(9).reshape((3,3))
def conv(A,Filter,s=1,zp=0) :
    Width = int((A.shape[0] - Filter.shape[0] + 2*zp)/s + 1)
    High = int((A.shape[1] - Filter.shape[1] + 2*zp)/s + 1)
    CC = np.zeros((Width,High))
    
    for i in range(CC.shape[0]):
        for j in range(CC.shape[1]) :
            CC[i,j] =np.sum(A[i:i+2,j:j+2] * Filter)
            print (A,'\n',Filter,'\n',CC)
    return CC

#conv(AA,BB)

CC = signal.convolve2d(AA,BB,'full')
print (AA.shape,BB.shape,CC.shape)
'''
def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))

A = np.arange(16).reshape((4,4))
B = FZ(A)
print(A,'\n',B)

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


def Conv3d(A,Filter,s=1,zp=0,bais = 0) :

    tmp_F = np.zeros_like(Filter)
    if Filter.ndim == 3 :        
        for d in range(Filter.shape[0]):
            tmp_F[d] = FZ(Filter[d])
        Width = int((A.shape[1] - Filter.shape[1] + 2*zp)/s + 1)
        High = int((A.shape[2] - Filter.shape[2] + 2*zp)/s + 1)
        Depth = Filter.shape[0]
        CC = np.zeros((A.shape[0],Depth,Width,High))
        for k in range(CC.shape[0]):
            for d in range(CC.shape[1]):
                CC[k,d,:,:]=signal.convolve2d(A[k],tmp_F[d],'valid')
                #print (A.shape,'\n',Filter.shape,'\n',CC.shape)
                
    if Filter.ndim == 4 :        
        tmp_F = Filter.reshape((Filter.shape[0],Filter.shape[1],Filter.shape[2]*Filter.shape[3]))[:,::-1,::-1].reshape((Filter.shape[0],Filter.shape[1],Filter.shape[2],Filter.shape[3]))
        print('tmp_F',tmp_F)        
        Width = int((A.shape[2] - Filter.shape[2] + 2*zp)/s + 1)
        High = int((A.shape[3] - Filter.shape[3] + 2*zp)/s + 1)
        Depth = Filter.shape[0]
        #print (Depth,Width,High)
        CC = np.zeros((A.shape[0],Depth,Width,High))
        for k in range(CC.shape[0]):
            for d in range(CC.shape[1]):
                CC[k,d,:,:]= signal.convolve(A[k,:,:,:],tmp_F[d,:,:,:],'valid')  
                print ('A_T',A[k,:,:,:],'\F_T',tmp_F[d,:,:,:],'\n',CC.shape)    
    CC += bais        
    return CC

    
AA = np.arange(54).reshape((2,3,3,3))
BB = np.arange(36).reshape((3,3,2,2))
CC = Conv(AA,BB)
print('AA:\n',AA,'\n BB:\n',BB,'\n CC:\n',CC)

#AA = AA.reshape((3,3*3))[:,::-1].reshape((3,3,3))
#BB = BB.reshape((3,2*2))[:,::-1].reshape((3,2,2))
print (AA,'\n',BB)

DD = np.arange(16).reshape((2,2,2,2))
print (DD)

DD = DD.reshape(2,2,4)[:,:,::-1].reshape(2,2,2,2)
print(DD)

DD = np.arange(16).reshape((2,2,2,2))
for d in range(2):
    for n in range(2):                
        DD[d,n,:,:] = FZ(DD[d,n,:,:])
print (DD)
        

CC = Conv(AA,BB)
print ('A :\n ',AA,'\n B :\n',BB,'\n C :\n',CC)

#BB = BB.reshape((2,3,2*2))[:,::-1].reshape((2,3,2,2))
CC = Conv3d(AA,BB)
print ('A3 :\n ',AA,'\n B3 :\n',BB,'\n C3 :\n',CC)

CC = signal.convolve(AA[0],BB[0],'valid')
print ('AA :\n ',AA,'\n BB :\n',BB,'\n CC :\n',CC)
'''