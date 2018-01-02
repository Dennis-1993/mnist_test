# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:02:41 2018

@author: dezhou
"""
import numpy as np

AA = [1,2,3,4,5,6,7,8,9]
BB = [5,6,7,8]
CC = np.convolve(AA,BB,'valid')
print (AA,'\n',BB,'\n',CC)

AA = np.reshape(AA,(3,3))
BB = np.reshape(BB,(2,2))
CC = np.random.random((2,2))
for i in range(CC.shape[0]):
    for j in range(CC.shape[1]) :
        CC[i,j] =np.sum(AA[i:i+2,j:j+2] * BB)
print (AA,'\n',BB,'\n',CC)