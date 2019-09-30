
# coding: utf-8

#
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import pandas as p
from PIL import Image
from scipy import ndimage
from io import BytesIO
import requests
#%%
test = p.read_csv("C:/Users/akshatb/Desktop/BE-Project/Own Scrapper/Dataset1.csv")
t1 = test["Poster"]
t2 = test["Name"]
t3 = test["Genre"]
print(t1.shape)
print(len(t1))
#%%
count = []
for i in range(len(t1)):
    response = requests.get(t1[i])
    print(i," ",t1[i])
    img = Image.open(BytesIO(response.content))
    temp = np.array(img)
    print(temp.dtype)
    #print(temp.shape)
    plt.imshow(temp)
    plt.show()
    temp = temp.reshape((1, 278*185*3))
    #print(temp.shape)
    count.append(temp[0])
#%%
X = np.array(count)
X.shape
# %%
np.save("C:/Users/akshatb/Desktop/BE-Project/Own Scrapper/X_10k.npy",X)
#%%
X_ret = np.load("C:/Users/akshatb/Desktop/BE-Project/Own Scrapper/X_10k.npy")
#%%
#test1 = p.read_csv("C:/Users/akshatb/Desktop/BE-Project/Own Scrapper/temp.csv", engine="python")
#t = np.array(test1,dtype=np.uint8)
print(X_ret.shape)
print(X_ret.dtype)
#t = np.delete(t, (0), axis=1)
#print(t.shape[0])
#for i in range(t.shape[0]):
   # u = t[i].reshape(278,185,3)
   # plt.imshow(u)
   # plt.show()
u = X_ret[5].reshape(278,185,3)
plt.imshow(u)
plt.show()
#%%
Y = np.zeros((len(t3),10),dtype=np.uint8)
print(Y.shape)
print(t3.shape)
#%%
for i in range(len(t3)):
    Y[i][0] += ("Action" in t3[i])
    Y[i][1] += ("Comedy" in t3[i])
    Y[i][2] += ("Crime" in t3[i])
    Y[i][3] += ("Documentary" in t3[i])
    Y[i][4] += ("Drama" in t3[i])
    Y[i][5] += ("Horror" in t3[i])
    Y[i][6] += ("Romance" in t3[i])
    Y[i][7] += ("Thriller" in t3[i])
    print(Y[i])
#%%
print(t3[1398])
print(Y[1398])
#%%
np.save("C:/Users/akshatb/Desktop/BE-Project/Code/Data/Y_10k.npy",Y)
#%%
Y_ret = np.load("C:/Users/akshatb/Desktop/BE-Project/Own Scrapper/Y_10k.npy")
print(Y_ret[1555])
