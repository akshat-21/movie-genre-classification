#importing packages
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import time
#%% loading images and labels
def loadData(X, Y):
    data = X
    labels = Y
    print("Data loaded")
    return data, labels
#%% reshaping the input data
def reshapeData(data):
    data = data.reshape(data.shape[0], 278, 185, 3)
    return data
#%% find descriptors for all images and stack all descriptors from all images in a single numpy array 'des'
def desGenStack(data):
    sift = cv2.xfeatures2d.SIFT_create()
    modd = data.shape[0] // 100
    modd *= 10
    des = []
    des_len = []
    kp = []
    print("Generating descriptors for ",data.shape[0]," movies...")
    print("Movies Completed:")
    for i in range(data.shape[0]):
        p,d = sift.detectAndCompute(data[i],None)
        des += d.tolist()
        des_len.append(d.shape[0])
        kp.append(p)
        if i == data.shape[0] - 1:
            print(data.shape[0])
        if i % modd == 0:
            print(i)
    des = np.array(des)
    print("Descriptors generated and stacked in a single array")
    return des, des_len,kp
#%% clustering the descriptors using KMeans
def clusterDes(des, n_c):
    #K_Means =  KMeans(n_c, init = 'k-means++', n_init = 5)
    K_Means = MiniBatchKMeans(n_c, init = 'k-means++',batch_size = 128)
    print("Clustering the descriptors\nThis might take time...")
    K_Means.fit(des)
    ret = K_Means.predict(des)
    clus = K_Means.cluster_centers_
    print("Clustering done!")
    return K_Means, ret, clus
#%% creating histogram in which for every movie there is count of total descriptors of that movie
#for every cluster
def createHist(ret, dShape, n_c, des_len):
    print("Creating Vocabulary Histogram...")
    histogram = np.array([np.zeros(n_c) for i in range(dShape)])
    old_count = 0
    for i in range(dShape):
    	l = des_len[i]
    	for j in range(l):
    		if ret is None:
    			idx = ret[old_count+j]
    		else:
    			idx = ret[old_count+j]
    		histogram[i][idx] += 1
    	old_count += l
    print ("Vocabulary Histogram Generated")
    return histogram
#%%normalizing histogram
def normHist(histogram):
    stdSlr = StandardScaler().fit(histogram)
    histogram = stdSlr.transform(histogram)
    return histogram, stdSlr
#%% visualing histogram
'''x_scalar = np.arange(n_c)
y_scalar = np.array([abs(np.sum(mega_histogram[:,h], dtype=np.int32)) for h in range(n_c)])
print (x_scalar, y_scalar)
plt.bar(x_scalar, y_scalar)
plt.xlabel("Visual Word Index")
plt.ylabel("Frequency")
plt.title("Complete Vocabulary Generated")
plt.xticks(x_scalar + 0.4, x_scalar)
plt.show()'''
#%% training classifier using the normalized descriptors
def learn(stdHist, labels):
    '''clf = DecisionTreeClassifier(random_state=0)
    print("Fitting the training model\nThis might take time...")
    clf.fit(stdHist, lables)
    print("Model trained!")
    return clf'''
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_dim=stdHist.shape[1]))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=labels.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print("Fitting the training model\nThis might take time...")
    model.fit(stdHist, labels, epochs=150, batch_size=32)
    print("Model trained!")
    return model
#%% training using all functions above
def training(X,Y,filename, modelname, n_c = 100):
    data, labels = loadData(X, Y)
    data = reshapeData(data)
    des, des_len, kp = desGenStack(data)
    K_Means, ret, clus = clusterDes(des, n_c)
    histogram = createHist(ret, data.shape[0], n_c, des_len)
    histogram,stdSlr = normHist(histogram)
    model = learn(histogram, labels)
    joblib.dump((K_Means, stdSlr, n_c, clus), filename, compress=3) #saving various parameters on disk
    model.save(modelname)
    #print("Accuracy on train set:",model.score(norm_histogram, lables))
    print("Model saved on Disk!")