#importing packages
import cv2
import train
import numpy as np
from PIL import Image
from scipy import ndimage
import requests
import argparse as ap
from io import BytesIO
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.externals import joblib
#%% initialization
sift = cv2.xfeatures2d.SIFT_create()
#_, K_Means, stdSlr, n_c,clus = joblib.load("C:/Users/akshatb/Desktop/BE-Project/Code/m_2k_250c_dt.kg")
#model = load_model("C:/Users/akshatb/Desktop/BE-Project/Code/m_2k_250c_nn.h5")
K_Means, stdSlr, n_c,clus = joblib.load("C:/Users/akshatb/Desktop/BE-Project/Code/5k_250c.kg")
model = load_model("C:/Users/akshatb/Desktop/BE-Project/Code/5k_250c_nn.h5")
#%%
parser = ap.ArgumentParser()
parser.add_argument("-lp", "--localPath", help="Path to local image")
parser.add_argument("-up", "--urlPath", help="Path to URL")
args = vars(parser.parse_args())
#%% for local files
def test_local_image(l_path):
    img = np.array(ndimage.imread(l_path, flatten=False))
    pred = des_hist_predict(img)
    plt.imshow(img)
    plt.title(pred)
    plt.show()
#%% for URL paths
def test_url(url):
    response = requests.get(url)
    img = np.array(Image.open(BytesIO(response.content)))
    pred = des_hist_predict(img)
    plt.imshow(img)
    plt.title(pred)
    plt.show()
#%% helper class
def des_hist_predict(img):
    kp, des = sift.detectAndCompute(img,None)
    ret = K_Means.predict(des)
    histogram = np.zeros((1,n_c))
    for w in ret:
        histogram[0][w] += 1
    histogram = stdSlr.transform(histogram)
    pred = model.predict(histogram)
    pred = pred>0.1
    pred = display(pred)
    return pred
#%%
def display(pred):
    #print("Genre of the movie:")
    pred = pred[0].tolist()
    lis =""
    genre = ["Action", "Comedy", "Crime", "Drama", "Horror", "Romance", "Sci Fiction", "Thriller"]
    for i in range(len(pred)):
        if pred[i]:
            lis = lis + genre[i] +"   "
    print(lis)
    return lis
#%% customised metrics for calculating precision of multilabel classes
def multilabel_prec(X,Y):
    X = train.reshapeData(X)
    des_X,des_len_X, kp_X = train.desGenStack(X)
    ret_X = K_Means.predict(des_X)
    hist_X = train.createHist(ret_X, X.shape[0], n_c, des_len_X)
    hist_X = stdSlr.transform(hist_X)
    pred = model.predict(hist_X)
    pred = pred>0.1
    prec = np.sum(pred+Y == 2) / np.sum(Y == 1)
    return prec
#%%
if args["localPath"] is not None:
    test_local_image(args["localPath"])
if args["urlPath"] is not None:
    test_url(args["urlPath"])
