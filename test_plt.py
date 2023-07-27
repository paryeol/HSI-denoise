import numpy as np
import scipy.io as scio
import cv2
from matplotlib import pyplot as plt
def aver_normlize(x):
    mean = np.mean(x)
    var = np.mean(np.square(x - mean))
    x = (x - mean) / (6*np.sqrt(var))+0.5
    return x

def scaler_mean(x):
    c,h,w=np.shape(x)
    y=[]
    for i in range(c):
        tmp=aver_normlize(x[i])
        y.append(tmp)
        #print(np.max(tmp), np.min(tmp), len(np.where(tmp < 0)[0]), len(np.where(tmp > 1)[0]))
    y=np.array(y)
    return np.clip(y,0,1)
# 归一化
def scaler(x):
    # mean=np.mean(x)
    # dis=np.max(np.abs(x-mean))
    # x=0.5+(x-mean)/(2*dis)

    # c,h,w=x.shape
    # for i in range(c):
    #     max = np.max(x[i])
    #     min = np.min(x[i])
    #     x[i]=(x[i]-min)/(max-min)


    return x
orig=scio.loadmat( r"..\data\GT_crop_train.mat")['img']
orig_scaler=scaler(orig)
#orig_scaler=scaler_mean(orig)
plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#plt.plot(orig_scaler[:,100,100],'black')
plt.imshow(orig_scaler[0])
plt.show()

cv2.imshow('band',orig_scaler[0])
cv2.waitKey()
cv2.destroyWindow('band')