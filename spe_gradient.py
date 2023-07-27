import os

import cv2
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
def get_noise(x):
    c,h,w=x.shape
    sigma_signal=np.zeros([1,c])
    for i in range(c):
        sigma_signal[0,i]+=np.sum(x[i]**2)
    sigma_signal=sigma_signal/(h*w)
    SNR = 50+np.random.rand(1,c) * 10                  #numpy.ndarray,1行c列，值在10-20之间
    SNR1 =np.power(10.,SNR/10)                          #1行c列，值在10的1-2次方之间
    sigma_noi = sigma_signal/ SNR1            #所需要的噪音强度
    d=np.random.randn(h, w)
    noise_signal = np.sum(d**2)/(h*w)
    for i in range(c):
        noise=np.sqrt(sigma_noi[0][i]/noise_signal)*d
        x[i,:,:] = x[i,:,:] + noise

    #哪40波段要加噪声
    # band= (np.ceil((c - 10) * np.random.rand(1, c))).astype(int)#1行c列，值在0~（c-10）向上取整，即1~（c-10）+1
    #每个想加条带的波段具体要加条带的数量
    stripnum = (np.ceil((w - 10) * np.random.rand(1,w))).astype(int)#1行0.7*w列，值在0~（w-10）向上取整，即1~（w-10）+1之间
    for i in range(c):
        loc = (np.ceil((w-10)* np.random.rand(1, stripnum[0][i]))).astype(int)#这c波段中，每个波段具体是哪些列要做条带处理,值在0~(w-10)向上取整
        t = np.random.rand(1, loc.shape[1]) * 0.05 - 0.15 #要减去的值，1行loc列
        for j in range(loc.shape[1]):
            x[i,:,loc[0,j]]= x[i,:, loc[0,j]]-t[0,j]
    orig=scaler2(x)
    return orig
# 归一化
def scaler1(x):
    max = np.max(x)
    min = np.min(x)
    y = []
    for i in range(x.shape[0]):
        # print('x[i]='+str(x[i]),'min='+str(min),'x[i] - min='+str(x[i] - min))
        scaler = (x[i] - min) / (max - min)
        y.append(scaler)
    y = np.array(y, dtype=np.float32)
    return y,max,min

def scaler2(x):
    max=np.max(x)
    min=np.min(x)
    y=[]
    for i in range(x.shape[0]):
        fenzi=x[i]-min
        fenmu=max-min
        # print('x[i]:' + str(x[i]))
        scaler=fenzi/fenmu
        y.append(scaler)
    y = np.array(y, dtype=np.float32)
    return y,max,min

def scaler3(x,max1,min1,max2,min2):
    y=[]
    for i in range(x.shape[0]):
        fenzi1=x[i]-min1
        fenmu1=max1-min1
        scaler1=fenzi1/fenmu1
        fenzi2=scaler1-min2
        fenmu2=max2-min2
        scaler2=fenzi2/fenmu2
        y.append(scaler2)
    y = np.array(y, dtype=np.float32)
    return y

def spe_gradient_11(x,index,K,channels_num):
    spe_gra = []
    for i in range(np.shape(x)[0]):
        if index<K//2:
            gra = x[i] - x[index]
            cv2.imwrite(save_path + 'band' + str(i + 1) + '-' + 'band' + str(index+1) + '.jpg', gra)
        elif index>channels_num-K//2-1:
            gra=x[i]-x[K-(channels_num-index)+1]
            cv2.imwrite(save_path + 'band' + str(i + 1) + '-' + 'band' + str(index + 1) + '.jpg', gra)
        else :
            gra = x[i] - x[K//2]
            cv2.imwrite(save_path + 'band' + str(i + 1) + '-' + 'band' + str(index+ 1) + '.jpg', gra)
        spe_gra.append(gra)
    spe_gra = np.array(spe_gra, dtype=np.float32)
    spe_gra_dezero=spe_gra[~(spe_gra==0).all(axis=(1,2))]
    return spe_gra_dezero,index

orig=scio.loadmat('D:\learn_pytorch\data\GT_crop_test.mat')['img']
print('输入高光谱数据的尺寸：{}'.format(orig.shape))
save_path='D:/learn_pytorch/SSGN/res/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
c,_,_=orig.shape

target_imgs = orig.copy()
orig_scaler1,max1,min1=scaler1(orig)
orig_noise,max2,min2=get_noise(orig_scaler1)
target_imgs=scaler3(target_imgs,max1,min1,max2,min2)

a=50
# b=orig[a]-orig[a]
# plt.imshow(b)
# plt.show()
x=orig_noise[-25:,:,:]#25*200*200
y,index=spe_gradient_11(x,a,24,c)
print(y)
print('运行结束')
#
# #计算梯度
# sobelx = cv2.Sobel(orig[a],-1,1,0,ksize=3)
# sobely = cv2.Sobel(orig[a],-1,0,1,ksize=3)
# orig_imgs = orig[np.newaxis, :, :]  #(1, h, w)，np.newaxis的作用就是在这一位置增加一个一维
# index=a
# clean_im = orig[index]
# clean_im = clean_im[np.newaxis, :, :]  #(1, h, w)，np.newaxis的作用就是在这一位置增加一个一维
# if index < 12:
#     clean_vol = orig[0:25, :, :]
# elif index > c-14:
#     clean_vol = orig[-25:, :, :]
# else:
#     clean_vol =orig[index-12:index+13, :, :]#已修改为取  “前12波段+该波段+后12波段”
#     # clean_vol_1 = orig[index-12:index, :, :]
#     # clean_vol_2 = orig[index+1:index+13, :, :]
#     # clean_vol = np.concatenate((clean_vol_1, clean_vol_2), axis=0)
# clean_vol= np.array(clean_vol)
# print('准备用{}大小求光谱维梯度'.format(clean_vol.shape))
# sobelz=spe_gradient(clean_vol)
# print('光谱梯度mat文件已保存')

# plt.figure(1)
# plt.title('第{}波段原图放大'.format(a+1))
# plt.imshow(orig[a])
#
# plt.figure(2)
# #解决中文显示问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.subplot(131)
# plt.title('第{}波段原图'.format(a+1))
# plt.imshow(orig[a])
#
# plt.subplot(132)
# plt.title('水平梯度图')
# plt.imshow(sobelx)
#
# plt.subplot(133)
# plt.title('垂直梯度图')
# plt.imshow(sobely)
#
# plt.show()