#旋转翻转放大
import cv2
import numpy as np
import scipy.io as scio
# import torchvision.transforms
from matplotlib import pyplot as plt
from PIL import Image
# from utils import scaler
from utils import *
from dataset_dc import TrainData,TrainData_ori
from torch.utils.data import DataLoader
import time
def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))#CHW->HWC
    if mode == 0:
        # original                    原图
        out = out
    elif mode == 1:
        # flip up and down            上下翻转
        #out = np.flipud(out)
        out = np.rot90(out, k=3)
        #out = np.flip(out,axis=0)#上下翻转
        #out = np.flip(out, axis=1)#左右翻转
        size=[0.8,1,1.2,1.4]
        #im.resize((320, 240), Image.BILINEAR)
        #out=np.resize(out,(int(size[2]*h),int(size[2]*w),c))
        #out=out.resize([int(size[2]*h),int(size[2]*w),c], Image.BILINEAR)
        h, w, c = out.shape
        out=cv2.resize(out, [int(size[3]*w),int(size[3]*h)], interpolation=cv2.INTER_LINEAR)

    elif mode == 2:
        # rotate counterwise 90 degree 逆向旋转90°
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down  翻转90°并上下翻转
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree             旋转180°
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip    旋转180°并反转
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree            旋转270°
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip    旋转270°并翻转
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))#HWC->CHW

def list_txt(path, list=None):
    '''

    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

# def bianhuan(image):
#     out_orig = np.transpose(image, (1, 2, 0))  # CHW->HWC
#     size = [0.8, 1, 1.2, 1.4]
#     paidui=[]
#     a=0
#     for i in range(4):
#         for m in range(4):
#             out = np.rot90(out_orig, k=i)  # 逆时针旋转i*90°
#             h, w, c = out.shape
#             #print(out.shape)
#             out = cv2.resize(out, [int(size[m] * w), int(size[m] * h)], interpolation=cv2.INTER_LINEAR)
#             out = np.transpose(out, (2, 0, 1))  # HWC->CHW
#             print('变换{}'.format(a),i,m,out.shape)
#             a += 1
#             paidui.append(out)
#     return paidui,a


# size=[0.8,1,1.2,1.4]
# print(size[0])
t1=time.time()
clean=scio.loadmat(r'D:\learn_pytorch\data\GT_crop_test_new.mat')['img']
clean=clean.copy()[:,0:50,0:50]
# print(np.shape(clean))
# clean_scaler= scaler_mean(clean)  # 干净HSI归一化
# target = clean_scaler.copy()
# img=data_augmentation(clean_scaler, mode=1)
list48,a=bianhuan(clean)
save_path=r'E:\read_test'
txt_path=r'D:\learn_pytorch\SSGN_orig/'
data_deal_new(list48,50,save_path,txt_path)


# for tmp in list48:
#     h=np.shape(tmp)[1]
#     w = np.shape(tmp)[2]
# list_part1=list48[:12]
# list_part2=list48[12:]
# np.save('list_part1.npy',list_part1)
# np.save('list_part2.npy',list_part2)

# print(len(res))
# #list_txt(path='D:\learn_pytorch\从服务器上copy过来\SSGN_orig\data\savelist.txt', list=list48)#保存
# #readlist=list_txt(path='D:\learn_pytorch\从服务器上copy过来\SSGN_orig\data\savelist.txt')#读取
# print(len(list48))

# t1=time.time()
# train_dataset = TrainData_ori( r"D:\learn_pytorch\data\GT_crop_train.mat",25,ToTensor(),24,2)
# train_loader = DataLoader(train_dataset,128, shuffle=True)
# for i,sample in enumerate(train_loader):
#     print(i)
#     print(time.time()-t1)
# data_shunxu = np.arange(0, 5, 1)
# np.random.shuffle(data_shunxu)
# print(data_shunxu,data_shunxu[0])
#
# img=list48[data_shunxu[0]]
# plt.figure()
# #解决中文显示问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# a=0
# plt.subplot(121)
# plt.title('波段{}变换前'.format(a+1))
# plt.imshow(target[a])
#
# plt.subplot(122)
# plt.title('波段{}变换后'.format(a+1))
# plt.imshow(img[a])
#
# plt.show()