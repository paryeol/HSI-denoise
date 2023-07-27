#import cv2
import cv2
import scipy.io as scio              #数据输入输出
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from utils import *

def data_expand(orig_dir,crop_num):
    orig_imgs=scio.loadmat(orig_dir)['img']
    orig_imgs_scaler, max1, min1 = scaler1(orig_imgs)  # 干净HSI归一化
    data_tmp, op_num = bianhuan(orig_imgs_scaler)
    data=[]
    for i in range(crop_num):
        data=data+data_tmp
    return data

class TrainData_ori(Dataset):
    def __init__(self, orig_dir, patch_size, my_transform, K, crop_num):
        self.orig_imgs = scio.loadmat(orig_dir)['img']  # 读取mat文件
        # self.target_imgs = self.orig_imgs.copy()  # 复制一份给targetHSI
        self.orig_imgs_scaler, max1, min1 = scaler1(self.orig_imgs)  # 干净HSI归一化
        self.crop_num = crop_num

        self.data4, self.op_num = bianhuan(self.orig_imgs_scaler)  # 现在data4是一个list
        self.target_imgs4 = self.data4.copy()
        self.my_transform = my_transform
        self.patch_size = patch_size  # 图像块的大小
        self.channels_num = np.shape(self.orig_imgs)[0]
        self.K = K
        for j in range(len(self.data4)):
            self.data4[j] = get_noisy_image(self.data4[j])
        # self.add_noise_imgs4 = get_noisy_image(self.data4)

    def __getitem__(self, index):
        op_index = np.int(np.floor(index / self.crop_num / self.channels_num))
        spec_index = np.int(index / self.crop_num) % 191
        self.add_noise_imgs = self.data4[op_index]
        self.target_imgs = self.target_imgs4[op_index]

        add_noise_img = self.add_noise_imgs[spec_index]
        target_img = self.target_imgs[spec_index]
        target_img = target_img[np.newaxis, :, :]
        sobelx = cv2.Sobel(add_noise_img, -1, 1, 0, ksize=3)
        sobelx = sobelx[np.newaxis, :, :]
        sobely = cv2.Sobel(add_noise_img, -1, 0, 1, ksize=3)
        sobely = sobely[np.newaxis, :, :]
        sobel_spa = np.concatenate((sobelx, sobely), axis=0)
        # print(sobel_spa.shape)
        # 周围K+1波段光谱梯度
        add_noise_img = add_noise_img[np.newaxis, :, :]  # (1, h, w)，np.newaxis的作用就是在这一位置增加一个一维
        if spec_index < self.K // 2:
            clean_vol = self.add_noise_imgs[0:self.K + 1, :, :]
            target_im_25 = self.target_imgs[0:self.K + 1, :, :]
        elif spec_index > self.channels_num - self.K // 2 - 1:
            clean_vol = self.add_noise_imgs[-self.K - 1:, :, :]
            target_im_25 = self.target_imgs[-self.K - 1:, :, :]
        else:
            clean_vol = self.add_noise_imgs[spec_index - self.K // 2:spec_index + self.K // 2 + 1, :, :]
            target_im_25 = self.target_imgs[spec_index - self.K // 2:spec_index + self.K // 2 + 1, :, :]
        input_im_25 = clean_vol.copy()
        sobelxy_25 = []
        for i in range(clean_vol.shape[0]):
            sobelx1 = cv2.Sobel(clean_vol[i], -1, 1, 0, ksize=3)
            sobelx1 = sobelx1[np.newaxis, :, :]
            sobely1 = cv2.Sobel(clean_vol[i], -1, 0, 1, ksize=3)
            sobely1 = sobely1[np.newaxis, :, :]
            sobelxy_ = np.concatenate((sobelx1, sobely1), axis=0)  # 2*20*20
            sobelxy_25.append(sobelxy_)  # 25*2*20*20
        sobelxy_25 = np.array(sobelxy_25, dtype=np.float32)
        b, c, h, w = sobelxy_25.shape
        sobelxy_25 = sobelxy_25.reshape((b * c, h, w))  # 50*20*20
        # sobel_spe=spe_gradient_1(clean_vol)
        sobel_spe, _ = spe_gradient_11(clean_vol, spec_index, self.K, self.channels_num)
        input_im, input_im_grad_spa, input_im_25, input_im_25_grad_spa, input_im_grad_spe, target_im, target_im_25 = \
            rand_crop_ori(add_noise_img, sobel_spa, input_im_25, sobelxy_25, sobel_spe, target_img, target_im_25,
                      self.patch_size)

        sample = {
            'input_im': input_im,  # 1
            'input_im_grad_spa': input_im_grad_spa,  # 2
            'input_im_25': input_im_25,  # 25
            'input_im_25_grad_spa': input_im_25_grad_spa,  # 50
            'input_im_grad_spe': input_im_grad_spe,  # 24
            'target_im': target_im,
            'target_im_25': target_im_25,
            'index': index
        }
        sample = self.my_transform(sample)
        return sample

    def __len__(self):
        return self.op_num * self.crop_num * self.channels_num


class TrainData(Dataset):
    def __init__(self,txt_path,data_path,my_transform):
        self.index_list=load_txt(txt_path)
        self.data_path=data_path
        self.my_transform=my_transform
        #self.add_noise_imgs4 = get_noisy_image(self.data4)

    def __getitem__(self, index):
        tmp_sample=np.load(os.path.join(self.data_path,self.index_list[index]+'.npy'))
        input_im=tmp_sample['input_im']
        input_im_grad_spa=tmp_sample['input_im_grad_spa']
        input_im_25=tmp_sample[ 'input_im_25']
        input_im_25_grad_spa=tmp_sample['input_im_25_grad_spa']
        input_im_grad_spe=tmp_sample['input_im_grad_spe']
        target_im=tmp_sample['target_im']
        target_im_25= tmp_sample['target_im_25']
        sample = {
            'input_im': input_im,                          #1
            'input_im_grad_spa':input_im_grad_spa,         #2
            'input_im_25': input_im_25,                    #25
            'input_im_25_grad_spa': input_im_25_grad_spa,  #50
            'input_im_grad_spe':input_im_grad_spe,         #24
            'target_im': target_im,
            'target_im_25': target_im_25,
            'index':index
        }
        sample = self.my_transform(sample)
        return sample

    def __len__(self):
        return len(self.index_list)

class TrainData_split(Dataset):
    def __init__(self,index_list,data_path,my_transform,tmp_seq,rand_list):
        self.index_list=index_list
        self.data_path=data_path
        self.tmp_seq=tmp_seq
        self.rand_list=rand_list
        # #tmp_seq=np.arange(0,len(self.index_list))
        # tmp_seq = np.arange(0, 32)
        # print(len(self.index_list))
        # #random.shuffle(tmp_seq)
        # #self.rand_list=tmp_seq[0:2000]
        # self.rand_list=np.random.choice(tmp_seq,size=16,replace=False,p=None)#生成挑选出来的size个编号
        # remaining_list = np.delete(np.arange(tmp_seq.shape[0]),self.rand_list)  # 从原数组下标中删除已经选中的下标数组
        # tmp_seq=tmp_seq[remaining_list]
        # print('rand_list:'+str(self.rand_list))
        self.my_transform=my_transform
        #self.add_noise_imgs4 = get_noisy_image(self.data4)
    def __getitem__(self, index):
        # print('-------------------------------------------')
        # print('现在index:'+str(index))
        # print('rand_list[index]:' + str(self.rand_list[index]))
        #print('波段号:' + str((self.rand_list[index])%191))
        #print('index_list[self.rand_list[index]]:'+str(self.index_list[self.rand_list[index]]))
        tmp_sample=np.load(os.path.join(self.data_path,'{:07}.npy'.format(self.index_list[self.rand_list[index]])),allow_pickle=True).item()
        input_im=tmp_sample['input_im']
        input_im_grad_spa=tmp_sample['input_im_grad_spa']
        input_im_25=tmp_sample[ 'input_im_25']
        input_im_25_grad_spa=tmp_sample['input_im_25_grad_spa']
        input_im_grad_spe=tmp_sample['input_im_grad_spe']
        target_im=tmp_sample['target_im']
        target_im_25= tmp_sample['target_im_25']

        # orig=scio.loadmat(r'D:\learn_pytorch\data\GT_crop_train_new.mat')['img']
        # orig1=orig.copy()[:,0:50,0:50]
        # target_im1=np.squeeze(target_im)
        # # print(target_im1-orig1[self.rand_list[index]])

        sample = {
            'input_im': input_im,                          #1
            'input_im_grad_spa':input_im_grad_spa,         #2
            'input_im_25': input_im_25,                    #25
            'input_im_25_grad_spa': input_im_25_grad_spa,  #50
            'input_im_grad_spe':input_im_grad_spe,         #24
            'target_im': target_im,
            'target_im_25': target_im_25,
            'index':index
        }
        sample = self.my_transform(sample)
        return sample

    def __len__(self):
        #return len(self.rand_list)
        return len(self.rand_list)