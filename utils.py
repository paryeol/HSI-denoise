import math
import os

import cv2
import torch
import torch.nn as nn
import numpy as np
import glob
import scipy.io as scio
from osgeo import gdal
from sklearn.preprocessing import MinMaxScaler
import skimage
import random
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tifffile import tifffile


def get_test_name():
    file = './ICVL_test_gauss.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

def get_train_name():
    file = './ICVL_train.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

# def bianhuan(image):
#     out_orig = np.transpose(image, (1, 2, 0))  # CHW->HWC
#     size = [0.8, 1, 1.2, 1.4]
#     paidui=[]
#     a=0
#     for i in range(4):
#         for j in range(2):
#             for m in range(2):
#                 if j == 2:
#                     out = np.rot90(out_orig, k=i)  # 逆时针旋转i*90°
#                     h, w, c = out.shape
#                     #print(out.shape,int(size[m] * w), int(size[m] * h))
#                     out = cv2.resize(out, [int(size[m] * w), int(size[m] * h)], interpolation=cv2.INTER_LINEAR)
#                     out = np.transpose(out, (2, 0, 1))  # HWC->CHW
#                 else:
#                     out = np.rot90(out_orig, k=i)  # 逆时针旋转i*90°
#                     out = np.flip(out, axis=j)  # j=0上下翻转，j=1左右翻转
#                     h, w, c = out.shape
#                     #print(out.shape)
#                     out = cv2.resize(out, [int(size[m] * w), int(size[m] * h)], interpolation=cv2.INTER_LINEAR)
#                     out = np.transpose(out, (2, 0, 1))  # HWC->CHW
#                 #print('变换{}'.format(a),i,j,m,out.shape)
#                 a += 1
#                 paidui.append(out)
#     return paidui,a
def bianhuan(image):
    out_orig = np.transpose(image, (1, 2, 0))  # CHW->HWC
    size = [0.8, 1, 1.2, 1.4]
    paidui=[]
    a=0
    # for i in range(2):
    #     for j in range(2):
    #         out = np.rot90(out_orig, k=i)  # 逆时针旋转i*90°
    #         h, w, c = out.shape
    #         #print(out.shape)
    #         if j==0:
    #             out = np.flip(out, axis=0)  # j=0上下翻转，j=1左右翻转
    #             # out = np.flip(cv2.resize(out, [int(size[m] * w), int(size[m] * h)], interpolation=cv2.INTER_LINEAR),out = np.trut, axis=1)  # j=0上下翻转，j=1左右翻转
    #         out = np.transpose(out, (2, 0, 1))  # HWC->CHW
    #         print('变换{}'.format(a),i,j,out.shape)
    #         a += 1
    #         paidui.append(out)
    for i in range(1):
        out = np.rot90(out_orig, k=i)  # 逆时针旋转i*90°
        out = np.transpose(out, (2, 0, 1))  # HWC->CHW
        print('变换{}'.format(a),i,out.shape)
        a += 1
        paidui.append(out)
    return paidui,a

def data_deal(list,patch_size,save_path,txt_path):
    K=24
    op_num=len(list)
    channel_num=len(list[0])

    # input_im=np.zeros([op_num*crop_num*channel_num,1,patch_size,patch_size])
    # input_im_grad_spa=np.zeros([op_num*crop_num*channel_num,2,patch_size,patch_size])
    # input_im_25=np.zeros([op_num*crop_num*channel_num,25,patch_size,patch_size])
    # input_im_25_grad_spa=np.zeros([op_num*crop_num*channel_num,50,patch_size,patch_size])
    # input_im_grad_spe=np.zeros([op_num*crop_num*channel_num,24,patch_size,patch_size])
    # target_im=np.zeros([op_num*crop_num*channel_num,1,patch_size,patch_size])
    # target_im_25=np.zeros([op_num*crop_num*channel_num,25,patch_size,patch_size])
    txt=open(os.path.join(txt_path,'train_list_zhu_mean.txt'),'w')
    #start_c=open('start_num.txt','r')
    #start_num=[np.int(t.split('\n')[0]) for t in start_c.readlines()]
    counter=0
    for p in range(op_num):
    # for p in range(16):
        tmp_target_imgs=list[p]
        # counter = start_num[p]
        tmp_noise_imgs=get_noisy_image(tmp_target_imgs.copy())
        for j in range(channel_num):
            add_noise_img = tmp_noise_imgs[j]
            target_img = tmp_target_imgs[j]
            target_img = target_img[np.newaxis, :, :]
            sobelx = cv2.Sobel(add_noise_img, -1, 1, 0, ksize=3)
            sobelx = sobelx[np.newaxis, :, :]
            sobely = cv2.Sobel(add_noise_img, -1, 0, 1, ksize=3)
            sobely = sobely[np.newaxis, :, :]
            sobel_spa = np.concatenate((sobelx, sobely), axis=0)
            # print(sobel_spa.shape)
            # 周围K+1波段光谱梯度
            add_noise_img = add_noise_img[np.newaxis, :, :]  # (1, h, w)，np.newaxis的作用就是在这一位置增加一个一维
            if j < K // 2:
                clean_vol = tmp_noise_imgs[0:K + 1, :, :]
                tar_im_25 = tmp_target_imgs[0:K + 1, :, :]
            elif j > channel_num - K // 2 - 1:
                clean_vol = tmp_noise_imgs[-K - 1:, :, :]
                tar_im_25 = tmp_target_imgs[-K - 1:, :, :]
            else:
                clean_vol = tmp_noise_imgs[j - K // 2:j + K // 2 + 1, :, :]
                tar_im_25 = tmp_target_imgs[j - K // 2:j+ K // 2 + 1, :, :]
            inp_im_25 = clean_vol.copy()
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
            sobel_spe=spe_gradient_1(clean_vol)
            # sobel_spe, _ = spe_gradient_11(clean_vol, j, K, channel_num)
            seq_crop_im,seq_crop_inp_im_grad_spa,seq_crop_inp_im_25,\
            seq_crop_inp_im_25_grad_spa,seq_crop_im_grad_spe,seq_crop_target_im,\
            seq_crop_tar_im_25=seq_crop(add_noise_img, sobel_spa, inp_im_25, sobelxy_25, sobel_spe, target_img, tar_im_25,patch_size)
            crop_num=len(seq_crop_im)
            for i in range(crop_num):
                sample = {
                    'input_im': seq_crop_im[i],  # 1
                    'input_im_grad_spa': seq_crop_inp_im_grad_spa[i],  # 2
                    'input_im_25': seq_crop_inp_im_25[i],  # 25
                    'input_im_25_grad_spa': seq_crop_inp_im_25_grad_spa[i],  # 50
                    'input_im_grad_spe': seq_crop_im_grad_spe[i],  # 24
                    'target_im': seq_crop_target_im[i],
                    'target_im_25': seq_crop_tar_im_25[i],
                }
                np.save(os.path.join(save_path,'{:07}.npy'.format(counter)),sample)
                txt.write('{:07}\n'.format(counter))
                print('{} th sample has been saved !'.format(counter))
                counter += 1

def data_deal_new(list,patch_size,save_path,txt_path):
    K=24
    op_num=len(list)   #4
    # print(op_num)
    # print(list[0].shape)
    channel_num=len(list[0])   #191
                    # input_im=np.zeros([op_num*crop_num*channel_num,1,patch_size,patch_size])
                    # input_im_grad_spa=np.zeros([op_num*crop_num*channel_num,2,patch_size,patch_size])
                    # input_im_25=np.zeros([op_num*crop_num*channel_num,25,patch_size,patch_size])
                    # input_im_25_grad_spa=np.zeros([op_num*crop_num*channel_num,50,patch_size,patch_size])
                    # input_im_grad_spe=np.zeros([op_num*crop_num*channel_num,24,patch_size,patch_size])
                    # target_im=np.zeros([op_num*crop_num*channel_num,1,patch_size,patch_size])
                    # target_im_25=np.zeros([op_num*crop_num*channel_num,25,patch_size,patch_size])
    txt=open(os.path.join(txt_path,'snr8_list.txt'),'w')
                    #start_c=open('start_num.txt','r')
                    #start_num=[np.int(t.split('\n')[0]) for t in start_c.readlines()]
    counter=0
    for p in range(op_num):   #p从0到4
    # for p in range(16):
        tmp_target_imgs=list[p]   #191*h*w
        seq_crop_target=seq_crop(tmp_target_imgs.copy(),patch_size)
        #seq_crop_mean=np.mean(np.mean(np.array(seq_crop_target)[:,0,:,:],1),1)
        #seq_sort=np.argsort(seq_crop_mean)
        # seq_len=len(seq_crop_mean)
        # cut_l=np.int(0.07*seq_len)
        # seq_crop_target=np.array(seq_crop_target)[seq_sort[cut_l:]]
        seq_crop_target = np.array(seq_crop_target)
        crop_num=len(seq_crop_target)  #crop_num=572
        #print(crop_num)
        for i in range(crop_num):
            tmp_crop_imgs=seq_crop_target[i]                        # 191*25*25
            tmp_clean_imgs =tmp_crop_imgs.copy()                    # 191*25*25
            tmp_noisy_imgs=get_noisy_image(tmp_crop_imgs.copy())

            tmp_trans_noisy = np.transpose(tmp_noisy_imgs.copy(), (1,2,0))  #25*25*191

            tmp_sobelx=cv2.Sobel(tmp_trans_noisy,-1, 1, 0, ksize=3)
            #print(tmp_sobelx)
            tmp_sobely=cv2.Sobel(tmp_trans_noisy, -1, 0, 1, ksize=3)
            tmp_sobelx= np.transpose(tmp_sobelx.copy(), (2,0,1))
            #print(tmp_sobelx)
            tmp_sobely = np.transpose(tmp_sobely.copy(), (2,0,1))

            tmp_spe_grad=tmp_noisy_imgs[1:]-tmp_noisy_imgs[:-1] #A[1:]取第二个到最后一个元素，a[:-1]除了最后一个取全部,190*25*25

            for j in range(channel_num):   #j从0到190
                cv2.imshow('clean', tmp_clean_imgs[j])
                cv2.imshow('noise',tmp_noisy_imgs[j])
                cv2.waitKey(1)
                seq_crop_im = tmp_noisy_imgs.copy()[j][np.newaxis, :]
                seq_crop_target_im = tmp_clean_imgs.copy()[j][np.newaxis, :]
                seq_crop_inp_im_grad_spa = np.array([tmp_sobelx[j], tmp_sobely[j]])
                #print('seq_crop_inp_im_grad_spa'+str(seq_crop_inp_im_grad_spa))
                if j < K // 2:
                    seq_crop_inp_im_25=tmp_noisy_imgs.copy()[0:K + 1]                                     #25
                    seq_crop_inp_im_25_grad_spa=np.concatenate([tmp_sobelx[0:K + 1],tmp_sobely[0:K + 1]]) #50
                    seq_crop_im_grad_spe=tmp_spe_grad[:K]                                                 #24
                    seq_crop_tar_im_25 = tmp_clean_imgs.copy()[0:K + 1, :, :]                             #target25
                    print('当前波段：' + str(j))
                    print('前后25波段：' + str(0)+'-'+str(K + 1))
                    print('前后25波段梯度24：' + str(0)+'-'+str(K))

                elif j > channel_num - K // 2 - 1:
                    seq_crop_inp_im_25 = tmp_noisy_imgs.copy()[ -K-1:]
                    seq_crop_inp_im_25_grad_spa = np.concatenate([tmp_sobelx[-K-1:], tmp_sobely[-K-1:]])
                    seq_crop_im_grad_spe = tmp_spe_grad[-K:]
                    seq_crop_tar_im_25 = tmp_clean_imgs.copy()[ -K-1:, :, :]
                    print('当前波段：' + str(j),)
                    print('前后25波段：' +str(-K-1)+'-'+str(-1))
                    print('前后25波段梯度24：' + str(-K) + '-' + str(-1))
                else:
                    seq_crop_inp_im_25 = tmp_noisy_imgs.copy()[ j-K//2:j+1+K//2]
                    seq_crop_inp_im_25_grad_spa = np.concatenate([tmp_sobelx[j-K//2:j+1+K//2], tmp_sobely[j-K//2:j+1+K//2]])
                    seq_crop_im_grad_spe = tmp_spe_grad[j-K//2:j+K//2]
                    seq_crop_tar_im_25 = tmp_clean_imgs.copy()[j-K//2:j+1+K//2, :, :]
                    print('当前波段：' + str(j), )
                    print('前后25波段：' + str(j-K//2) + '-' + str(j+1+K//2))
                    print('前后25波段梯度24：' + str(j-K//2) + '-' + str(j+K//2))

                sample = {
                            'input_im': seq_crop_im,  # 1
                            'input_im_grad_spa': seq_crop_inp_im_grad_spa,  # 2
                            'input_im_25': seq_crop_inp_im_25,  # 25
                            'input_im_25_grad_spa': seq_crop_inp_im_25_grad_spa,  # 50
                            'input_im_grad_spe': seq_crop_im_grad_spe,  # 24
                            'target_im': seq_crop_target_im,
                            'target_im_25': seq_crop_tar_im_25,
                        }
                np.save(os.path.join(save_path,'{:07}.npy'.format(counter)),sample)
                txt.write('{:07}\n'.format(counter))
                print('{} th sample has been saved !'.format(counter))
                print('-------------------------------------')
                counter += 1

            #
# class RandGaNoise(object):
#     def __init__(self, sigma):
#         self.sigma_ratio = sigma / 255.
#
#     def __call__(self, sample):
#         im = sample['input_im']
#         stddev_random = self.sigma_ratio * np.random.rand(1)  # 范围 stddev * (0 ~ 1)
#         noise = np.random.randn(*im.shape) * stddev_random
#         sample['input_im'] = im+noise
#
#         vol = sample['input_vol']
#         c, _, _ = vol.shape
#         noise = np.random.randn(*vol.shape)
#         for i in range(c):
#             stddev_random = self.sigma_ratio * np.random.rand(1)  # 范围 stddev * (0 ~ 1)
#             noise[i] = noise[i] * stddev_random
#         sample['input_vol'] = vol + noise
#
#         return sample

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample = {
            'input_im': torch.from_numpy(sample['input_im']).float(),
            'input_im_grad_spa': torch.from_numpy(sample['input_im_grad_spa']).float(),
            'input_im_25': torch.from_numpy(sample['input_im_25']).float(),
            'input_im_25_grad_spa': torch.from_numpy(sample['input_im_25_grad_spa']).float(),
            'input_im_grad_spe': torch.from_numpy(sample['input_im_grad_spe']).float(),
            'target_im': torch.from_numpy(sample['target_im']).float(),
            'target_im_25': torch.from_numpy(sample['target_im_25']).float(),
            # 'index':torch.tensor(sample['index']).float(),
            'index':torch.tensor(sample['index']).type(torch.long)
        }
        return sample

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)

def save_train(path, model, optimizer, epoch=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if epoch is not None:
        state['epoch'] = epoch
    # `_use_new_zipfile...=False` support pytorch version < 1.6
    torch.save(state, os.path.join(path, 'epoch_{}'.format(epoch)))
    return os.path.join(path, 'epoch_{}'.format(epoch))

def init_exps(exp_root_dir):
    if not os.path.exists(exp_root_dir):
        os.makedirs(exp_root_dir)
    all_exps = glob.glob(f'{exp_root_dir}/experiment*')

    cur_exp_id = None
    if len(all_exps) == 0:
        cur_exp_id = 0
    else:
        exp_ids = [int(os.path.basename(s).split('_')[1]) for s in all_exps]
        exp_ids.sort()
        cur_exp_id = exp_ids[-1] + 1

    log_dir = f'{exp_root_dir}/experiment_{cur_exp_id}'
    os.makedirs(log_dir)
    return log_dir

def cont_exps(exp_root_dir):
    if not os.path.exists(exp_root_dir):
        os.makedirs(exp_root_dir)
    all_exps = glob.glob(f'{exp_root_dir}/experiment*')
    cur_exp_id = None
    if len(all_exps) == 0:
        cur_exp_id = 0
    else:
        exp_ids = [int(os.path.basename(s).split('_')[1]) for s in all_exps]
        exp_ids.sort()
        cur_exp_id = exp_ids[-1]
    log_dir = f'{exp_root_dir}/experiment_{cur_exp_id}'
    return log_dir

def calc_psnr(im, recon, verbose=False):
    im = np.squeeze(im)
    recon = np.squeeze(recon)
    mse = np.sum((im - recon) ** 2) / np.prod(im.shape)
    #max_val =np.max(recon)
    max_val = 1.0
    psnr = 10 * np.log10(max_val ** 2 / mse)
    #psnr = 10 * np.log10(max_val/np.mean(np.mean(np.mean((im - recon) ** 2))))
    if verbose:
        print('PSNR %f'.format(psnr))
    return psnr

def calc_snr(clean,clean_addnoise):#必须得是干净在前，带噪图在后
    clean=np.squeeze(clean)
    clean_addnoise=np.squeeze(clean_addnoise)
    #print(clean.shape)
    h,w=clean.shape
    noise=clean_addnoise-clean
    P_clean=(np.sum(clean**2))/(h*w)#算方差
    P_noise=(np.sum(noise**2))/(h*w)
    snr=10*np.log10(P_clean/P_noise)
    return snr

def batch_PSNR(img, imclean):
    # Img = img.data.cpu().numpy().astype(np.float32)
    # Iclean = imclean.data.cpu().numpy().astype(np.float32)
    # PSNR = 0
    # for i in range(Img.shape[0]):
    #     PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :])
    # return (PSNR / Img.shape[0])
    Img = img.astype(np.float32)
    Iclean = imclean.astype(np.float32)
    PSNR = 0.0
    for i in range(Img.shape[0]):
        PSNR += calc_psnr(Iclean[i, :, :, :], Img[i, :, :, :])
    return (PSNR / Img.shape[0])

def batch_PSNR_list(img, imclean):
    # Img = img.data.cpu().numpy().astype(np.float32)
    # Iclean = imclean.data.cpu().numpy().astype(np.float32)
    # PSNR = 0
    # for i in range(Img.shape[0]):
    #     PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :])
    # return (PSNR / Img.shape[0])
    Img = img.astype(np.float32)
    Iclean = imclean.astype(np.float32)
    PSNR = []
    for i in range(Img.shape[0]):
        PSNR.append(calc_psnr(Iclean[i, :, :, :], Img[i, :, :, :]))
    return PSNR
    # PSNR = compare_psnr(img, imclean, data_range=data_range)
    # return PSNR

def calc_sam(true, pred):
    '''
    :param true: 给定   格式：CHW
    :param pred: 待测   格式：CHW
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    '''
    assert true.ndim ==3 and true.shape == pred.shape
    c,h,w=pred.shape
    sam_rad = np.zeros((h,w))
    #print(pred[0].shape)
    for x in range(h):
        for y in range(w):
            tmp_pred = pred[:,x, y].ravel()   #ravel()将数据拉成一维数组
            tmp_true = true[:,x, y].ravel()
            #sam_rad[x, y] = np.arccos(tmp_pred / (np.norm(tmp_pred) * tmp_true / np.norm(tmp_true)))
            sam_rad[x, y] = float(np.dot(tmp_true.T, tmp_pred) )/ (np.linalg.norm(tmp_true)* np.linalg.norm(tmp_pred))
            #print(sam_rad[x, y])
            eps = 1e-6
            if 1.0 < sam_rad[x, y] < 1.0 + eps:
                sam_rad[x, y] = 1.0
            elif -1.0 - eps < sam_rad[x, y] < -1.0:
                sam_rad[x, y] = -1.0
            sam_rad[x, y]=np.arccos(sam_rad[x, y])
    sam_deg = sam_rad.mean() * 180 / np.pi
    return sam_deg

class data_augmentation(object):
    def __init__(self):
        pass
    def __call__(self, mode, image):
        out = np.transpose(image, (1, 2, 0))  # CHW->HWC
        if mode == 0:
            # original                    原图
            out = out
        elif mode == 1:
            # flip up and down            上下翻转
            out = np.flipud(out)
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
        return np.transpose(out, (2, 0, 1))  # HWC->CHW

# 光谱梯度，后波段减前波段
def spe_gradient_1(x):
    spe_gra = []
    for i in range(np.shape(x)[0] - 1):
        gra = x[i + 1] - x[i]
        spe_gra.append(gra)
    spe_gra = np.array(spe_gra, dtype=np.float32)
    return spe_gra
# 光谱梯度，逐波段递减
def spe_gradient_2(x):#输入四维
    spe_gra1=[]
    for i in range(x.shape[0]):
        spe_gra2 = []
        for j in range(x.shape[1] - 1):
            gra = x[i, j + 1, :, :] - x[i, j, :, :]
            spe_gra2.append(gra)
        spe_gra2 = torch.tensor(np.array([item.cpu().detach().numpy() for item in spe_gra2]))#<class 'torch.Tensor'>,torch.Size([24, 25, 25])
        spe_gra1.append(spe_gra2)
    spe_gra= torch.tensor(np.array([item.cpu().detach().numpy() for item in spe_gra1])).cuda()
    return spe_gra

#光谱梯度，其余24波段减该波段
def spe_gradient_11(x,index,K,channels_num):
    spe_gra = []
    for i in range(np.shape(x)[0]):
        if index<K//2:
            gra = x[i] - x[index]
        elif index>channels_num-K//2-1:
            gra = x[i]-x[K-(channels_num-index)+1]
        else :
            gra = x[i] - x[K//2]
        spe_gra.append(gra)
    spe_gra = np.array(spe_gra, dtype=np.float32)
    spe_gra_dezero=spe_gra[~(spe_gra==0).all(axis=(1, 2))]
    return spe_gra_dezero,index

def spe_gradient_22(x,index,K,channels_num):#输入四维,index.shape16,
    spe_gra1=[]
    for i in range(x.shape[0]):
        spe_gra2 = []
        for j in range(x.shape[1]):
            if index[i] < (K // 2):
                gra = x[i,j] - x[i,index[i]]
            elif index[i] > (channels_num - K // 2 - 1):
                gra = x[i,j] - x[i,K-(channels_num - index[i]) + 1]
            else:
                gra = x[i,j] - x[i,K//2]
            spe_gra2.append(gra)
        spe_gra2 = np.array([item.cpu().detach().numpy() for item in spe_gra2])
        spe_gra3 = spe_gra2[~(spe_gra2 == 0).all(axis=(1, 2))]
        spe_gra3 = torch.tensor(spe_gra3)
        spe_gra1.append(spe_gra3)
    spe_gra = torch.tensor(np.array([item.cpu().detach().numpy() for item in spe_gra1])).cuda()
    return spe_gra

# 将空间图像及光谱图像裁剪成patch_size的形状。
#def rand_crop(input_im, input_im_grad_spa, input_im_grad_spe, target_im, target_im_grad_spe,patch_size):
def seq_crop(target_im,patch_size):
    _, y, x = target_im.shape
    x1 = np.array(np.arange(0,x-patch_size,patch_size))  # 产生[起点，终点，步长]之间的整数
    y1 =  np.array(np.arange(0,y-patch_size,patch_size))
    #print('x1,y1' + str(x1), y1)
    add_x=x-patch_size
    add_y=y-patch_size

    seq_x=x1.tolist()
    seq_x.append(add_x)
    seq_x=np.array(seq_x).astype(np.int)

    seq_y = y1.tolist()
    seq_y.append(add_y)
    seq_y = np.array(seq_y).astype(np.int)
    seq_crop_img=[]
    for tmp_x in seq_x:
        for tmp_y in seq_y:
            seq_crop_img.append(target_im[:,tmp_y:tmp_y+ patch_size, tmp_x:tmp_x + patch_size])
    # im = np.array([input_im[:,y1[i]:y1[i] + patch_size, x1[i]:x1[i] + patch_size] for i in range(crop_num)])
    # input_im_grad_spa = np.array([input_im_grad_spa[:,y1[i]:y1[i] + patch_size, x1[i]:x1[i] + patch_size] for i in range(crop_num)])
    # input_im_25 = np.array([input_im_25[:,y1[i]:y1[i] + patch_size, x1[i]:x1[i] + patch_size] for i in range(crop_num)])
    # input_im_25_grad_spa = np.array([input_im_25_grad_spa[:,y1[i]:y1[i] + patch_size, x1[i]:x1[i] + patch_size] for i in range(crop_num)])
    # input_im_grad_spe = np.array([input_im_grad_spe[:,y1[i]:y1[i] + patch_size, x1[i]:x1[i] + patch_size] for i in range(crop_num)])
    # target_im = np.array([target_im[:,y1[i]:y1[i] + patch_size, x1[i]:x1[i] + patch_size] for i in range(crop_num)])
    # target_im_25= np.array([target_im_25[:,y1[i]:y1[i] + patch_size, x1[i]:x1[i] + patch_size] for i in range(crop_num)])
    return seq_crop_img

def rand_crop_ori(input_im, input_im_grad_spa,input_im_25,input_im_25_grad_spa, input_im_grad_spe, target_im,target_im_25,patch_size):
    _, y, x = input_im.shape  # 单个独立下划线是用作一个名字，来表示某个变量是临时的或无关紧要.在这行代码中，"_"作为占位符变量可以派上用场
    x1 = random.randint(0, x - patch_size)  # W,随机产生[0,x-patch_size]之间的整数
    y1 = random.randint(0, y - patch_size)  # H
    im = input_im[:, y1:y1 + patch_size, x1:x1 + patch_size].copy()
    input_im_grad_spa = input_im_grad_spa[:, y1:y1 + patch_size, x1:x1 + patch_size].copy()
    input_im_25 = input_im_25[:, y1:y1 + patch_size, x1:x1 + patch_size].copy()
    input_im_25_grad_spa = input_im_25_grad_spa[:, y1:y1 + patch_size, x1:x1 + patch_size].copy()
    input_im_grad_spe = input_im_grad_spe[:, y1:y1 + patch_size, x1:x1 + patch_size].copy()
    target_im = target_im[:, y1:y1 + patch_size, x1:x1 + patch_size].copy()
    target_im_25= target_im_25[:, y1:y1 + patch_size, x1:x1 + patch_size].copy()
    return im, input_im_grad_spa,input_im_25,input_im_25_grad_spa ,input_im_grad_spe, target_im,target_im_25
# def get_patches(input_im, sobel_spa,sobel_spe,target_im,patch_size,stride):
#     _, h, w = input_im.shape
#     for i in range(0, h-patch_size+1, stride):
#         input_im = input_im[:, i:i + patch_size, i:i + patch_size].copy()
#         sobel_spa = sobel_spa[:, i:i + patch_size, i:i + patch_size].copy()
#         sobel_spe= sobel_spe[:, i:i + patch_size, i:i + patch_size].copy()
#         target_im = target_im[:, i:i + patch_size, i:i + patch_size].copy()
#     return input_im,sobel_spa,sobel_spe,target_im

# class RandGaNoise(object):
#     def __init__(self, orig_img,sigma):
#         self.orig_img=orig_img
#         self.sigma_ratio = sigma / 255.
#
#     def __call__(self):
#         c, _, _ = self.orig_img.shape
#         noise = np.random.randn(*self.orig_img.shape)#从标准正态分布中返回指定维度的矩阵
#         for i in range(c):
#             stddev_random = self.sigma_ratio * np.random.rand(1)  # 范围 stddev * [0 , 1)
#             noise[i] = noise[i] * stddev_random
#         vol_gaussian= self.orig_img+ noise
#         return vol_gaussian

def get_noisy_image(orig):
    # 添加高斯噪声
    c, h, w = orig.shape
    # sigma_signal = np.zeros([1, c])
    # for i in range(c):
    #     sigma_signal[0, i] += np.sum(orig[i] ** 2)
    # sigma_signal = sigma_signal / (h * w)
    # SNR = 30 + np.random.rand(1, c) * 10  # numpy.ndarray,1行c列
    # SNR1 = np.power(10., SNR / 10)  # 1行c列，值在10的1-2次方之间
    # sigma_noi = sigma_signal / SNR1  # 所需要的噪音强度
    # d = np.random.randn(h, w)
    # noise_signal = np.sum(abs(d) ** 2) / (h * w)
    # for i in range(c):
    #     noise = np.sqrt(sigma_noi[0][i] / noise_signal) * d
    #     orig[i, :, :] = orig[i, :, :] + noise

    #加高斯噪声
    # for i in range(c):
    #     max=np.max(orig[i])
    #     min=np.min(orig[i])
    #     sigma = (10 / 255)*(max-min)
    #     #sigma = (5/ 255) * (max - min)
    #     noise = np.random.normal(0, sigma, (h, w))
    #     orig[i] = orig[i] + noise
    # sigma = 10/255
    # noise = np.random.normal(0, sigma, (c,h, w))
    # orig= orig+ noise


    #加条带噪声
    #总波段选取10个波段，添加条带，随机选取一些行，一半按均值增大，一半按均值减小
    #band=np.random.randint(0,c,10)
    # band=np.arange(0,c)
    # random.shuffle(band)
    # band=band[0:10]
    # stripnum = np.random.randint(1, w-10, len(band))  # 每个想加条带的波段具体要加条带的数量,不允许为0
    # #print('band:'+str(band),'stripnum:'+str(stripnum))
    # for i in range(band.shape[0]):
    #     loc=np.random.randint(0,w, stripnum[i])
    #     mean = 0.0
    #     for j in range(loc.shape[0]):
    #         mean += np.mean(orig[band[i], :, loc[j]])
    #     mean /= loc.shape[0]
    #     if loc.shape[0] % 2 == 0:  #若该波段条带数为偶数
    #         for j in range(loc.shape[0] // 2):
    #             orig[band[i], :, loc[j]] = orig[band[i], :, loc[j]] - mean
    #         for j in range(loc.shape[0] // 2, loc.shape[0]):
    #             orig[band[i], :, loc[j]] = orig[band[i], :, loc[j]] + mean
    #     else:  # 若该波段条带数为奇数
    #         for j in range((loc.shape[0] + 1) // 2):
    #             orig[band[i], :, loc[j]] = orig[band[i], :, loc[j]] - mean
    #         for j in range((loc.shape[0] + 1) // 2, loc.shape[0]):
    #             orig[band[i], :, loc[j]] = orig[band[i], :, loc[j]] + mean

    #全波段添加条带，随机选取一些行，一半按均值增大，一半按均值减小
    small=np.random.randint(95,100,c//3)
    big=np.random.randint(100,105,c//3)
    # small = np.random.randint(w//2-5,w//2,5)
    # big = np.random.randint(w//2,w//2+5, 5)
    # #background = np.ones(c-len(small)-len(big)) * np.ceil(w//100)
    background = np.ones(c - len(small) - len(big)) * np.ceil(2)
    stripnum=(np.concatenate((small,big,background),axis=0)).astype(np.uint8)
    random.shuffle(stripnum)
    #stripnum = np.random.randint(1,w,c) #每个想加条带的波段具体要加条带的数量,不允许为0
    #print('stripnum='+str(stripnum),stripnum.shape[0])
    for i in range(c):
        #loc = (np.ceil((w - 10) * np.random.rand(1, stripnum[0][i]))).astype(int)  # 这10波段中，每个波段具体是哪些列要做条带处理,值在0~(w-10)向上取整
        loc = np.random.randint(0, w, stripnum[i])#stripnum[i]个[0,w)之间的值
        #print('loc:'+str(loc),loc.shape)
        mean=0.0
        for j in range(loc.shape[0]):
            mean+=np.mean(orig[i, :, loc[j]])
        mean/=loc.shape[0]
        #print(mean)
        if loc.shape[0]==1:
            orig[i,:,loc[0]]=orig[i,:,loc[0]]+mean/2
        elif loc.shape[0]%2==0:#若该波段条带数为偶数
            for j in range(loc.shape[0]//2):
                orig[i, :, loc[j]] = orig[i, :, loc[j]] - mean
            for j in range(loc.shape[0]//2,loc.shape[0]):
                orig[i, :, loc[j]] = orig[i, :, loc[j]] + mean
        else:#若该波段条带数为奇数
            for j in range((loc.shape[0]+1)//2):
                orig[i, :, loc[j]] = orig[i, :, loc[j]] - mean
            for j in range((loc.shape[0]+1)//2,loc.shape[0]):
                orig[i, :, loc[j]] = orig[i, :, loc[j]] + mean

    # #随机波段有噪声，强度不同
    # band = (np.ceil((c - 10) * np.random.rand(1, c))).astype(int)
    # stripnum = (np.ceil((w - 10) * np.random.rand(1, w))).astype(int) #每个想加条带的波段具体要加条带的数量
    # for i in range(band.shape[1]):
    #     loc = (np.ceil((w - 10) * np.random.rand(1, stripnum[0][i]))).astype(int)  # 这c波段中，每个波段具体是哪些列要做条带处理,值在0~(w-10)向上取整
    #     t = np.random.rand(1, loc.shape[1]) * 0.05 - 0.15  # 要减去的值，1行loc列
    #     for j in range(loc.shape[1]):
    #         orig[band[0, i], :, loc[0, j]] = orig[band[0, i], :, loc[0, j]] - t[0, j]

    #全波段有噪声，强度不同
    # stripnum = (np.ceil((w - 10) * np.random.rand(1, c))).astype(int)  # 1行w列，值在0~（w-10）向上取整，即1~（w-10）+1之间
    # for i in range(c):
    #     loc = (np.ceil((w - 10) * np.random.rand(1, stripnum[0][i]))).astype(int)  # 这c波段中，每个波段具体是哪些列要做条带处理,值在0~(w-10)向上取整
    #     t = np.random.rand(1, loc.shape[1]) * 0.05 - 0.15  # 要减去的值，1行loc列
    #     for j in range(loc.shape[1]):
    #         orig[i, :, loc[0, j]] = orig[i, :, loc[0, j]] - t[0, j]

    #全波段有噪声，强度相同
    # stripnum = np.random.randint(1, w)
    # loc = (np.ceil((w - 10) * np.random.rand(1, stripnum))).astype(int)  # 这c波段中，每个波段具体是哪些列要做条带处理,值在0~(w-10)向上取整
    # t = np.random.rand(1, loc.shape[1]) * 0.05 - 0.15  # 要减去的值，1行loc列
    # for i in range(c):
    #     for j in range(loc.shape[1]):
    #         orig[i, :, loc[0, j]] = orig[i, :, loc[0, j]] - t[0, j]

    #奇波段有噪声，偶波段无，强度相同
    # band=np.arange(0,c+1,2)      #即[0,c],间隔为2，即1，3，5...波段
    # stripnum = np.random.randint(1, w)
    # loc=(np.ceil((w - 10) * np.random.rand(1, stripnum))).astype(int)  # 这c波段中，每个波段具体是哪些列要做条带处理,值在0~(w-10)向上取整
    # t = np.random.rand(1, loc.shape[1]) * 0.05 - 0.15 #要减去的值，1行loc列
    # for i in range(band.shape[0]):
    #     for j in range(loc.shape[1]):
    #         orig[band[i],:,loc[0,j]]= orig[band[i],:, loc[0,j]]-t[0,j]

    #orig,max2,min2= scaler2(orig)

    # 添加死线，对每个波段，随机挑选num_stripe行，将值改为0
    # band_dead_tmp = np.arange(0, c)
    # random.shuffle(band_dead_tmp)
    # band_dead = band_dead_tmp[0:c*2//3]
    # # print('band_dead:'+str(band_dead),type(band_dead))
    # for i in range(len(band_dead)):
    #     deadline_num = np.arange(0, h)
    #     random.shuffle(deadline_num)
    #     deadline_num = deadline_num[0:h//10]
    #     # print(deadline_num)
    #     # rand_deadline = np.random.randint(orig.shape[1], size=30)  #size=条带噪声数量
    #     for j in range(len(deadline_num)):
    #         orig[band_dead[i], deadline_num[j], :] = 0
    return orig

def scaler_mean_all(x):
    mean = np.mean(np.mean(x,1),1)
    std=np.array([np.std(x[i]-mean[i]) for i in range(len(mean))])
    y=np.array([0.5+(x[i]-mean[i])/(6*std[i]) for i in range(len(mean))])
    # x_l=len(x)
    # cut_l=np.int(x_l*0.025)
    # sort_seq=np.argsort(mean)
    # sort_seq=sort_seq[cut_l:-cut_l]
    # y=y[sort_seq,:]
    y=np.clip(y,0,1)
    return y

def scaler_max_all(x):
    tmp_max=np.max(np.max(x,1),1)
    tmp_min=np.min(np.min(x,1),1)
    #print('scaler_max_all',tmp_max,tmp_min)
    y=np.array([(x[i]-tmp_min[i])/(tmp_max[i]-tmp_min[i]) for i in range(len(tmp_max))])
    return y

def scaler_max_all_addnoise(x,addnoise):
    tmp_max=np.max(np.max(addnoise,1),1)
    tmp_min=np.min(np.min(addnoise,1),1)
    #print('scaler_max_all_addnoise',tmp_max,tmp_min)
    y=np.array([(x[i]-tmp_min[i])/(tmp_max[i]-tmp_min[i]) for i in range(len(tmp_max))])
    return y
def scaler_mean(x):
    c,h,w=np.shape(x)
    y=[]
    for i in range(c):
        tmp=aver_normlize(x[i])
        y.append(tmp)
        #print(np.max(tmp), np.min(tmp), len(np.where(tmp < 0)[0]), len(np.where(tmp > 1)[0]))
    y=np.array(y)
    return np.clip(y,0,1)

def aver_normlize(x):
    mean = np.mean(x)
    var = np.mean(np.square(x - mean))
    y = (x - mean) / (6 * np.sqrt(var)) + 0.5
    return np.clip(y,0,1)
# 归一化
def scaler(x):
    # mean=np.mean(x)
    # dis=np.max(np.abs(x-mean))
    # x=0.5+(x-mean)/(2*dis)

    c,h,w=x.shape
    for i in range(c):
        max = np.max(x[i])
        min = np.min(x[i])
        x[i]=(x[i]-min)/(max-min)


    return x

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

def back_scaler(x,max,min):
    y=[]
    for i in range(x.shape[0]):
        back_scaler=x[i]*(max-min)+min
        y.append(back_scaler)
    y = np.array(y, dtype=np.float32)
    return y

#loss=l_spa+l_spe
def cal_loss1 (out_noise_img,input_noise_img,target_img,out_noise_clean_25_grad_spe,input_im_grad_spe,alpha):
    a=out_noise_img
    b=input_noise_img-target_img
    c=out_noise_clean_25_grad_spe
    d=input_im_grad_spe
    l_spa = 0.0
    l_spe = 0.0
    for i in range(a.shape[0]):
        l1 = (torch.norm(a[i][0] - b[i][0])) ** 2
        l_spa += l1
        l_spe_24 = 0.0
        for j in range(c.shape[1]):
            l2 = (torch.norm(c[i][j] - d[i][j])) ** 2
            l_spe_24 += l2
        l_spe += l_spe_24
    l_spa /= 2 * a.shape[0]
    l_spe /= 2 * a.shape[0]
    loss=(1-alpha)*l_spa+alpha*l_spe
    return loss
#loss=l_spa
def cal_loss2 (out_noise_img,input_noise_img,target_img,out_noise_clean_25_grad_spe,input_im_grad_spe,alpha):
    a=out_noise_img
    b=input_noise_img-target_img
    c=out_noise_clean_25_grad_spe
    d=input_im_grad_spe
    l_spa = 0.0
    l_spe = 0.0
    for i in range(a.shape[0]):
        l1 = (torch.norm(a[i][0] - b[i][0])) ** 2
        l_spa += l1
    l_spa /= 2 * a.shape[0]
    loss=l_spa
    return loss

#loss=l_spa+l_spe+l_denoise
def cal_loss3 (out_noise_img,input_noise_img,target_img,out_noise_clean_25_grad_spe,input_im_grad_spe,alpha):
    a=out_noise_img
    b=input_noise_img-target_img
    c=out_noise_clean_25_grad_spe
    d=input_im_grad_spe
    l_spa = 0.0
    l_spe = 0.0
    l_c=0.0
    for i in range(a.shape[0]):
        l1 = (torch.norm(a[i][0] - b[i][0])) ** 2
        l_spa += l1
        l_spe_24 = 0.0
        for j in range(c.shape[1]):
            l2 = (torch.norm(c[i][j] - d[i][j])) ** 2
            l_spe_24 += l2
        l_spe += l_spe_24
    for i in range(a.shape[0]):
        l3 = (torch.norm(c[i][0])) ** 2
        l_c += l3
    l_spa /= 2 * a.shape[0]
    l_spe /= 2 * a.shape[0]
    l_c /= 2 * a.shape[0]
    loss=(1-alpha)*l_spa+alpha/2*l_spe+alpha/2*l_c
    return loss

def load_txt(txt_path):
    # cont=open(txt_path,'r')
    # t=cont.readlines()
    # txt_seq=[t_.split('\n')[0] for t_ in t]
    txt_seq=np.loadtxt(txt_path)
    return txt_seq

# def data_augmentation(image, mode):
#     out = np.transpose(image, (1, 2, 0))#CHW->HWC
#     if mode == 0:
#         # original                    原图
#         out = out
#     elif mode == 1:
#         # flip up and down            上下翻转
#         out = np.flipud(out)
#     elif mode == 2:
#         # rotate counterwise 90 degree 逆向旋转90°
#         out = np.rot90(out)
#     elif mode == 3:
#         # rotate 90 degree and flip up and down  翻转90°并上下翻转
#         out = np.rot90(out)
#         out = np.flipud(out)
#     elif mode == 4:
#         # rotate 180 degree             旋转180°
#         out = np.rot90(out, k=2)
#     elif mode == 5:
#         # rotate 180 degree and flip    旋转180°并反转
#         out = np.rot90(out, k=2)
#         out = np.flipud(out)
#     elif mode == 6:
#         # rotate 270 degree            旋转270°
#         out = np.rot90(out, k=3)
#     elif mode == 7:
#         # rotate 270 degree and flip    旋转270°并翻转
#         out = np.rot90(out, k=3)
#         out = np.flipud(out)
#     return np.transpose(out, (2, 0, 1))#HWC->CHW


if __name__ == '__main__':
    path = r'E:\data\adjust\clean.tif'
    img = gdal.Open(path)
    data = img.ReadAsArray()#chw

    #保存为mat
    scio.savemat(r'E:\data\adjust\clean.mat', {'img': data})
    print(data.shape)





