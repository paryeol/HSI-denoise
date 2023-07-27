import csv
import os
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
from cv2 import rotate
from matplotlib import pyplot as plt
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from tqdm import tqdm
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from network_SSGN import SSGN
from dataset_dc import TrainData_split
from utils import *
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
print(torch.cuda.is_available())
DEVICE = torch.device('cuda:0')

TRAIN_CFG = {
    "GT_crop": {
        "epoch": 300,
        "batch_size": 4,
        "learning_rate": 0.0001,
        "train_dir": r"D:\learn_pytorch\data\GT_crop_train_new.mat",
        "test_dir": r"D:\learn_pytorch\data\GT_crop_test_new.mat",
        # 'txt_path': r'F:\lian\SSGN_orig\train_list.txt',
        # 'data_path': r'F:\read_data',
        'txt_path': r'D:\learn_pytorch\SSGN_orig\list_test.txt',
        'data_path': r'E:\read_test',
        "log_dir": "saved_models/GT_crop",
        'patch_size':25,
        'K':24,   #K个临近波段
        'num_deadline':5 , #每个波段死线数量
        'alpha':0.001,
        'shuzu':np.arange(50,300,50)
    }
}

def get_train_val_loaders(args):
    print('Loading dataset...')
    txt_path=args['txt_path']
    data_path=args['data_path']
    index_list = np.array(load_txt(txt_path)).astype(np.int)
    # tmp_seq=np.arange(0,len(self.index_list))
    tmp_seq = np.arange(0, 32)
    print(len(index_list))
    # random.shuffle(tmp_seq)
    # self.rand_list=tmp_seq[0:2000]
    rand_list = np.random.choice(tmp_seq, size=16, replace=False, p=None)  # 生成挑选出来的size个编号
    remaining_list = np.delete(np.arange(tmp_seq.shape[0]),rand_list)  # 从原数组下标中删除已经选中的下标数组
    tmp_seq = tmp_seq[remaining_list]
    print('rand_list:' + str(rand_list))
    train_dataset = TrainData_split(index_list,data_path,ToTensor(),tmp_seq,rand_list)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    #print('batch_size='+str(args['batch_size']))
    return train_loader

class Trainer:
    def __init__(self,con_tra=False):
        self.con_tra=con_tra
        self.init_epoch=0
        train_cfg = TRAIN_CFG['GT_crop']
        #print(train_cfg['shuzu'])
        # init experiments
        base_name= f'pretrained'
        if self.con_tra==False:
            log_dir = init_exps(os.path.join(train_cfg['log_dir'], base_name))
        else:
            log_dir = cont_exps(os.path.join(train_cfg['log_dir'], base_name))
        train_cfg['log_dir'] = log_dir
        # save_model_path=train_cfg['save_model_path']
        # record parameter
        with open(os.path.join(log_dir, 'params.yaml'), 'w') as f:
            yaml.dump(train_cfg, f, default_flow_style=False, allow_unicode=True)
        logger = open(os.path.join(log_dir,'logger.txt'),'a+')
        # build model
        model = SSGN(train_cfg['K'])#24通道
        model.apply(weights_init_kaiming)
        optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])   #优化器
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['epoch'],eta_min=1e-05)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=train_cfg['shuzu'],gamma=0.8,last_epoch=-1)#学习速率每10epoch*0.5
        if con_tra==True:
            weights_dir=os.path.join(log_dir,'models')
            weights_name=os.listdir(weights_dir)
            if len(weights_name)!=0:
                num=[]
                for name in weights_name:
                    num.append(np.int(name.split('epoch_')[-1]))
                max_num=max(np.array(num))
                self.init_epoch=max_num+1
                model.load_state_dict(torch.load(os.path.join(weights_dir, 'epoch_{}'.format(max_num)))['model'])
                optimizer.load_state_dict(torch.load(os.path.join(weights_dir,'epoch_{}'.format(max_num)))['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                lr=optimizer.state_dict()['param_groups'][0]['lr']
                print('加载 epoch {} 成功,当前lr={:.5f}'.format(max_num,lr))

        model = model.to(DEVICE)

        # train
        self.train(model, train_cfg, logger,optimizer,scheduler)
        #self.train(model, train_cfg, logger, optimizer)

    #def train(self, model, args, logger,optimizer):
    def train(self, model, args, logger, optimizer,scheduler):
        optimizer = optimizer
        scheduler = scheduler
        model_save_dir = os.path.join(args['log_dir'], 'models')
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        print('Start training...'+ TIMESTAMP, file=logger, flush=True)

        writer = SummaryWriter(os.path.join(args['log_dir'],'loss_and_psnr')+TIMESTAMP)
        max_psnr=30.0
        for epoch in range(args['epoch']):
            current_lr=optimizer.state_dict()['param_groups'][0]['lr']
            if epoch<self.init_epoch:
                continue
            train_loader = get_train_val_loaders(args)
            train_bar = tqdm(total=len(train_loader),bar_format="{l_bar}{bar:30}{r_bar}",ncols=80,position=0)  # tqmd进度条可视化
            train_bar.set_description(f"[{epoch}/{args['epoch']-1}]")
            model.train()
            train_loss = 0.0
            train_psnr = 0.0
            #print('len_train_loader:'+str(len(train_loader)))
            for i,sample in enumerate(train_loader):
                #print('i:'+str(i))
                input_im, input_im_grad_spa,input_im_grad_spe, target_im = sample['input_im'].to(DEVICE), sample['input_im_grad_spa'].to(DEVICE), sample['input_im_grad_spe'].to(DEVICE),sample['target_im'].to(DEVICE)
                input_im_25,input_im_25_grad_spa,target_im_25=sample['input_im_25'].to(DEVICE),sample['input_im_25_grad_spa'].to(DEVICE),sample['target_im_25'].to(DEVICE)
                index=sample['index'].to(DEVICE)
                print('train的index是:'+str(index))
                # forward & backward
                optimizer.zero_grad()  #将梯度归零
                out_noise_img,out_noise_img_25= model(input_im,input_im_grad_spa,input_im_25,input_im_25_grad_spa,input_im_grad_spe)
                #[16, 1, 25, 25],[16, 25, 25, 25]
                denoise_img = input_im - out_noise_img           #[16, 1, 25, 25]
                #print('denoise_img:'+str(denoise_img))
                out_noise_clean_25=target_im_25+out_noise_img_25     #[16, 25, 25, 25]
                denoise_grad_spe=spe_gradient_2(out_noise_clean_25)     #[16, 24, 25, 25]
                K=24
                K=torch.tensor(K).to(DEVICE).type(torch.long)
                channels_num=191
                channels_num= torch.tensor(channels_num).to(DEVICE).type(torch.long)
                #out_noise_clean_25_grad_spe = spe_gradient_22(out_noise_clean_25,index,K,channels_num)                                          #x,index,K
                #loss=cal_loss1(out_noise_img,input_im,target_im,denoise_grad_spe,input_im_grad_spe,args['alpha'])                  #l_spa+l_spe
                criterion=nn.MSELoss()
                alpha=0.001
                loss=(1-alpha)*1000*criterion(out_noise_img,input_im-target_im)+alpha*1000*criterion(denoise_grad_spe,input_im_grad_spe)
                #loss = cal_loss2(out_noise_img, input_im, target_im, out_noise_clean_25_grad_spe, input_im_grad_spe,args['alpha'])            #l_spa
                #loss = cal_loss3(out_noise_img, input_im, target_im, denoise_grad_spe, input_im_grad_spe,args['alpha'])  #l_spa+l_spe+l_denoise
                loss.backward()#反向传播得到每个参数的梯度
                if loss != loss:
                    raise Exception('NaN in loss, crack!')
                optimizer.step()#梯度下降，参数更新
                # record training info
                train_loss += loss.item()
                out_noise_img=out_noise_img.detach().cpu().numpy()
                input_im=input_im.detach().cpu().numpy()
                denoise_img = denoise_img.detach().cpu().numpy()#阻止反向传播，转到cpu上
                denoise_img=np.clip(denoise_img,0,1)
                #print(denoise_img.shape)
                target_im = target_im.detach().cpu().numpy()
                #print(np.max(target_im), np.max(denoise_img))
                temp_psnr=batch_PSNR_list(target_im, denoise_img)
                #print('{} th batch_psnr_max={:.3f} batch_psnr_min={:.3f} batch_psnr_mean={:.3f}'.format(i,np.max(temp_psnr),np.min(temp_psnr),np.mean(temp_psnr)))
                train_psnr += np.sum(temp_psnr)
                train_bar.update(1)
            train_bar.reset()
            scheduler.step()
            #print('train_loader长度:'+str(len(train_loader)))
            train_psnr /= (len(train_loader)*args['batch_size'])
            train_loss /= (len(train_loader)*args['batch_size'])
            # writer.add_scalar('loss/epoch', train_loss, epoch)  # 标题，y轴，x轴
            writer.add_scalars('Loss/train'+TIMESTAMP,  {"loss": train_loss}, epoch)
            writer.add_scalars('PSNR/train'+TIMESTAMP, {"psnr": train_psnr}, epoch)
            writer.add_scalars('lr/train' + TIMESTAMP, {"lr": current_lr}, epoch)
            # val_loss, val_psnr = self.valid(model, val_loader, val_bar)
            # if (epoch + 1) % 10 == 0:
            #     save_train(model_save_dir, model, optimizer,epoch=epoch)#保存模型，参数，以及学习率

            # if train_psnr>max_psnr:
            #     print('train_psnr='+str(train_psnr),'max_psnr='+str(max_psnr))
            #     max_psnr=train_psnr
            #     save_train(model_save_dir, model, optimizer, epoch=epoch)  # 保存模型，参数，以及学习率
            #     test_psnr=Tester(model_save_dir,model)
            # else:
            #     test_psnr=0

            save_train(model_save_dir, model, optimizer, epoch=epoch)  # 保存模型，参数，以及学习率
            # test_psnr, test_ssim,test_sam,pre_noise,imgs= Tester(model)
            # writer.add_scalars('test_psnr' + TIMESTAMP, {"test_psnr": test_psnr,"psnr": train_psnr}, epoch)
            # writer.add_scalars('test_ssim' + TIMESTAMP, {"test_ssim": test_ssim}, epoch)
            # writer.add_scalars('test_sam' + TIMESTAMP, {"test_sam": test_sam}, epoch)
            # print('[{}/{}] | train_loss: {:.5f} | train_psnr: {:.3f}| lr: {:.5f}| test_psnr: {:.3f}| test_ssim: {:.3f}| test_sam: {:.3f}'
            #         .format(epoch, args['epoch']-1, train_loss, train_psnr,current_lr,test_psnr,test_ssim,test_sam), file=logger, flush=True)
            # print('[{}/{}] | train_loss: {:.5f} | train_psnr: {:.3f}| lr: {:.5f}| test_psnr: {:.3f}| test_ssim: {:.3f}| test_sam: {:.3f}'
            #       .format(epoch, args['epoch'] - 1, train_loss, train_psnr,current_lr,test_psnr,test_ssim,test_sam))

def get_data(imgs,index,channels_num):
    K=24
    img = imgs[index]
    sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
    sobelx = sobelx[np.newaxis, :, :]
    sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)
    sobely = sobely[np.newaxis, :, :]
    #2
    sobel_spa = np.concatenate((sobelx, sobely), axis=0)

    if index < K // 2:
        clean_vol = imgs[0:K + 1, :, :]
    elif index > channels_num//2 - 1:
        clean_vol = imgs[-K - 1:, :, :]
    else:
        clean_vol = imgs[index - K // 2:index + K // 2 + 1, :, :]
    #25
    input_im_25 = clean_vol.copy()
    #24
    sobel_spe = spe_gradient_1(clean_vol)
    #sobel_spe,index = spe_gradient_11(clean_vol,index,K,channels_num)

    #print(sobel_spe.shape)
    sobelxy_25 = []
    for i in range(input_im_25.shape[0]):
        sobelx1 = cv2.Sobel(input_im_25[i], -1, 1, 0, ksize=3)
        sobelx1 = sobelx1[np.newaxis, :, :]
        sobely1 = cv2.Sobel(input_im_25[i], -1, 0, 1, ksize=3)
        sobely1 = sobely1[np.newaxis, :, :]
        sobelxy_ = np.concatenate((sobelx1, sobely1), axis=0)
        sobelxy_25.append(sobelxy_)
    sobelxy_25 = np.array(sobelxy_25, dtype=np.float32)
    b, c, h, w = sobelxy_25.shape
    #50
    sobelxy_25 = sobelxy_25.reshape((b * c, h, w))

    noise_im = imgs[index]
    noise_im = noise_im[np.newaxis, :, :]

    return noise_im,sobel_spa,input_im_25,sobelxy_25,sobel_spe

def Tester(model):
    clean=scio.loadmat(TRAIN_CFG['GT_crop']['test_dir'])['img']
    #target=clean.copy()
    orig=get_noisy_image(clean.copy())
    orig_scaler = scaler_max_all(orig)                 #加噪后归一化
    target=scaler_max_all_addnoise(clean.copy(),orig)  #target按照假造后的每波段最大最小值归一化

    imgs = []
    pre_noise=[]
    model.eval()
    with torch.no_grad():
        psnr=0.0
        for i in range(np.shape(orig)[0]):
            input_im1, sobel_spa2,input_im_25,sobelxy_50,sobel_spe24= get_data(orig_scaler, i, np.shape(orig)[0])
            input_im1 = torch.from_numpy(input_im1[np.newaxis,:]).float().to(DEVICE)#转化为tensor
            sobel_spa2 =  torch.from_numpy(sobel_spa2[np.newaxis]).float().to(DEVICE)
            input_im_25= torch.from_numpy(input_im_25[np.newaxis, :]).float().to(DEVICE)
            sobelxy_50 = torch.from_numpy(sobelxy_50[np.newaxis, :]).float().to(DEVICE)
            sobel_spe24 = torch.from_numpy(sobel_spe24[np.newaxis, :]).float().to(DEVICE)

            noise_res,out_noise_img_25 = model(input_im1, sobel_spa2,input_im_25,sobelxy_50,sobel_spe24)   #残差输出
            denoise_img = input_im1 - noise_res       #去噪后的波段
            denoise_img = denoise_img.detach().cpu().squeeze().numpy()
            denoise_img =np.clip(denoise_img,0,1)
            imgs.append(denoise_img.copy())
            noise_res = noise_res.detach().cpu().squeeze().numpy()
            pre_noise.append(noise_res.copy())
            psnr+=calc_psnr(target[i],denoise_img)
            print('第{}个通道正在降噪'.format(i),psnr)
        imgs = np.array(imgs, dtype=np.float32)
        pre_noise = np.array(pre_noise, dtype=np.float32)
        psnr/=np.shape(orig)[0]
        ssim=compare_ssim(target,imgs)
        sam=calc_sam(target,imgs)
        scio.savemat(os.path.join(TRAIN_CFG['GT_crop']['log_dir'],'denoise.mat'),{'img':imgs})
        scio.savemat(os.path.join(TRAIN_CFG['GT_crop']['log_dir'], 'pre_noise.mat'), {'img': pre_noise})
    return psnr,ssim,sam,pre_noise,imgs

def getSpanTimes(lngTimeSpan, unit=''):
    unit_sim = 'Y |M |D |:|:|;'
    unit_en = ' Years | Months | Days | Hour | Minute | Second'
    unit_zh = '年 |个月 |天 |小时 |分 |秒'
    if unit == 'en':
        unit_list = unit_en.split('|')
    elif unit == 'zh':
        unit_list = unit_zh.split('|')
    else:
        unit_list = unit_sim.split('|')
    t = time.gmtime(float(round(lngTimeSpan, 3)))
    # print(t)
    total_time = ''
    total_time += ('%d%s' % (t.tm_year - 1970, unit_list[0])) if t.tm_year > 1970 else ''
    total_time += ('%d%s' % (t.tm_mon - 1, unit_list[1])) if t.tm_mon > 1 else ''
    total_time += ('%d%s' % (t.tm_mday - 1, unit_list[2])) if t.tm_mday > 1 else ''
    tm = "%%H%s%%M%s%%S%s" % tuple(unit_list[3:])
    total_time += time.strftime(tm, t)
    return total_time

if __name__ == '__main__':
    start_time=time.time()
    Trainer(con_tra=True)
    end_time=time.time()
    train_time=end_time-start_time
    total_time=getSpanTimes (train_time, unit='zh')
    print('Running time:{}'.format(total_time))


