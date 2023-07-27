import time
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import scipy.io as scio
from utils import *
from network_SSGN import SSGN
print(torch.cuda.is_available())
DEVICE = torch.device('cuda')
TEST_CFG = {
    "GF": {
        'K':24,
        "orig_dir": r"D:\learn_pytorch\data\real_Indian_pines_withoutnormalization.mat",   #这儿放加噪图
        "result_dir": r"D:\learn_pytorch\SSGN_orig\real_data_res",
        "model_path": r"D:\learn_pytorch\SSGN_orig\saved_models\GT_crop\pretrained\experiment_0\models\epoch_0lll",
        #'target_img':r'F:\lian\SSGN_orig\data\gausse\target.mat',
        # "orig_dir": r"D:\learn_pytorch\data\t50.mat",
        # "result_dir": r"D:\learn_pytorch\res",
        # "model_path": r"D:\learn_pytorch\SSGN\saved_models\GT_crop\pretrained\experiment_10\models\epoch_222",
    },
}

def get_data(imgs,index,channels_num):
    K=TEST_CFG['GF']['K']
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

def Tester(cfg_name):
    cfg = TEST_CFG[cfg_name]
    saved_model_path = cfg['model_path']
    print(f'Load weights: {saved_model_path}')
    model =SSGN(24)
    model.load_state_dict(torch.load(saved_model_path, map_location='cpu')['model'], strict=True)
    model = model.to(DEVICE)
    result_dir = cfg['result_dir']
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    imgs,pre_noise,t,psnr,ssim,sam=use(model)
    scio.savemat(os.path.join(result_dir,'denoise_Indian_lll.mat'),{'img':imgs})
    #scio.savemat(os.path.join(result_dir, 'pre_noise.mat'), {'img': pre_noise})
    return imgs,pre_noise,t,psnr,ssim,sam

def use(model):
    # cfg = TEST_CFG['GF']
    # orig= scio.loadmat(cfg['orig_dir'])['img']    #改一下
    # #orig_imgs_scaler= scaler_mean(orig)
    # c, y, x = orig.shape  # 单个独立下划线是用作一个名字，来表示某个变量是临时的或无关紧要.在这行代码中，"_"作为占位符变量可以派上用场
    # # x1 = random.randint(0, x - 25)  # W,随机产生[0,x-patch_size]之间的整数
    # # y1 = random.randint(0, y - 25)  # H
    # # im = orig[:, y1:y1 + 25, x1:x1 + 25].copy()
    # orig_imgs_scaler = scaler_max_all(orig)
    # clean=orig_imgs_scaler.copy()
    # orig_addnoise=get_noisy_image(orig_imgs_scaler)
    orig_addnoise=scio.loadmat(TEST_CFG['GF']['orig_dir'])['img']
    orig_addnoise_scaler = scaler_max_all(orig_addnoise.copy())
    target=orig_addnoise_scaler
    #target=scio.loadmat(r"D:\learn_pytorch\SSGN_orig\data\gausse\target.mat")['img']  #测试高五这一行给个路径就行，给的是谁都可以
    imgs = []
    pre_noise=[]
    model.eval()
    psnr=[]
    ssim=0.0
    t=0.0
    with torch.no_grad():
        for i in range(np.shape(orig_addnoise_scaler)[0]):
            input_im1, sobel_spa2,input_im_25,sobelxy_50,sobel_spe24= get_data(orig_addnoise_scaler, i, np.shape(orig_addnoise_scaler)[0])
            input_im1 = torch.from_numpy(input_im1[np.newaxis,:]).float().to(DEVICE)#转化为tensor
            sobel_spa2 =  torch.from_numpy(sobel_spa2[np.newaxis]).float().to(DEVICE)
            input_im_25= torch.from_numpy(input_im_25[np.newaxis, :]).float().to(DEVICE)
            sobelxy_50 = torch.from_numpy(sobelxy_50[np.newaxis, :]).float().to(DEVICE)
            sobel_spe24 = torch.from_numpy(sobel_spe24[np.newaxis, :]).float().to(DEVICE)
            start_time = time.time()
            noise_res,out_noise_img_25 = model(input_im1, sobel_spa2,input_im_25,sobelxy_50,sobel_spe24)   #残差输出
            denoise_img = input_im1 - noise_res       #去噪后的波段
            denoise_img = denoise_img.detach().cpu().squeeze().numpy()
            #denoise_img = np.clip(denoise_img,0,1)
            imgs.append(denoise_img)
            end_time = time.time()
            noise_res = noise_res.detach().cpu().squeeze().numpy()
            #noise_res = np.clip(noise_res,0,1)

            cv2.imshow('denoise', denoise_img)                   #高五去噪后的图
            cv2.imshow('orig_addnoise',orig_addnoise_scaler[i])  #高五原始带噪图
            cv2.imshow('pre_noise', noise_res)                   #预测的高五噪声
            cv2.waitKey(2)

            pre_noise.append(noise_res)
            psnr.append(calc_psnr(denoise_img,target[i]))
            ssim+=compare_ssim(target[i],denoise_img)
            t+=end_time-start_time
            print('第{}个通道正在降噪'.format(i))
        imgs = np.array(imgs, dtype=np.float32)
        pre_noise = np.array(pre_noise, dtype=np.float32)
        psnr=np.array(psnr, dtype=np.float32)
        print(psnr)
        psnr=np.sum(psnr)/np.shape(orig_addnoise)[0]
        ssim/=np.shape(orig_addnoise)[0]
        sam=calc_sam(target, imgs)
    return imgs,pre_noise,t,psnr,ssim,sam

# class Tester:
#     def __init__(self, cfg_name, scene_id=-1) -> None:
#         self.cfg = TEST_CFG[cfg_name]
#         saved_model_path = self.cfg['model_path']
#         print(f'Load weights: {saved_model_path}')
#         model =SSGN(24)
#         model.load_state_dict(torch.load(saved_model_path, map_location='cpu')['model'], strict=True)
#         model = model.to(DEVICE)
#         self.result_dir = self.cfg['result_dir']
#         if not os.path.exists(self.result_dir):
#             os.mkdir(self.result_dir)
#         imgs,pre_noise,start_time,end_time=self.use(model)
#         scio.savemat(os.path.join(self.result_dir,'denoise313.mat'),{'img':imgs})
#         scio.savemat(os.path.join(self.result_dir, 'pre_noise313.mat'), {'img': pre_noise})
#
#     def use(self, model):
#         orig_imgs= scio.loadmat(self.cfg['orig_dir'])['img']    #改一下
#         orig,max,min=scaler1(orig_imgs)
#         #orig =h5py.File('D:\learn_pytorch\hsid_cnn_pytorch_main\data\GF.mat', mode='r')['data']
#         #orig = np.transpose(orig, (2, 0, 1))   #HWC->CHW
#         imgs = []
#         pre_noise=[]
#         model.eval()
#         with torch.no_grad():
#             start_time = time.time()
#             for i in range(np.shape(orig)[0]):
#                 input_im1, sobel_spa2,input_im_25,sobelxy_50,sobel_spe24= get_data( orig, i, np.shape(orig)[0])
#                 input_im1 = torch.from_numpy(input_im1[np.newaxis,:]).float().to(DEVICE)#转化为tensor
#                 sobel_spa2 =  torch.from_numpy(sobel_spa2[np.newaxis]).float().to(DEVICE)
#                 input_im_25= torch.from_numpy(input_im_25[np.newaxis, :]).float().to(DEVICE)
#                 sobelxy_50 = torch.from_numpy(sobelxy_50[np.newaxis, :]).float().to(DEVICE)
#                 sobel_spe24 = torch.from_numpy(sobel_spe24[np.newaxis, :]).float().to(DEVICE)
#                 noise_res,out_noise_img_25 = model(input_im1, sobel_spa2,input_im_25,sobelxy_50,sobel_spe24)   #残差输出
#                 denoise_img = input_im1 - noise_res       #去噪后的波段
#                 denoise_img = denoise_img.detach().cpu().numpy()
#                 denoise_img = np.squeeze(denoise_img)
#                 imgs.append(denoise_img)
#                 noise_res = noise_res.detach().cpu().numpy()
#                 noise_res = np.squeeze(noise_res)
#                 pre_noise.append(noise_res)
#                 print('第{}个通道正在降噪'.format(i))
#             imgs = np.array(imgs, dtype=np.float32)
#             pre_noise = np.array(pre_noise, dtype=np.float32)
#             imgs=back_scaler(imgs, max, min)
#             end_time = time.time()
#             #imgs = np.transpose(imgs, (1,2,0))  #CHW->HWC
#         return imgs,pre_noise,start_time,end_time

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
    imgs,pre_noise,t,psnr,ssim,sam=Tester('GF')
    total_time = getSpanTimes(t, unit='zh')
    print('PSNR={:.3f},SSIM={:.3f},Running time:{},SAM={:.3f}'.format(psnr,ssim,total_time,sam))

