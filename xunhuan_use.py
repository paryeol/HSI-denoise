import cv2
import scipy.io as scio
from torch.utils.tensorboard import SummaryWriter

from utils import *
from network_SSGN_fpn import SSGN_spe_fpn
from utils import calc_psnr

print(torch.cuda.is_available())
DEVICE = torch.device('cuda')
TEST_CFG = {
    "GF": {
        'K': 24,
        "orig_dir": r"D:\learn_pytorch\data\GT_add_noise313.mat",
        "result_dir": r'D:\learn_pytorch\SSGN\saved_models\GT_crop\pretrained\experiment_10',
        # "model_path": r"D:\learn_pytorch\SSGN\saved_models\GT_crop\pretrained\experiment_10",
        "clean_dir": r"D:\learn_pytorch\data\GTcrop_150.mat"
    },
}


# 获取保存的模型的列表
def get_model_list(x):
    model_list = []
    weights_dir = os.path.join('D:\learn_pytorch\SSGN\saved_models\GT_crop\pretrained\experiment_10', 'models')
    weights_name = os.listdir(weights_dir)
    for i in range(len(weights_name)):
        juedui = os.path.join(weights_dir, weights_name[i])
        model_list.append(juedui)
    return model_list


def scaler1(x):
    max = np.max(x)
    min = np.min(x)
    y = []
    for i in range(x.shape[0]):
        # print('x[i]='+str(x[i]),'min='+str(min),'x[i] - min='+str(x[i] - min))
        scaler = (x[i] - min) / (max - min)
        y.append(scaler)
    y = np.array(y, dtype=np.float32)
    return y


def get_data(imgs, index, channels_num):
    K = TEST_CFG['GF']['K']
    img = imgs[index]
    sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
    sobelx = sobelx[np.newaxis, :, :]
    sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)
    sobely = sobely[np.newaxis, :, :]
    # 2
    sobel_spa = np.concatenate((sobelx, sobely), axis=0)

    if index < K // 2:
        clean_vol = imgs[0:K + 1, :, :]
    elif index > channels_num // 2 - 1:
        clean_vol = imgs[-K - 1:, :, :]
    else:
        clean_vol = imgs[index - K // 2:index + K // 2 + 1, :, :]
    # 25
    input_im_25 = clean_vol.copy()
    # 24
    sobel_spe = spe_gradient_1(clean_vol)
    # sobel_spe,index = spe_gradient_11(clean_vol,index,K,channels_num)

    # print(sobel_spe.shape)
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
    # 50
    sobelxy_25 = sobelxy_25.reshape((b * c, h, w))

    noise_im = imgs[index]
    noise_im = noise_im[np.newaxis, :, :]

    return noise_im, sobel_spa, input_im_25, sobelxy_25, sobel_spe


class Tester:
    def __init__(self, cfg_name, scene_id=-1) -> None:
        self.cfg = TEST_CFG[cfg_name]
        saved_model = get_model_list('1')
        print(f'Load weights: {saved_model}')
        model = SSGN_spe_fpn(24)
        self.result_dir = os.path.join(self.cfg['result_dir'], 'result')
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        clean = scaler1(scio.loadmat(self.cfg['clean_dir'])['img'])

        writer = SummaryWriter(self.result_dir+'psnr/save_epoch')
        PSNRzhi=[]
        for i in range(len(saved_model)):
            model.load_state_dict(torch.load(saved_model[i], map_location='cpu')['model'], strict=True)
            model = model.to(DEVICE)
            imgs, pre_noise = self.use(model)
            psnr = calc_psnr(imgs, clean)
            PSNRzhi.append(psnr)
            writer.add_scalar('PSNR/train',psnr,i)
            print(psnr)
        print('最终'+str(PSNRzhi))
            # scio.savemat(os.path.join(self.result_dir,'denoise313.mat'),{'img':imgs})
            # scio.savemat(os.path.join(self.result_dir, 'pre_noise313.mat'), {'img': pre_noise})

    def use(self, model):
        orig_imgs = scio.loadmat(self.cfg['orig_dir'])['img']  # 改一下
        orig, max, min = scaler1(orig_imgs)
        # orig =h5py.File('D:\learn_pytorch\hsid_cnn_pytorch_main\data\GF.mat', mode='r')['data']
        # orig = np.transpose(orig, (2, 0, 1))   #HWC->CHW
        imgs = []
        pre_noise = []
        model.eval()
        with torch.no_grad():
            for i in range(np.shape(orig)[0]):
                input_im1, sobel_spa2, input_im_25, sobelxy_50, sobel_spe24 = get_data(orig, i, np.shape(orig)[0])
                input_im1 = torch.from_numpy(input_im1[np.newaxis, :]).float().to(DEVICE)  # 转化为tensor
                sobel_spa2 = torch.from_numpy(sobel_spa2[np.newaxis]).float().to(DEVICE)
                input_im_25 = torch.from_numpy(input_im_25[np.newaxis, :]).float().to(DEVICE)
                sobelxy_50 = torch.from_numpy(sobelxy_50[np.newaxis, :]).float().to(DEVICE)
                sobel_spe24 = torch.from_numpy(sobel_spe24[np.newaxis, :]).float().to(DEVICE)
                noise_res, out_noise_img_25 = model(input_im1, sobel_spa2, input_im_25, sobelxy_50, sobel_spe24)  # 残差输出
                denoise_img = input_im1 - noise_res  # 去噪后的波段
                denoise_img = denoise_img.detach().cpu().numpy()
                denoise_img = np.squeeze(denoise_img)
                imgs.append(denoise_img)
                noise_res = noise_res.detach().cpu().numpy()
                noise_res = np.squeeze(noise_res)
                pre_noise.append(noise_res)
                print('第{}个通道正在降噪'.format(i))
            imgs = np.array(imgs, dtype=np.float32)
            pre_noise = np.array(pre_noise, dtype=np.float32)
            imgs = back_scaler(imgs, max, min)
            # imgs = np.transpose(imgs, (1,2,0))  #CHW->HWC
        return imgs, pre_noise


if __name__ == '__main__':
    Tester('GF', 1)
