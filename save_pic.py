import h5py
from matplotlib.pyplot import gcf

from utils import *
from matplotlib import pyplot as plt
#更改orig1-orig9的地址
#更改要保存的图片的地址
orig_path=r'D:\learn_pytorch\SSGN_orig\data\snr811/'
save_path=r'D:\learn_pytorch\SSGN_orig\save_pic\snr811/'
#
chooseh=64
choosew=64

orig1=scio.loadmat(orig_path+'target.mat')['img']
orig2=scio.loadmat(orig_path+'orig_addnoise.mat')['img']
orig3=h5py.File(orig_path+'LRTA_denoise.mat')['img']
orig3=np.transpose(orig3,(0,2,1))#CWH->CHW
orig4=h5py.File(orig_path+'BM4D_denoise.mat')['img']
orig4=np.transpose(orig4,(0,2,1))#CWH->CHW
orig5=h5py.File(orig_path+'LRTV_denoise.mat')['img']
orig5=np.transpose(orig5,(0,2,1))#CWH->CHW
orig6=h5py.File(orig_path+'LRMR_denoise.mat')['img']
orig6=np.transpose(orig6,(0,2,1))#CWH->CHW
orig7=h5py.File(orig_path+'NALLRMA_denoise.mat')['img']
orig7=np.transpose(orig7,(0,2,1))#CWH->CHW
orig8=scio.loadmat(orig_path+'SSGN_denoise.mat')['img']
orig9=scio.loadmat(orig_path+'our_denoise.mat')['img']


c,h,w=orig1.shape
################################################## 画指标对比图   ########################################################
##各种方法的PSNR图和SSIM图
orig_addnoise_psnr=[]
LRTA_psnr=[]
BM4D_psnr=[]
LRTV_psnr=[]
LRMR_psnr=[]
NALLRMA_psnr=[]
SSGN_psnr=[]
our_psnr=[]

orig_addnoise_ssim=[]
LRTA_ssim=[]
BM4D_ssim=[]
LRTV_ssim=[]
LRMR_ssim=[]
NALLRMA_ssim=[]
SSGN_ssim=[]
our_ssim=[]

target_DN=[]
orig_addnoise_DN=[]
LRTA_DN=[]
BM4D_DN=[]
LRTV_DN=[]
LRMR_DN=[]
NALLRMA_DN=[]
SSGN_DN=[]
our_DN=[]

for i in range(c):
    print(i)
    #psnr
    orig_addnoise_psnr_tmp=calc_psnr(orig1[i],orig2[i])
    LRTA_psnr_tmp=calc_psnr(orig1[i],orig3[i])
    BM4D_psnr_tmp = calc_psnr(orig1[i], orig4[i])
    LRTV_psnr_tmp = calc_psnr(orig1[i], orig5[i])
    LRMR_psnr_tmp = calc_psnr(orig1[i], orig6[i])
    NALLRMA_psnr_tmp = calc_psnr(orig1[i], orig7[i])
    SSGN_psnr_tmp = calc_psnr(orig1[i], orig8[i])
    our_psnr_tmp = calc_psnr(orig1[i], orig9[i])

    orig_addnoise_psnr.append(orig_addnoise_psnr_tmp)
    LRTA_psnr.append(LRTA_psnr_tmp)
    BM4D_psnr.append(BM4D_psnr_tmp)
    LRTV_psnr.append(LRTV_psnr_tmp)
    LRMR_psnr.append(LRMR_psnr_tmp)
    NALLRMA_psnr.append(NALLRMA_psnr_tmp)
    SSGN_psnr.append(SSGN_psnr_tmp)
    our_psnr.append(our_psnr_tmp)

    #ssim
    orig_addnoise_ssim_tmp = compare_ssim(orig1[i],orig2[i])
    LRTA_ssim_tmp = compare_ssim(orig1[i],orig3[i])
    BM4D_ssim_tmp = compare_ssim(orig1[i], orig4[i])
    LRTV_ssim_tmp = compare_ssim(orig1[i], orig5[i])
    LRMR_ssim_tmp = compare_ssim(orig1[i], orig6[i])
    NALLRMA_ssim_tmp = compare_ssim(orig1[i], orig7[i])
    SSGN_ssim_tmp = compare_ssim(orig1[i], orig8[i])
    our_ssim_tmp = compare_ssim(orig1[i], orig9[i])


    # orig_addnoise_ssim_tmp = np.mean(orig2[i])
    # LRTA_ssim_tmp = np.mean(orig3[i])
    # BM4D_ssim_tmp = np.mean( orig4[i])
    # LRTV_ssim_tmp = np.mean( orig5[i])
    # LRMR_ssim_tmp = np.mean(orig6[i])
    # NALLRMA_ssim_tmp = np.mean(orig7[i])
    # SSGN_ssim_tmp = np.mean(orig8[i])
    # our_ssim_tmp = np.mean(orig9[i])

    orig_addnoise_ssim.append(orig_addnoise_ssim_tmp)
    LRTA_ssim.append(LRTA_ssim_tmp)
    BM4D_ssim.append(BM4D_ssim_tmp)
    LRTV_ssim.append(LRTV_ssim_tmp)
    LRMR_ssim.append(LRMR_ssim_tmp)
    NALLRMA_ssim.append(NALLRMA_ssim_tmp)
    SSGN_ssim.append(SSGN_ssim_tmp)
    our_ssim.append(our_ssim_tmp)

    #DN
    target_DN_tmp=orig1[i,chooseh,choosew]
    orig_addnoise_DN_tmp=orig2[i,chooseh,choosew]
    LRTA_DN_tmp=orig3[i,chooseh,choosew]
    BM4D_DN_tmp =orig4[i,chooseh,choosew]
    LRTV_DN_tmp =orig5[i,chooseh,choosew]
    LRMR_DN_tmp = orig6[i,chooseh,choosew]
    NALLRMA_DN_tmp =orig7[i,chooseh,choosew]
    SSGN_DN_tmp =orig8[i,chooseh,choosew]
    our_DN_tmp = orig9[i,chooseh,choosew]


    target_DN.append(target_DN_tmp)
    orig_addnoise_DN.append(orig_addnoise_DN_tmp)
    LRTA_DN.append(LRTA_DN_tmp)
    BM4D_DN.append(BM4D_DN_tmp)
    LRTV_DN.append(LRTV_DN_tmp)
    LRMR_DN.append(LRMR_DN_tmp)
    NALLRMA_DN.append(NALLRMA_DN_tmp)
    SSGN_DN.append(SSGN_DN_tmp)
    our_DN.append(our_DN_tmp)


plt.figure(1)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("PSNR",fontdict={'family':'Times New Roman'})#y轴上的名字
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,orig_addnoise_psnr,label='Noisy',color='k')
plt.plot(x,LRTA_psnr,label='LRTA',color='blue')
plt.plot(x,BM4D_psnr,label='BM4D',color='yellow')
plt.plot(x,LRTV_psnr,label='LRTV',color='g')
plt.plot(x,LRMR_psnr,label='LRMR',color='pink')
plt.plot(x,NALLRMA_psnr,label='NAILRMA',color='c')
plt.plot(x,SSGN_psnr,label='SSGN',color='m')
plt.plot(x,our_psnr,label='our',color='r')
#plt.ylim(15,43)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(loc=4,prop={'family':'Times New Roman'})
plt.savefig(save_path+'psnr.jpg', bbox_inches='tight',dpi=300)

plt.figure(2)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['Arial']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("SSIM",fontdict={'family':'Times New Roman'})#y轴上的名字
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,orig_addnoise_ssim,label='Noisy',color='k')
plt.plot(x,LRTA_ssim,label='LRTA',color='blue')
plt.plot(x,BM4D_ssim,label='BM4D',color='yellow')
plt.plot(x,LRTV_ssim,label='LRTV',color='g')
plt.plot(x,LRMR_ssim,label='LRMR',color='pink')
plt.plot(x,NALLRMA_ssim,label='NAILRMA',color='c')
plt.plot(x,SSGN_ssim,label='SSGN',color='m')
plt.plot(x,our_ssim,label='our',color='r')
plt.ylim(0.2,1)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(loc='lower right',prop={'family':'Times New Roman'})
plt.savefig(save_path+'ssim.jpg', bbox_inches='tight',dpi=300)

plt.figure(3)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['Arial']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("Normalized DN",fontdict={'family':'Times New Roman'})#y轴上的名字
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,target_DN,label='Original',color='k')
plt.plot(x,orig_addnoise_DN,label='Noisy',color='r')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path+'DN_noise.jpg', bbox_inches='tight',dpi=300)

plt.figure(4)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("Normalized DN",fontdict={'family':'Times New Roman'})#y轴上的名字
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,target_DN,label='Original',color='k')
plt.plot(x,LRTA_DN,label='LRTA',color='r')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path+'DN_LRTA.jpg', bbox_inches='tight',dpi=300)

plt.figure(5)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("Normalized DN",fontdict={'family':'Times New Roman'})#y轴上的名字
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,target_DN,label='Original',color='k')
plt.plot(x,BM4D_DN,label='BM4D',color='r')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path+'DN_BM4D.jpg', bbox_inches='tight',dpi=300)

plt.figure(6)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("Normalized DN",fontdict={'family':'Times New Roman'})#y轴上的名字
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,target_DN,label='Original',color='k')
plt.plot(x,LRTV_DN,label='LRTV',color='r')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path+'DN_LRTV.jpg', bbox_inches='tight',dpi=300)

plt.figure(7)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("Normalized DN",fontdict={'family':'Times New Roman'})#y轴上的名字
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,target_DN,label='Original',color='k')
plt.plot(x,LRMR_DN,label='LRMR',color='r')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path+'DN_LRMR.jpg', bbox_inches='tight',dpi=300)

plt.figure(8)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("Normalized DN",fontdict={'family':'Times New Roman'})#y轴上的名字
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,target_DN,label='Original',color='k')
plt.plot(x,NALLRMA_DN,label='NAILRMA',color='r')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path+'DN_NAILRMA.jpg', bbox_inches='tight',dpi=300)

plt.figure(9)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("Normalized DN",fontdict={'family':'Times New Roman'})#y轴上的名字
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,target_DN,label='Original',color='k')
plt.plot(x,SSGN_DN,label='SSGN',color='r')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path+'DN_SSGN.jpg', bbox_inches='tight',dpi=300)

plt.figure(10)
x=np.arange(1,c+1)
#解决中文显示问题
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("Band Number",fontdict={'family':'Times New Roman'})#x轴上的名字
plt.ylabel("Normalized DN",fontdict={'family':'Times New Roman'})#y轴上的名字
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
#rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,target_DN,label='Original',color='k')
plt.plot(x,our_DN,label='our',color='r')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.yticks(fontproperties='Times New Roman') #设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path+'DN_our.jpg', bbox_inches='tight',dpi=300)



# for i in range(c):
#     cv2.imshow('target', orig1[i])
#     #print(type(orig1[i]))
#     cv2.imshow('add_noise', orig2[i])
#     cv2.imshow('LRTA', orig3[i])
#     cv2.imshow('BM4D', orig4[i])
#     cv2.imshow('LRTV', orig5[i])
#     cv2.imshow('LRMR', orig6[i])
#     cv2.imshow('NAILRMA', orig7[i])
#     cv2.imshow('SSGN', orig8[i])
#     cv2.imshow('our', orig9[i])
#     cv2.waitKey(100)

# orig10=scio.loadmat(r'D:\learn_pytorch\data\GT_crop_test.mat')['img']
# orig10 = scaler_max_all(orig10.copy())
# b1 = orig10[17,:,:]
# g1 = orig10[27,:,:]
# r1 = orig10[57,:,:]
# img_target = cv2.merge((b1,g1,r1)) #传入bgr构成的元组
# cv2.imshow('target',img_target)
# cv2.waitKey(0)

