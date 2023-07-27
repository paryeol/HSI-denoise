#本程序是画图：各种方法去除高五噪声后，光谱曲线对比。要反归一化
import cv2
import h5py
import scipy.io as scio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def scaler_max_all(x):
    tmp_max=np.max(np.max(x,1),1)
    tmp_min=np.min(np.min(x,1),1)
    #print('scaler_max_all',tmp_max,tmp_min)
    y=np.array([(x[i]-tmp_min[i])/(tmp_max[i]-tmp_min[i]) for i in range(len(tmp_max))])
    return y

def back_scaler(r,x):
#r：参考物，提供最大最小值
#x：需要进行反归一化的东西
#让x以r每波段的最大最小为标准进行反归一化
    y=[]
    for i in range(x.shape[0]):
        max = np.max(r[i])
        min = np.min(r[i])
        print(max,min)
        back_scaler=x[i]*(max-min)+min
        y.append(back_scaler)
    y = np.array(y, dtype=np.float32)
    return y

def black_red(img):
#将单通道图像的指定像素点变成红色，输入参数img为单通道图像
    img=Image.fromarray(img)
    img_rgb = img.convert("RGB")
    img_array = np.array(img_rgb)
    print(img_array.shape)
    h1 = img.size[0]
    w1 = img.size[1]
    img2_array = []
    # pos表示修改像素点的位置，嵌套列表中第一个数值表示行号，第二个数值表示列号
    #pos = [[15, 18], [18, 19], [13, 17], [13, 7], [13, 9], [15, 6]]
    pos=[[126,63],[208,104]]
    for i in range(0, h1):
        for j in range(0, w1):
            temp = [i, j]
            if temp in pos:
                img_array[i, j] = [255, 0, 0]
                # [255, 0, 0]为红色，[255, 255, 255]为白色，[0, 0, 0]为黑色等
                img2_array.append(img_array[i, j])
            else:
                img2_array.append(img_array[i, j])
    img2_array = np.array(img2_array)
    print(img2_array.shape)
    img2_array = img2_array.reshape(h1, w1, 3)
    img3 = Image.fromarray(img2_array)
    return img3

#更改orig1-orig9的地址
#更改要保存的图片的地址
orig_path=r'E:\data\adjust/'
save_path=r'E:\data\adjust\各方法单像素点光谱曲线\318-159/'
chooseh=318
choosew=159

orig1=scio.loadmat(r"E:\data\adjust\clean_cut.mat")['img']
c,h,w=orig1.shape
orig2=h5py.File(orig_path+'LRTA_denoise.mat')['img']
orig2=np.transpose(orig2,(0,2,1))#CWH->CHW
orig3=h5py.File(orig_path+'BM4D_denoise.mat')['img']
orig3=np.transpose(orig3,(0,2,1))#CWH->CHW
orig4=h5py.File(orig_path+'LRTV_denoise.mat')['img']
orig4=np.transpose(orig4,(0,2,1))#CWH->CHW
orig5=h5py.File(orig_path+'LRMR_denoise.mat')['img']
orig5=np.transpose(orig5,(0,2,1))#CWH->CHW
orig6=h5py.File(orig_path+'NAILRMA_denoise.mat')['img']
orig6=np.transpose(orig6,(0,2,1))#CWH->CHW
orig7=scio.loadmat(orig_path+'SSGN_denoise.mat')['img']
orig8=scio.loadmat(orig_path+'our_denoise.mat')['img']

orig2=back_scaler(orig1,orig2)
orig3=back_scaler(orig1,orig3)
orig4=back_scaler(orig1,orig4)
orig5=back_scaler(orig1,orig5)
orig6=back_scaler(orig1,orig6)
orig7=back_scaler(orig1,orig7)
orig8=back_scaler(orig1,orig8)
print(np.max(orig1),np.min(orig1))

# img = Image.fromarray(orig1[0])
# img_rgb = img.convert("RGB")
# img_array = np.array(img_rgb)
# print(img_array.shape)
# w1 = img.size[0]#200
# h1 = img.size[1]#400
# print(w1,h1)
# img2_array = []
# # pos表示修改像素点的位置，嵌套列表中第一个数值表示行号，第二个数值表示列号
# # pos = [[15, 18], [18, 19], [13, 17], [13, 7], [13, 9], [15, 6]]
# pos = [[314,157], [208, 104]]
# for i in range(0, h1):
#     for j in range(0, w1):
#         temp = [i, j]
#         if temp in pos:
#             img_array[i, j] = [255, 0, 0]
#             # [255, 0, 0]为红色，[255, 255, 255]为白色，[0, 0, 0]为黑色等
#             img2_array.append(img_array[i, j])
#         else:
#             img2_array.append(img_array[i, j])
# img2_array = np.array(img2_array)
# print(img2_array.shape)
# img2_array = img2_array.reshape(h1, w1, 3)
# img3 = Image.fromarray(img2_array)
# img3.show()
# img3.save(save_path+"1111.png")

# for j in range(w):
#     print(j)
#     # 画图，画光谱曲线
#     orig_DN = []
#     LRTA_DN = []
#     BM4D_DN = []
#     LRTV_DN = []
#     LRMR_DN = []
#     NAILRMA_DN = []
#     SSGN_DN = []
#     our_DN = []
#     for i in range(c):
#         #orig_DN_tmp=orig1[i,chooseh,choosew]
#         orig_DN_tmp = orig1[i, 2*j, j]
#         LRTA_DN_tmp = orig2[i,  2*j, j]
#         BM4D_DN_tmp = orig3[i, 2*j, j]
#         LRTV_DN_tmp = orig4[i, 2*j, j]
#         LRMR_DN_tmp = orig5[i, 2*j, j]
#         NAILRMA_DN_tmp = orig6[i,  2*j, j]
#         SSGN_DN_tmp = orig7[i, 2*j, j]
#         our_DN_tmp = orig8[i, 2*j, j]
#
#         orig_DN.append(orig_DN_tmp)
#         LRTA_DN.append(LRTA_DN_tmp)
#         BM4D_DN.append(BM4D_DN_tmp)
#         LRTV_DN.append(LRTV_DN_tmp)
#         LRMR_DN.append(LRMR_DN_tmp)
#         NAILRMA_DN.append(NAILRMA_DN_tmp)
#         SSGN_DN.append(SSGN_DN_tmp)
#         our_DN.append(our_DN_tmp)
#
#     plt.figure(1)
#     x = np.arange(1, c + 1)
#     plt.rcParams['font.sans-serif'] = ['Arial']  # 显示中文字体用SimHei，显示英文字体用Arial
#     plt.rcParams['axes.unicode_minus'] = False  # 显示负号
#     plt.xlabel("Band number", fontdict={'family': 'Times New Roman'})  # x轴上的名字
#     plt.ylabel("DN", fontdict={'family': 'Times New Roman'})  # y轴上的名字
#     plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向向内
#     plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度线方向向内
#     # rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
#     plt.plot(x, LRTA_DN, label='LRTA_denoise', color='r')
#     plt.plot(x, BM4D_DN, label='BM4D_denoise', color='g')
#     plt.plot(x, LRTV_DN, label='LRTV_denoise', color='b')
#     plt.plot(x, LRMR_DN, label='LRMR_denoise', color='c')
#     plt.plot(x, NAILRMA_DN, label='NAILRMA_denoise', color='m')
#     plt.plot(x, SSGN_DN, label='SSGN_denoise', color='y')
#     plt.plot(x, our_DN, label='our_denoise', color='pink')
#     plt.plot(x, orig_DN, label='target', color='k')
#     plt.rcParams.update({'font.size': 10})  # 设置图例字体大小
#     plt.legend(loc='upper right')
#     # plt.ylim(0.3, 0.5)
#     #plt.ylim(0,1)
#     plt.yticks(fontproperties='Times New Roman')  # 设置字体为新罗马体
#     plt.xticks(fontproperties='Times New Roman')
#     plt.legend(prop={'family': 'Times New Roman'})
#     plt.savefig(save_path + str(2*j)+'-'+str(j)+'.jpg', bbox_inches='tight', dpi=300)
#     plt.close()

# #画7张图
orig_DN=[]
LRTA_DN=[]
BM4D_DN=[]
LRTV_DN=[]
LRMR_DN=[]
NAILRMA_DN=[]
SSGN_DN=[]
our_DN=[]
for i in range(c):
    orig_DN_tmp = orig1[i,chooseh,choosew]
    LRTA_DN_tmp = orig2[i,chooseh,choosew]
    BM4D_DN_tmp = orig3[i,chooseh,choosew]
    LRTV_DN_tmp = orig4[i,chooseh,choosew]
    LRMR_DN_tmp = orig5[i,chooseh,choosew]
    NAILRMA_DN_tmp = orig6[i,chooseh,choosew]
    SSGN_DN_tmp = orig7[i,chooseh,choosew]
    our_DN_tmp = orig8[i,chooseh,choosew]

    orig_DN.append(orig_DN_tmp)
    LRTA_DN.append(LRTA_DN_tmp)
    BM4D_DN.append(BM4D_DN_tmp)
    LRTV_DN.append(LRTV_DN_tmp)
    LRMR_DN.append(LRMR_DN_tmp)
    NAILRMA_DN.append(NAILRMA_DN_tmp)
    SSGN_DN.append(SSGN_DN_tmp)
    our_DN.append(our_DN_tmp)

plt.figure(1)
x = np.arange(1,c+1)
plt.rcParams['font.sans-serif'] = ['Arial']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False   #显示负号
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
plt.xlabel("Band number",fontdict={'family':'Times New Roman'} )  # x轴上的名字
plt.ylabel("DN",fontdict={'family':'Times New Roman'} )  # y轴上的名字
# rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,orig_DN,label='Original',color='k')
plt.plot(x,LRTA_DN,label='LRTA',color='r')
plt.legend(loc='upper right')
#plt.ylim(0, 1600)
plt.ylim(0,180)
plt.yticks(fontproperties='Times New Roman')  # 设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path + 'LRTA单像素点归一化DN对比.jpg', bbox_inches='tight',dpi=300)

plt.figure(2)
x = np.arange(1,c+1)
plt.rcParams['font.sans-serif'] = ['Arial']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False   #显示负号
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
plt.xlabel("Band number",fontdict={'family':'Times New Roman'} )  # x轴上的名字
plt.ylabel("DN",fontdict={'family':'Times New Roman'} )  # y轴上的名字
# rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,orig_DN,label='Original',color='k')
plt.plot(x,BM4D_DN,label='BM4D',color='r')
plt.legend(loc='upper right')
#plt.ylim(0, 1600)
plt.ylim(0,180)
plt.yticks(fontproperties='Times New Roman')  # 设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path + 'BM4D单像素点归一化DN对比.jpg', bbox_inches='tight',dpi=300)

plt.figure(3)
x = np.arange(1,c+1)
plt.rcParams['font.sans-serif'] = ['Arial']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False   #显示负号
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
plt.xlabel("Band number",fontdict={'family':'Times New Roman'} )  # x轴上的名字
plt.ylabel("DN",fontdict={'family':'Times New Roman'} )  # y轴上的名字
# rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,orig_DN,label='Original',color='k')
plt.plot(x,LRTV_DN,label='LRTV',color='r')
plt.legend(loc='upper right')
#plt.ylim(0, 1600)
plt.ylim(0,180)
plt.yticks(fontproperties='Times New Roman')  # 设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path + 'LRTV单像素点归一化DN对比.jpg', bbox_inches='tight',dpi=300)

plt.figure(4)
x = np.arange(1,c+1)
plt.rcParams['font.sans-serif'] = ['Arial']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False   #显示负号
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
plt.xlabel("Band number",fontdict={'family':'Times New Roman'} )  # x轴上的名字
plt.ylabel("DN",fontdict={'family':'Times New Roman'} )  # y轴上的名字
# rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,orig_DN,label='Original',color='k')
plt.plot(x,LRMR_DN,label='LRMR',color='r')
plt.legend(loc='upper right')
#plt.ylim(0, 1600)
plt.ylim(0,180)
plt.yticks(fontproperties='Times New Roman')  # 设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path + 'LRMR单像素点归一化DN对比.jpg', bbox_inches='tight',dpi=300)

plt.figure(5)
x = np.arange(1,c+1)
plt.rcParams['font.sans-serif'] = ['Arial']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False   #显示负号
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
plt.xlabel("Band number",fontdict={'family':'Times New Roman'} )  # x轴上的名字
plt.ylabel("DN",fontdict={'family':'Times New Roman'} )  # y轴上的名字

# rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,orig_DN,label='Original',color='k')
plt.plot(x,NAILRMA_DN,label='NAILRMA',color='r')
plt.legend(loc='upper right')
#plt.ylim(0, 1600)
plt.ylim(0,180)
plt.yticks(fontproperties='Times New Roman')  # 设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path + 'NAILRMA单像素点归一化DN对比.jpg', bbox_inches='tight',dpi=300)

plt.figure(6)
x = np.arange(1,c+1)
plt.rcParams['font.sans-serif'] = ['Arial']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False   #显示负号
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
plt.xlabel("Band number",fontdict={'family':'Times New Roman'} )  # x轴上的名字
plt.ylabel("DN",fontdict={'family':'Times New Roman'} )  # y轴上的名字
# rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,orig_DN,label='Original',color='k')
plt.plot(x,SSGN_DN,label='SSGN',color='r')
plt.legend(loc='upper right')
#plt.ylim(0, 1600)
plt.ylim(0,180)
plt.yticks(fontproperties='Times New Roman')  # 设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path + 'SSGN单像素点归一化DN对比.jpg', bbox_inches='tight',dpi=300)

plt.figure(7)
x = np.arange(1,c+1)
plt.rcParams['font.sans-serif'] = ['Arial']#显示中文字体用SimHei，显示英文字体用Arial
plt.rcParams['axes.unicode_minus'] = False   #显示负号
plt.rcParams['xtick.direction']='in'#将x轴的刻度线方向向内
plt.rcParams['ytick.direction']='in'#将y轴的刻度线方向向内
plt.xlabel("Band number",fontdict={'family':'Times New Roman'} )  # x轴上的名字
plt.ylabel("DN",fontdict={'family':'Times New Roman'} )  # y轴上的名字
# rgb是红绿蓝,c青绿色,m洋红色,y黄色,k黑色,w白色
plt.plot(x,orig_DN,label='Original',color='k')
plt.plot(x,our_DN,label='our',color='r')
plt.legend(loc='upper right')
#plt.ylim(0, 1600)
plt.ylim(0,180)
plt.yticks(fontproperties='Times New Roman')  # 设置字体为新罗马体
plt.xticks(fontproperties='Times New Roman')
plt.legend(prop={'family':'Times New Roman'})
plt.savefig(save_path + 'our单像素点归一化DN对比.jpg', bbox_inches='tight',dpi=300)