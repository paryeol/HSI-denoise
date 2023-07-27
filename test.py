import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def load_txt(txt_path):
    # cont=open(txt_path,'r')
    # t=cont.readlines()
    # txt_seq=[t_.split('\n')[0] for t_ in t]
    txt_seq=np.loadtxt(txt_path)
    return txt_seq

# txt_path=r'F:\lian\SSGN_orig\train_list.txt'
# data_path=r'F:\read_data'
# index_list=load_txt(txt_path)
# txt=open(r'F:\lian\SSGN_orig\good.txt','w')
# print(len(index_list))
# for i in range(len(index_list)):
#     tmp_sample = np.load(os.path.join(data_path, index_list[i] + '.npy'),allow_pickle=True).item()
#     c,h,w= tmp_sample['input_im'].shape
#     if c==1 and h==25 and w==25:
#         txt.write('{:07}\n'.format(i))
#     else:
#         print(i,c,h,w )
    #print('{} finish! all:2284360'.format(i))


tmp_sample = np.load(r"F:\0000000.npy",allow_pickle=True).item()
# cv2.imshow('input_im',tmp_sample['input_im'][0])
# cv2.imshow('target_im',tmp_sample['target_im'][0])
# cv2.waitKey(0)

plt.figure()
plt.subplot(121)
plt.title('input_im')
plt.imshow(tmp_sample['input_im'][0])

plt.subplot(122)
plt.title('target_im')
plt.imshow(tmp_sample['target_im'][0])

plt.show()