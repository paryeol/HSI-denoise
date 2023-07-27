import torch
import torch.nn as nn

class SpatialNet(nn.Module):
    def __init__(self):
        super(SpatialNet, self).__init__()
        self.spatial_cnn_3 = nn.Conv2d(1, 30, 3, 1, 1)#输入通道数，输出通道数，卷积核大小，步幅，填充
        self.spatial_cnn_5 = nn.Conv2d(1, 30, 5, 1, 2)
        self.spatial_cnn_7 = nn.Conv2d(1, 30, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, spatial_x):
        spatial_feature_3 = self.spatial_cnn_3(spatial_x)
        spatial_feature_5 = self.spatial_cnn_5(spatial_x)
        spatial_feature_7 = self.spatial_cnn_7(spatial_x)
        spatial_feature = torch.cat((spatial_feature_5, spatial_feature_7, spatial_feature_3), dim=1)
        spatial_feature = self.relu(spatial_feature)
        return spatial_feature

class Gradient_spa(nn.Module):
    def __init__(self):
        super(Gradient_spa, self).__init__()
        self.spatial_cnn_3 = nn.Conv2d(2, 30, 3, 1, 1)
        self.spatial_cnn_5 = nn.Conv2d(2, 30, 5, 1, 2)
        self.spatial_cnn_7 = nn.Conv2d(2, 30, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, spatial_x):
        spatial_feature_3 = self.spatial_cnn_3(spatial_x)
        spatial_feature_5 = self.spatial_cnn_5(spatial_x)
        spatial_feature_7 = self.spatial_cnn_7(spatial_x)
        Gradient_spa_feature = torch.cat(( spatial_feature_5, spatial_feature_7,spatial_feature_3), dim=1)
        Gradient_spa_feature= self.relu(Gradient_spa_feature)
        return Gradient_spa_feature

class Gradient_spe(nn.Module):
    def __init__(self, in_channels):
        super(Gradient_spe, self).__init__()
        self.spectral_cnn_3 = nn.Conv2d(in_channels, 30, 3, 1, 1)
        self.spectral_cnn_5 = nn.Conv2d(in_channels, 30, 5, 1, 2)
        self.spectral_cnn_7 = nn.Conv2d(in_channels, 30, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, spectral_x):
        spectral_feature_3 = self.spectral_cnn_3(spectral_x)
        spectral_feature_5 = self.spectral_cnn_5(spectral_x)
        spectral_feature_7 = self.spectral_cnn_7(spectral_x)
        Gradient_spe_feature = torch.cat((spectral_feature_5, spectral_feature_7, spectral_feature_3), dim=1)
        Gradient_spe_feature = self.relu(Gradient_spe_feature)
        return Gradient_spe_feature

# def Block(c):
#     Block_cnn_3 = nn.Conv2d(300, 30, 3, 1, 1)
#     Block_cnn_5 = nn.Conv2d(300, 30, 5, 1, 2)
#     Block_cnn_7 = nn.Conv2d(300, 30, 7, 1, 3)
#     Block_cnn_1 = nn.Conv2d(300, 30, 1, 1)
#     Gradient_spe_feature = torch.cat((spectral_feature_5, spectral_feature_7, spectral_feature_3,spectral_feature_1), dim=1)
#     Gradient_spe_feature = nn.ReLU(inplace=True)
#     return Gradient_spe_feature

class SSGN(nn.Module):
    def __init__(self, in_channels):
        super(SSGN, self).__init__()
        self.Gradient_spe_net = Gradient_spe(in_channels)
        self.Gradient_spa_net = Gradient_spa()
        self.spatial_net = SpatialNet()

        self.f5 = nn.Conv2d(270, 30, 5, 1, 2)
        self.f7 = nn.Conv2d(270, 30, 7, 1, 3)
        self.f3 = nn.Conv2d(270, 30, 3, 1, 1)

        self.f5_ = nn.Conv2d(90, 30, 5, 1, 2)
        self.f7_ = nn.Conv2d(90, 30, 7, 1, 3)
        self.f3_ = nn.Conv2d(90, 30, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.conv_out1 = nn.Conv2d(90, 1, 3, 1, 1)

    def forward(self, spatial_x,gra_spa,spatial_x_spe, gra_spa_spe,gra_spe):
        batch_size,channel_size,height,width=spatial_x_spe.shape        #[16,25,20,20]
        spatial_x_spe=torch.reshape(spatial_x_spe,[batch_size*channel_size,1,height,width])#[16,25,20,20]->[16*25,1,20,20]
        gra_spa_spe=torch.reshape(gra_spa_spe,[batch_size*channel_size,2,height,width])    #[16,50,20,20]->[16*25,2,20,20]

        spatial_feature = self.spatial_net(spatial_x)            #[16,1,20,20]->[16,90,_,_]
        spatial_feature_spe = self.spatial_net(spatial_x_spe)    #[16*25,1,20,20]->[16*25,90,_,_]

        gra_spa_feature = self.Gradient_spa_net(gra_spa)         #[16,2,20,20]->[16,90,_,_]
        gra_spa_feature_spe = self.Gradient_spa_net(gra_spa_spe) #[16*25,2,20,20]->[16*25,120,_,_]
        gra_spe_feature = self.Gradient_spe_net(gra_spe)         #[16,24,20,20]->[16,120,_,_]

        gra_channel_num=gra_spe.shape[1]    #24
        gra_spe1 = torch.unsqueeze(gra_spe, 1)#[16,24,_,_]->[16,1,24,_,_]
        gra_spe1 = torch.repeat_interleave(gra_spe1,channel_size,dim=1)#复制25份，[16,1,24,_,_]->[16,25,24,_,_]
        gra_spe1 = torch.reshape(gra_spe1,[batch_size*channel_size,gra_channel_num,height,width])#[16,25,24,_,_]->[16*25,24,_,_]
        gra_spe1_feature = self.Gradient_spe_net(gra_spe1)
        #origin_channel
        feat = torch.cat((gra_spa_feature, spatial_feature, gra_spe_feature), dim=1)
        # block_1
        feat = torch.cat((self.f5(feat), self.f7(feat), self.f3(feat)), dim=1)
        feat = self.relu(feat)
        # block_2
        feat = torch.cat((self.f5_(feat), self.f7_(feat), self.f3_(feat)), dim=1)
        feat = self.relu(feat)
        # block_3
        feat = torch.cat((self.f5_(feat), self.f7_(feat), self.f3_(feat)), dim=1)
        feat = self.relu(feat)
        # block_4
        feat = torch.cat((self.f5_(feat), self.f7_(feat), self.f3_(feat)), dim=1)
        feat = self.relu(feat)
        # block_5
        feat = torch.cat((self.f5_(feat), self.f7_(feat), self.f3_(feat)), dim=1)
        feat = self.relu(feat)

        out_noise_img = self.conv_out1(feat)
        #out_noise_img=self.sigmoid(out_noise_img)

        # out_noise_img = torch.reshape(out_noise_img, [batch_size, channel_size, height, width])
        # #spe_channel
        feat_spe = torch.cat((gra_spa_feature_spe,spatial_feature_spe,gra_spe1_feature), dim=1)
        # block_1
        feat_spe=torch.cat((self.f5(feat_spe),self.f7(feat_spe),self.f3(feat_spe)), dim=1)
        feat_spe=self.relu(feat_spe)
        # block_2
        feat_spe=torch.cat((self.f5_(feat_spe),self.f7_(feat_spe),self.f3_(feat_spe)), dim=1)
        feat_spe=self.relu(feat_spe)
        # block_3
        feat_spe = torch.cat((self.f5_(feat_spe), self.f7_(feat_spe), self.f3_(feat_spe)), dim=1)
        feat_spe = self.relu(feat_spe)
        # block_4
        feat_spe = torch.cat((self.f5_(feat_spe), self.f7_(feat_spe), self.f3_(feat_spe)), dim=1)
        feat_spe = self.relu(feat_spe)
        # block_5
        feat_spe = torch.cat((self.f5_(feat_spe), self.f7_(feat_spe), self.f3_(feat_spe)), dim=1)
        feat_spe = self.relu(feat_spe)

        out_noise_img_25 = self.conv_out1(feat_spe)
        #out_noise_img_25 = self.sigmoid(out_noise_img_25)
        out_noise_img_25=torch.reshape(out_noise_img_25,[batch_size,channel_size,height,width])

        return out_noise_img,out_noise_img_25

# class SSGN(nn.Module):
#     def __init__(self, in_channels):
#         super(SSGN, self).__init__()
#         self.Gradient_spe_net = Gradient_spe(in_channels)
#         self.Gradient_spa_net = Gradient_spa()
#         self.spatial_net = SpatialNet()
#
#         self.f5 = nn.Conv2d(300, 30, 5, 1, 2)
#         self.f7 = nn.Conv2d(300, 30, 7, 1, 3)
#         self.f3 = nn.Conv2d(300, 30, 3, 1, 1)
#         self.f1 = nn.Conv2d(300, 30, 1, 1)
#
#         self.f5_ = nn.Conv2d(120, 30, 5, 1, 2)
#         self.f7_ = nn.Conv2d(120, 30, 7, 1, 3)
#         self.f3_ = nn.Conv2d(120, 30, 3, 1, 1)
#         self.f1_ = nn.Conv2d(120, 30, 1, 1)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.conv_out1 = nn.Conv2d(120, 1, 3, 1, 1)
#         #self.conv_out2 = nn.Conv2d(120, in_channels, 3, 1, 1)
#     def forward(self, spatial_x, gra_spa,gra_spe):
#         spatial_feature = self.spatial_net(spatial_x)
#         gra_spa_feature = self.Gradient_spa_net(gra_spa)
#         gra_spe_feature = self.Gradient_spe_net(gra_spe)
#         print(gra_spe_feature.shape)
#
#
#         feat = torch.cat((gra_spa_feature,spatial_feature,gra_spe_feature), dim=1)
#         # block_1
#         feat=torch.cat((self.f5(feat),self.f7(feat),self.f3(feat),self.f1(feat)), dim=1)
#         feat=self.relu(feat)
#         # block_2
#         feat=torch.cat((self.f5_(feat),self.f7_(feat),self.f3_(feat),self.f1_(feat)), dim=1)
#         feat=self.relu(feat)
#         # block_3
#         feat = torch.cat((self.f5_(feat), self.f7_(feat), self.f3_(feat), self.f1_(feat)), dim=1)
#         feat = self.relu(feat)
#         # block_4
#         feat = torch.cat((self.f5_(feat), self.f7_(feat), self.f3_(feat), self.f1_(feat)), dim=1)
#         feat = self.relu(feat)
#         # block_5
#         feat = torch.cat((self.f5_(feat), self.f7_(feat), self.f3_(feat), self.f1_(feat)), dim=1)
#         feat = self.relu(feat)
#
#         out_noise_img = self.conv_out1(feat)
        #out_noise_grad_spe=self.conv_out2(feat)
        #out_noise_grad_spe=torch.from_numpy(out_noise_grad_spe).float() #转化为张量

        #return out_noise_img,out_noise_grad_spe
        #return out_noise_img
if __name__ == '__main__':
    # device = torch.device('cuda:1')
    device = torch.device('cuda')
    x1 = torch.randn((16, 1, 20, 20)).to(device)
    x2 = torch.randn((16, 2, 20, 20)).to(device)
    x3 = torch.randn((16, 25, 20, 20)).to(device)
    x4 = torch.randn((16, 50, 20, 20)).to(device)
    x5 = torch.randn((16, 24, 20, 20)).to(device)
    with torch.no_grad():
        model = SSGN(24).to(device)
        pred,pred_spe = model(x1, x2,x3,x4,x5)
    print(pred.shape,pred_spe.shape)






