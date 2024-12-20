import torch
import torch.nn as nn
from model.unetseries import _EncoderBlock,_DecoderBlock,UNet
import torch.nn.functional as F


class incontext_encoder(nn.Module):
    def __init__(self,in_channels=1,bn=False):
        super().__init__()
        self.enc1 = _EncoderBlock(in_channels, 32, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32, 64, bn=bn)
        self.enc3 = _EncoderBlock(64, 128, bn=bn)
        self.enc4 = _EncoderBlock(128, 256, bn=bn, dropout=False)
        self.polling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(256, 512, 256, bn=bn)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        return center
class mainUNet(nn.Module):
    def __init__(self, sample_num,in_channels=2, out_channels=1, bn=False):
        super(mainUNet, self).__init__()
        self.samplesEncoder = incontext_encoder()
        self.enc1 = _EncoderBlock(in_channels, 32, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32, 64, bn=bn)
        self.enc3 = _EncoderBlock(64, 128, bn=bn)
        self.enc4 = _EncoderBlock(128, 256, bn=bn, dropout=False)
        self.polling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(256, 512, 256, bn=bn)
        self.dec4 = _DecoderBlock(512+256*sample_num, 256, 128, bn=bn)
        self.dec3 = _DecoderBlock(256, 128, 64, bn=bn)
        self.dec2 = _DecoderBlock(128, 64, 32, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if bn else nn.GroupNorm(32, 64),
            nn.GELU()
        )
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, com,samples):
        batch_size, sample_num, _, _ = samples.shape  # 假设samples的形状是[10, 2, 64, 64]

        embeddings = []   #存储示例的中间嵌入
        # 逐个样本进行projlayer1的前向传播
        for i in range(sample_num):
            sample_i = samples[:, i, :, :].unsqueeze(1)  # 获取第i个样本，并增加一个维度，形状变为[10, 1, 64, 64]
            embedding = self.samplesEncoder(sample_i)
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings,dim=1)
        enc1 = self.enc1(com)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        incontext_center = torch.cat([center,embeddings],1)
        dec4 = self.dec4(
            torch.cat([F.interpolate(incontext_center, enc4.size()[-2:], mode='bilinear', align_corners=True), enc4], 1))
        dec3 = self.dec3(
            torch.cat([F.interpolate(dec4, enc3.size()[-2:], mode='bilinear', align_corners=True), enc3], 1))
        dec2 = self.dec2(
            torch.cat([F.interpolate(dec3, enc2.size()[-2:], mode='bilinear', align_corners=True), enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        final = self.final(dec1)
        return final


if __name__=="__main__":
    x =torch.randn(10,2,64,64)
    sample = torch.randn(10,2,64,64)
    incontextmodel = mainUNet(sample_num=2)
    model_dict = incontextmodel.state_dict()
    sampleencoder_dict = incontextmodel.samplesEncoder.state_dict()
    unet = UNet(in_channels=1,out_channels=1)
    unet_dict = unet.state_dict()
    # for name,para in model_dict.items():
    #     print(name)
    # for name,_ in sampleencoder_dict.items():
    #     print(name)
    # for name,_ in unet_dict.items():
    #     print(name)

    # print(incontextmodel.samplesEncoder.state_dict())
    out =incontextmodel(sample,x)
    print(out.shape)