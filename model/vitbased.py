import torch
import torch.nn as nn
from vit import SimpleViT
class vitbased(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1,dtype=torch.float32)
        self.vit = SimpleViT(image_size=(64, 64), patch_size=(4, 4), num_classes=4096, dim = 256, depth = 7, heads = 8, mlp_dim = 4)

    def forward(self, x):

        x_mid = self.conv1(x)
        x_final = self.vit(x_mid)
        return x_final

# 创建模型实例
if __name__ == '__main__':
    i = torch.randn(6,1,64,64)
    model = vitbased()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i = i.to(device)
    model.to(device)
    o = model(i)
    print(o.shape)