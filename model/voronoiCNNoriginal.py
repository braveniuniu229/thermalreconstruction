import torch
import torch.nn as nn

class VoronoiCNN(nn.Module):
    def __init__(self):
        super(VoronoiCNN, self).__init__()
        # 定义模型的层
        self.conv1 = nn.Conv2d(2, 48, kernel_size=(7, 7), padding='same')
        self.conv2 = nn.Conv2d(48, 48, kernel_size=(7, 7), padding='same')
        self.conv3 = nn.Conv2d(48, 48, kernel_size=(7, 7), padding='same')
        self.conv4 = nn.Conv2d(48, 48, kernel_size=(7, 7), padding='same')
        self.conv5 = nn.Conv2d(48, 48, kernel_size=(7, 7), padding='same')
        self.conv6 = nn.Conv2d(48, 48, kernel_size=(7, 7), padding='same')
        self.conv7 = nn.Conv2d(48, 48, kernel_size=(7, 7), padding='same')
        self.conv_final = nn.Conv2d(48, 1, kernel_size=(3, 3), padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        # 描述数据如何通过网络
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.conv_final(x)
        return x

# 实例化模型
model = VoronoiCNN()

# # 编译模型
#
# x = torch.randn(10,2,64,64)  #out (b,1,64,64)
# out = model(x) #
# out = out.squeeze(1)
# print(out.shape)