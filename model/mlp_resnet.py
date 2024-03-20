import torch.nn as nn
import torch
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)




        out = self.relu(out)
        out = self.down(out)

        return out


class Incontextmlp(nn.Module):
    def __init__(self, exp_num):
        super(Incontextmlp, self).__init__()
        self.in_channel = exp_num
        self.dropout = nn.Dropout(0.1)
        self.gelu =nn.GELU()
        # Initial convolutional layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Residual blocks for feature extraction
        self.resblock1 = ResidualBlock(16, 32)
        self.resfc1 = nn.Linear(2048,128)
        self.resblock2 = ResidualBlock(32, 64)
        self.resfc2 = nn.Linear(1024, 256)
        self.resblock3 = ResidualBlock(64, 128)
        self.resfc3 = nn.Linear(512, 512)

        # MLP for query points
        self.embedding_query = nn.Linear(16, 128)
        self.bn_query = nn.BatchNorm1d(128)


        # Additional BN layers for MLP
        self.fc2 = nn.Linear(256, 500)  # For conv1 features
        # self.bn2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(756, 1200)  # For conv2 features
        # self.bn3 = nn.BatchNorm1d(1200)
        self.fc4 = nn.Linear(1712, 3600)  # For conv3 features
        # self.bn4 = nn.BatchNorm1d(3600)
        self.fc5 = nn.Linear(3600, 4096)

    def forward(self,query,samples):
        # Process the query through the MLP
        query_features = self.embedding_query(query)
        query_features = F.relu(self.bn_query(query_features))




        # Initial convolution
        x = self.initial_conv(samples)

        # Residual blocks
        x = self.resblock1(x)
        x1 = x.view(x.size(0),-1)
        x1 = self.gelu(self.resfc1(x1))  #5, 32, 16, 16   256
        x = self.resblock2(x)
        x2 = x.view(x.shape[0],-1)
        x2 = self.gelu(self.resfc2(x2))  #5, 64, 8, 8    512
        x = self.resblock3(x)
        x3 = x.view(x.shape[0], -1)
        x3 = self.gelu(self.resfc3(x3)) #5, 128, 4, 4    1024

        # Combine query features with conv features at different depths
        x = torch.cat([query_features, x1], dim=-1)  #256
        x = self.dropout(x)
        x = self.fc2(x)      #600
        # x = self.bn2(x)
        x = self.gelu(x)

        x = torch.cat([x, x2], dim=-1)  #856
        x = self.dropout(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = self.gelu(x)        #1200

        x = torch.cat([x, x3], dim=-1)   #1712
        x = self.dropout(x)
        x = self.fc4(x)
        # x = self.bn4(x)
        x = self.gelu(x)  #3600

        x = self.dropout(x)
        # # Final layer for the prediction
        output = self.fc5(x)        #4096

        return output
if __name__ == "__main__":
    def countparam(m:nn.Module):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    model = Incontextmlp(4)
    query = torch.rand(5,16)
    example = torch.rand(5,4,64,64)
    out = model(query,example)
    print(out.shape,countparam(model))
