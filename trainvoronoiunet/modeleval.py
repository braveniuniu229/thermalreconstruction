import torch
import torch.nn as nn
from model.unetseries import UNet
from utils.visualization import plot_single
import os
from torch.utils.data import DataLoader
from dataset.vordataset import dataset_test
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=5)
ckpt = torch.load('../checkpoint/voronoiUnetBaseline_typeNum_50/checkpoint_best.pth')
model_dict = ckpt['model_state_dict']
model = UNet(in_channels=2,out_channels=1)
model.load_state_dict(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()
type_num = 'num_50'
exp = os.path.join('figure',type_num)
if not os.path.exists(exp):
    os.makedirs(exp)
def eval(model):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for iteration, (data,labels) in enumerate(test_loader):
            data, labels = data, labels
            data, labels = data.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            outputs = model(data)
            outputs = outputs.squeeze(1)
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(5):
                current_file_name = os.path.join(exp, f"image_{i}.png")
                plot_single(labels[i],outputs[i],current_file_name)

            break



            # Set common color limits for the output and real images for better comparison


if __name__=="__main__":
    eval(model)
