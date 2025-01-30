import torch
import torch.nn as nn
from utils.visualization import plot3x1beta
import os
from torch.utils.data import DataLoader
from dataset.maskeddataset import dataset_test
import numpy as np

from model.unetseries import UNet
from mpl_toolkits.axes_grid1 import make_axes_locatable
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=400)
model = UNet(in_channels=1,out_channels=1)
model_ckpt = torch.load('./checkpoint/maskedunet_typeNum_100000.85/checkpoint_best.pth')
model_state_dict = model_ckpt['model_state_dict']
model.load_state_dict(model_state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()
fileexp = './figures/maskedunet'
dataorigin = np.load('./data/Heat_Types10_source4_number30fixed_normalized.npz')
mean = dataorigin['m']
stdvar = dataorigin['v']
maskratio = 'ratio0.85'
filepath =os.path.join(fileexp,maskratio)
def eval(model):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for iteration, (masked_labels,labels) in enumerate(test_loader):
            masked_labels, labels = masked_labels, labels
            masked_labels, labels = masked_labels.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            masked_labels = masked_labels.unsqueeze(1)
            outputs = model(masked_labels)
            outputs = outputs.squeeze(1)
            masked_labels = masked_labels.squeeze(1)
            masked_labels = masked_labels.cpu().numpy()

            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for i in range(400):
                current_file_name = os.path.join(filepath, f"image_{i}.png")
                plot3x1beta(masked_labels[i],outputs_np[i],labels_np[i],current_file_name)
                break





if __name__=="__main__":
    eval(model)
