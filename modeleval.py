import torch
import torch.nn as nn
from model.incontextunet import mainUNet
from dataset.incontextunetdataset import dataset_test
import os
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=300)
model = mainUNet(sample_num=1)
model_ckpt = torch.load('path')
model_state_dict = model_ckpt['model_state_dict']
model.load_state_dict(model_state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()

dataorigin = np.load('./data/Heat_Types10000_source4_number10fixed_normalized.npz')
mean = dataorigin['m']
stdvar = dataorigin['v']
def eval(model):
    model.eval()
    total_loss =0
    with torch.no_grad():
        for iteration, (com, labels, samples) in enumerate(test_loader):
            com, labels, samples = com, labels, samples
            com, labels, samples = com.to(device).to(torch.float32), labels.to(device).to(torch.float32), samples.to(
                device).to(torch.float32)
            outputs = model(com, samples)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            outputs_np = outputs.detach().numpy()
            outputs_np = (outputs_np+mean)*stdvar
            labels_np = labels.detach().numpy()
            labels_np = (labels_np+mean)*stdvar
            total_loss += loss.item()
            loss = (outputs-labels)*stdvar
            loss_ref = (outputs-labels)/labels

            fig, axis = plt.subplots(5, 4, figsize=(10, 8), dpi=200)  # Smaller figsize
            plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce the space between images
            fig.suptitle('eval result on ood situation', fontsize=16)
            for i in range(5):  # 6 rows of images
                    ax1 = axis[i, 1]  # Even index for masked images
                    ax1.imshow(outputs_np[i], vmin=-2, vmax=2, cmap='bwr')
                    ax1.set_title('output', fontsize=8)
                    ax1.axis('off')
                    ax2 = axis[i, 2]  # Even index for masked images
                    ax2.imshow(labels_np[i], vmin=-2, vmax=2, cmap='bwr')
                    ax2.set_title('real', fontsize=8)
                    ax2.axis('off')
                    ax3 = axis[i, 3]  # Even index for masked images
                    ax3.imshow(loss[i], vmin=-2, vmax=2, cmap='bwr')
                    ax3.set_title('loss', fontsize=8)
                    ax3.axis('off')
                    ax4 = axis[i, 4]  # Even index for masked images
                    ax4.imshow(loss_ref[4], vmin=-2, vmax=2, cmap='bwr')
                    ax4.set_title('loss_ref', fontsize=8)
                    ax4.axis('off')
            plt.show()



    avg_loss = total_loss / len(test_loader)
    print((avg_loss))
if __name__=="__main__":
    eval()
