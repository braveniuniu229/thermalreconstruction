import torch
import torch.nn as nn
from model.incontextunet import mainUNet
from utils.visualization import plot3x1
from dataset.incontextunetdataset import dataset_test
import os
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model.unetseries import UNet
from dataset.vordataset import dataset_train,dataset_test
from mpl_toolkits.axes_grid1 import make_axes_locatable
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=300)
model = mainUNet(sample_num=1)
model2 = UNet(in_channels=2,out_channels=1)
model_ckpt = torch.load('./checkpoint/voronoiUnetBaseline_typeNum_10000/checkpoint_best.pth')
model_state_dict = model_ckpt['model_state_dict']
model2.load_state_dict(model_state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()

dataorigin = np.load('./data/Heat_Types10_source4_number30fixed_normalized.npz')
mean = dataorigin['m']
stdvar = dataorigin['v']
def eval(model):
    model.to(device)
    model.eval()
    total_loss =0
    with torch.no_grad():
        for iteration, (com, labels) in enumerate(test_loader):
            com, labels= com, labels
            com, labels= com.to(device).to(torch.float32), labels.to(device).to(torch.float32)

            outputs = model(com)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            total_loss +=loss
            outputs_np = (outputs.cpu().numpy() + mean) * stdvar
            labels_np = (labels.cpu().numpy() + mean) * stdvar

            # 计算逆标准化后的 loss，这里不应该用原始的 stdvar 直接乘，因为它是全局标量
            # 如果 stdvar 是单一值，确保它转换为 tensor 以匹配维度
            stdvar_tensor = torch.tensor(stdvar, device=device, dtype=torch.float32)
            loss_scaled = (outputs - labels) * stdvar_tensor

            # 从 CUDA 转移回 CPU，然后转换为 numpy
            loss_np = loss_scaled.cpu().numpy()
            loss_ref = (loss_np / labels).cpu().numpy()

            fig, axis = plt.subplots(5, 4, figsize=(12, 10), dpi=100)  # Adjust figsize to give more space
            fig.suptitle('Eval Result on OOD Situation', fontsize=16, weight='bold')

            # Set common color limits for the output and real images for better comparison
            color_limit_output_real = [min(outputs_np.min(), labels_np.min()), max(outputs_np.max(), labels_np.max())]

            for i in range(5):
                for j in range(4):
                    ax = axis[i, j]
                    if j == 0:  # Output
                        im = ax.imshow(outputs_np[i], cmap='coolwarm', vmin=color_limit_output_real[0],
                                       vmax=color_limit_output_real[1])
                        if i == 0: ax.set_title('Output', fontsize=10, weight='bold')
                    elif j == 1:  # Real
                        im = ax.imshow(labels_np[i], cmap='coolwarm', vmin=color_limit_output_real[0],
                                       vmax=color_limit_output_real[1])
                        if i == 0: ax.set_title('Real', fontsize=10, weight='bold')
                    elif j == 2:  # Loss
                        # Use diverging colormap centered around zero
                        im = ax.imshow(loss_np[i], cmap='seismic', vmin=-np.abs(loss_np).max(),
                                       vmax=np.abs(loss_np).max())
                        if i == 0: ax.set_title('Loss', fontsize=10, weight='bold')
                    elif j == 3:  # Loss Ref
                        # Adjust the scale for loss reference if it's too small/large
                        scale_factor = np.abs(loss_ref).max() if np.abs(loss_ref).max() != 0 else 1
                        im = ax.imshow(loss_ref[i], cmap='seismic', vmin=-scale_factor, vmax=scale_factor)
                        if i == 0: ax.set_title('Loss Ref', fontsize=10, weight='bold')

                    # Hide axes and add colorbar to each subplot
                    ax.axis('off')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout
            plt.show()



    avg_loss = total_loss / len(test_loader)
    print((avg_loss))
if __name__=="__main__":
    eval(model2)
