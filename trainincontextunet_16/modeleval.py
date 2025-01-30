import torch
import torch.nn as nn
from model.incontextunet import mainUNet
from utils.visualization import plot_pres,plot_truth,plot_error,plot_locations,plot_error2
import os
from torch.utils.data import DataLoader
from dataset.incontextunetdataset import dataset_test
import numpy as np
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=5)
ckpt = torch.load('../checkpoint/incontextunet_typeNum_10000basedratio0.7finetune_with_diff_lr/checkpoint_best.pth')
model_dict = ckpt['model_state_dict']
model = mainUNet(sample_num=1)
model.load_state_dict(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()
type_num = 'ood1t'
exp = os.path.join('figure',type_num)
positions = np.array([[8,8],
 [23,8],
 [39,8],
 [55,8],
 [8,23],
 [23,23],
 [39,23],
 [55,23],
 [8,39],
 [23,39],
 [39,39],
 [55,39],
 [8,55],
 [23,55],
 [39,55],
 [55,55]])
if not os.path.exists(exp):
    os.makedirs(exp)
def eval(model):
    model.to(device)
    model.eval()

    with torch.no_grad():
        for iteration, (com, labels,samples) in enumerate(test_loader):
            com, labels, samples = com, labels, samples
            com, labels, samples = com.to(device).to(torch.float32), labels.to(device).to(torch.float32), samples.to(
                device).to(torch.float32)
            outputs = model(com,samples)
            outputs = outputs.squeeze(1)
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            error = abs(outputs-labels)

            for i in range(5):

                pre_pth = os.path.join(exp,f'pre{i}.png')
                tru_pth = os.path.join(exp, f'tru{i}.png')
                err_pth = os.path.join(exp, f'err{i}.png')
                plot_truth(outputs[i],pre_pth)
                plot_truth(labels[i],tru_pth)
                plot_error2(error[i],err_pth)
            break




            # Set common color limits for the output and real images for better comparison


if __name__=="__main__":
    eval(model)
