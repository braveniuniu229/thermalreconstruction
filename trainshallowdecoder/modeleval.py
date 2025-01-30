import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset.shallowdecoder.dataset_type_50 import dataset_test
from model.shallowdecodermlp import shallow_decoder
from utils.visualization import plot_error2,plot_pres,plot_single,plot_truth,plot_locations
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=3000)
ckpt = torch.load('../checkpoint/shallowdecoderBaseline_shallowdecodermlp_typeNum_100005/checkpoint_best.pth')
model_dict = ckpt['model_state_dict']
model = shallow_decoder(n_sensors=16,outputlayer_size=4096)
model.load_state_dict(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
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
        for iteration, (data,labels) in enumerate(test_loader):
            data, labels = data, labels
            data, labels = data.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            outputs = model(data)
            outputs = outputs.reshape(-1,64,64)
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            errors = abs(outputs-labels)

            for i in range(5):
                err_pth = os.path.join(exp, f'err{i}.png')
                pre_pth = os.path.join(exp, f'pre{i}.png')
                tru_pth = os.path.join(exp, f'tru{i}.png')
                plot_locations(positions,labels[i], tru_pth)
                plot_truth(outputs[i],pre_pth)
                plot_error2(errors[i], err_pth)
            break





            # Set common color limits for the output and real images for better comparison


if __name__=="__main__":
    eval(model)
