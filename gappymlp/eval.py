import torch
from torch.utils.data import DataLoader
from dataset.shallowdecoder.dataset_type_50 import dataset_test
from model.shallowdecodermlp import shallow_decoder
import tqdm
import torch.nn as nn
import numpy as np
from model.gappypod import GappyPodWeight
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=2)
ckpt = torch.load('../checkpoint/shallowdecoderBaseline2_typeNum_15/checkpoint_best.pth')
model_dict = ckpt['model_state_dict']
model = shallow_decoder(n_sensors=16,outputlayer_size=4096)
model.load_state_dict(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()


path = '../data/Heat_Types1_source4_number100000fixed_normalized.npz'
origin_data = np.load(path)
pod_data = origin_data['T'][:,:80000,:]
pod_data = pod_data.reshape(-1,64,64)
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

gappy_pod = GappyPodWeight(
    data = pod_data,map_size=pod_data.shape[-2:],n_components=50,observe_weight=50,positions = positions)
def eval(model):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0  # 用于累积每个 epoch 的总损失
        Max_loss = 0
        pbar = tqdm.tqdm(total=len(test_loader), leave=True, colour='white')
        for iteration, (data,labels) in enumerate(test_loader):
            batch,_,_=labels.shape
            data, labels = data, labels
            data, labels = data.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            outputs = model(data)
            outputs = outputs.reshape(-1,64,64)
            outputs = gappy_pod.reconstruct(outputs, data, weight=torch.ones_like(outputs))
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            outputs = outputs.reshape(-1,4096)
            labels = labels.reshape(-1,4096)
            max_loss = torch.sum(torch.max(torch.abs(outputs-labels),dim=1)[0]).item()/batch
            total_loss += loss.item()
            Max_loss +=max_loss
            pbar.update(1)
        pbar.close()
        average_loss = total_loss / len(test_loader)
        average_Max_loss = Max_loss/len(test_loader)
        print(average_loss)
        print(average_Max_loss)



            # Set common color limits for the output and real images for better comparison


if __name__=="__main__":
    eval(model)
