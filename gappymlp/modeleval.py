import torch
import os
from torch.utils.data import DataLoader
from dataset.shallowdecoder.dataset_type_50 import dataset_test
from model.shallowdecodermlp import shallow_decoder
from model.gappypod import GappyPodWeight
from utils.visualization import plot_single,plot_locations
import numpy as np
path = '../data/Heat_Types50_source4_number2000fixed_normalized.npz'
origin_data = np.load(path)
pod_data = origin_data['T'][:,:1600,:]
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
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=5)
ckpt = torch.load('../checkpoint/shallowdecoderBaseline_shallowdecodermlp_typeNum_505/checkpoint_best.pth')
model_dict = ckpt['model_state_dict']
model = shallow_decoder(n_sensors=16,outputlayer_size=4096)
model.load_state_dict(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
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
            outputs = outputs.reshape(-1,64,64)
            outputs = gappy_pod.reconstruct(outputs, data, weight=torch.ones_like(outputs))
            outputs = outputs.squeeze(1)
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(5):
                current_file_name = os.path.join(exp, f"image_{i}.png")
                current_file_name2 = os.path.join(exp, f"sensors_location.png")
                plot_single(labels[i],outputs[i],current_file_name)
                plot_locations(positions,labels[i],current_file_name2)
                break
            break



            # Set common color limits for the output and real images for better comparison


if __name__=="__main__":
    eval(model)
