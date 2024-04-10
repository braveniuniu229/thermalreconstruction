import torch
import os
from torch.utils.data import DataLoader
from dataset.shallowdecoder.dataset_type_50 import dataset_test
from model.shallowdecodermlp import shallow_decoder
from utils.visualization import plot_single
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=2)
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
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(2):
                current_file_name = os.path.join(exp, f"image_{i}.png")
                plot_single(labels[i],outputs[i],current_file_name)
                break
            break



            # Set common color limits for the output and real images for better comparison


if __name__=="__main__":
    eval(model)
