import torch
import os
from torch.utils.data import DataLoader
from dataset.vordataset import dataset_test
from model.voronoiCNNoriginal import VoronoiCNN
from utils.visualization import plot_error2,plot_pres
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=5)
ckpt = torch.load('./checkpoint/voronoi_CNN_typeNum_10000batchsize_8/checkpoint_best.pth')
model_dict = ckpt['model_state_dict']
model = VoronoiCNN()
model.load_state_dict(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
type_num = 'ood1t'
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
            error = abs(outputs-labels)

            for i in range(5):
                err_pth = os.path.join(exp, f'err{i}.png')
                pre_pth = os.path.join(exp, f'pre{i}.png')
                # plot_pres(outputs[i], pre_pth)
                plot_error2(error[i], err_pth)

            break



            # Set common color limits for the output and real images for better comparison


if __name__=="__main__":
    eval(model)
