import torch
from torch.utils.data import DataLoader
from dataset.shallowdecoder.dataset_type_50 import dataset_test
from model.shallowdecodermlp import shallow_decoder
import tqdm
import torch.nn as nn
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=2)
ckpt = torch.load('../checkpoint/shallowdecoderBaseline_shallowdecodermlp_typeNum_100005/checkpoint_best.pth')
model_dict = ckpt['model_state_dict']
model = shallow_decoder(n_sensors=16,outputlayer_size=4096)
model.load_state_dict(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()

def eval(model):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0 # 用于累积每个 epoch 的总损失
        Max_loss = 0
        pbar = tqdm.tqdm(total=len(test_loader), leave=True, colour='white')
        for iteration, (data,labels) in enumerate(test_loader):
            data, labels = data, labels
            data, labels = data.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            batch,_,_ = labels.shape
            outputs = model(data)
            labels = labels.view(labels.shape[0], -1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            max_loss = torch.sum(torch.max(torch.abs(outputs - labels).flatten(1), dim=1)[0]).item() / batch
            Max_loss += max_loss
            pbar.update(1)
        pbar.close()
        average_loss = total_loss / len(test_loader)
        average_loss_max = Max_loss/len(test_loader)
        print(average_loss)
        print(average_loss_max)



            # Set common color limits for the output and real images for better comparison


if __name__=="__main__":
    eval(model)
