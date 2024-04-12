import torch
import torch.nn as nn
from model.unetseries import UNet
from torch.utils.data import DataLoader
from dataset.vordataset import dataset_test
import tqdm
 #加载模型
test_loader = DataLoader(dataset_test,batch_size=5)
ckpt = torch.load('../checkpoint/voronoiUnetBaseline_typeNum_10000/checkpoint_best.pth')
model_dict = ckpt['model_state_dict']
model = UNet(in_channels=2,out_channels=1)
model.load_state_dict(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()

def eval(model):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0
        pbar = tqdm.tqdm(total=len(test_loader),  leave=True, colour='white')
        for iteration, (data,labels) in enumerate(test_loader):
            data, labels = data, labels
            data, labels = data.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            outputs = model(data)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            total_loss+=loss.item()
            pbar.update(1)  # 更新进度条
        pbar.close()
        average_loss = total_loss / len(test_loader)
        print(average_loss)





            # Set common color limits for the output and real images for better comparison


if __name__=="__main__":
    eval(model)
