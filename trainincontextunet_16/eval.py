import torch
import torch.nn as nn
from model.incontextunet import mainUNet
from utils.visualization import plot_single
import os
from torch.utils.data import DataLoader
from dataset.incontextunetdataset import dataset_test
import tqdm

# 加载模型
test_loader = DataLoader(dataset_test, batch_size=5)
ckpt = torch.load('../checkpoint/incontextunet_typeNum_10000basedratio0.7finetune_with_diff_lr/checkpoint_best.pth')
model_dict = ckpt['model_state_dict']
model = mainUNet(sample_num=1)
model.load_state_dict(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()


def eval(model):
    model.to(device)
    model.eval()
    total_loss = 0
    total_max = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(test_loader), leave=True, colour='white')
        for iteration, (com, labels, samples) in enumerate(test_loader):
            com, labels, samples = com, labels, samples
            com, labels, samples = com.to(device).to(torch.float32), labels.to(device).to(torch.float32), samples.to(
                device).to(torch.float32)
            batch,_,_ = labels.shape
            outputs = model(com, samples)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            max_loss = torch.sum(torch.max(torch.abs(outputs - labels).flatten(1), dim=1)[0]).item() / batch
            total_loss += loss.item()
            total_max += max_loss
            pbar.update(1)
        pbar.close()
        avg_loss = total_loss / len(test_loader)
        avg_max = total_max/len(test_loader)
        print(avg_loss)
        print(avg_max)



            # Set common color limits for the output and real images for better comparison


if __name__ == "__main__":
    eval(model)
