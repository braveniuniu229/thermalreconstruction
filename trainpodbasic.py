import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
import time
import os
from dataset.shallowdecoder.dataset_type_100 import dataset_train,dataset_test
from model.shallowdecodermlp import shallow_decoder
import csv
import numpy as np
import wandb
from model.gappypod import GappyPodWeight
import utils.argsbasic
wandb.init(
    project='podnn',
    config={
        'lr':0.01,
        'arch':'podnn_baseline',
        'config':[16,60,65,300,4096],
        'weightdecay':1e-4,
        'dataset':'typeNum_100',
        'epochs':1000,
        'tag':'baseline',
        'lr_decay_epoch':100,
        'batch_size':8000,
        'dropout':False,
        'num':5
    }
)
path = ''
origin_data = np.load("path")
pod_data = origin_data['T']
model = shallow_decoder(n_sensors=16,outputlayer_size=4096)
gappy_pod = GappyPodWeight(
    data = pod_data,map_size=pod_data.shape[-2:],n_components=50,observe_weight=50
)
test_loader = DataLoader(dataset_test,batch_size=10000,shuffle=False)
args = wandb.config

file = args.arch +'_'+args.dataset+str(args.num)


criterion = nn.L1Loss()  # 假设使用均方误差损失

# 记录文件和检查点路径
if not os.path.exists(f'checkpoint/{file}'):
    os.makedirs(f'checkpoint/{file}')
checkpoint_save_path = os.path.join('./checkpoint', file)
if not os.path.exists(f'trainingResult/{file}'):
    os.makedirs(f'trainingResult/{file}')



def write_to_csv(file_path, loss):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([loss])



def load_checkpoint(checkpoint_path, model):
    """加载 checkpoint."""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)


        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded checkpoint '{checkpoint_save_path}' (epoch {checkpoint['epoch']})")
        return
    else:
        print(f"No checkpoint found at '{checkpoint_save_path}'")
        return
def test():

    model.eval()
    start_time = time.time()

    total_loss = 0  # 用于累积每个 epoch 的总损失
    pbar = tqdm.tqdm(total=len(test_loader), leave=True,colour='white')
    for iteration, (data,labels) in enumerate(test_loader):
        data,labels = data,labels
        data,labels = data.to(device).to(torch.float32),labels.to(device).to(torch.float32)
        pre = model(data)
        pre =pre.view(-1,64,64)
        pre = gappy_pod.reconstruct(pre, data, weight=torch.ones_like(pre))
        loss = criterion(pre,labels)
        total_loss += loss.item()

        # if (iteration+1)%200 == 0:
        #     iter_loss = loss.item()
        #     write_to_csv(f'experiresult/{file}/train_log_iter.csv', epoch * len(train_loader) + iteration, iter_loss)
        #     """xiugai"""

        pbar.update(1)# 更新进度条

    pbar.close()
    average_loss = total_loss / len(test_loader)  # 计算平均损失
    epoch_time = time.time() - start_time
    write_to_csv(f'trainingResult/{file}/eval_log.csv',  average_loss)
    wandb.log({"average_loss_train":average_loss,'epoch_usedtime':epoch_time })
    # githubllllkk
    """这里要修改模型的路径"""
    print("loss:",average_loss,'\n',"epoch time used:",epoch_time)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if __name__ == "__main__":

    print(device)
    checkpoint_path = f"{checkpoint_save_path}/checkpoint_best.pth"
    load_checkpoint(checkpoint_path, model)
    test()
