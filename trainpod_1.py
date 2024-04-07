import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
import time
import os
from dataset.shallowdecoder.dataset_type_1 import dataset_test
import csv
import numpy as np
import wandb
from model.gappypod import GappyPod
import utils.argsbasic
wandb.init(
    project='podbasic',
    config={
        'lr':0.01,
        'arch':'podBaseline',
        'weightdecay':1e-4,
        'dataset':'typeNum_1',
        'epochs':1000,
        'tag':'baseline',
        'lr_decay_epoch':100,
        'batch_size':8000,
        'dropout':False,
        'num':5
    }
)
path = './data/Heat_Types1_source4_number100000fixed_normalized.npz'
origin_data = np.load(path)
pod_data = origin_data['T'][0,:80000,:]
pod_data = pod_data.reshape(-1,64,64)


gappy_pod = GappyPod(
    data = pod_data,map_size=pod_data.shape[-2:],n_components=50,positions = np.array([[8,8],
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
)
test_loader = DataLoader(dataset_test,batch_size=1,shuffle=False)
args = wandb.config


file = args.arch +'_'+args.dataset+str(args.num)


criterion = nn.L1Loss()  # 假设使用均方误差损失

# 记录文件和检查点路径

if not os.path.exists(f'trainingResult/{file}'):
    os.makedirs(f'trainingResult/{file}')



def write_to_csv(file_path, loss):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([loss])




def test():


    start_time = time.time()
    total_loss = 0
     # 用于累积每个 epoch 的总损失
    pbar = tqdm.tqdm(total=len(test_loader), leave=True,colour='white')
    for iteration, (data,labels) in enumerate(test_loader):
        N,_ = data.shape
        data,labels = data,labels
        data,labels = data.to(device).to(torch.float32),labels.to(device).to(torch.float32)
        pre = gappy_pod.reconstruct(data)
        pre = pre.squeeze(1)
        loss = criterion(pre,labels)
        total_loss += loss.item()



        # if (iteration+1)%200 == 0:
        #     iter_loss = loss.item()
        #     write_to_csv(f'experiresult/{file}/train_log_iter.csv', epoch * len(train_loader) + iteration, iter_loss)
        #     """xiugai"""

        pbar.update(1)# 更新进度条

    pbar.close()
    average_loss = total_loss / len(test_loader)

    epoch_time = time.time() - start_time
    write_to_csv(f'trainingResult/{file}/eval_log.csv',  average_loss)
    wandb.log({"average_loss_train":average_loss,'epoch_usedtime':epoch_time })
    # githubllllkk
    """这里要修改模型的路径"""
    print("loss:",average_loss,'\n',"epoch time used:",epoch_time)






if __name__ == "__main__":
    device = torch.device('cuda')
    test()
