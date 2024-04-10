import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
import time
import os
from dataset.maskeddataset import dataset_train,dataset_test
from model.unetseries import UNet
import csv
import wandb
import utils.argsbasic
wandb.init(
    project='masked_unet',
    config={
        'lr':0.001,
        'arch':'maskedunet',
        'interval':5,       #进行eval的间隔轮数
        'weightdecay':1e-4,
        'dataset':'typeNum_1',
        'mask_ratio':0.7,
        'epochs':300,
        'tag':'baseline',
        'lr_decay_epoch':100,
        'batch_size':8,
        'dropout':False
    }
)
model = UNet(in_channels=1,out_channels=1)
args = wandb.config
train_loader = DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True)
test_loader = DataLoader(dataset_test,batch_size=64,shuffle=False)

file = args.arch +'_'+args.dataset+'maskratio_'+str(args.mask_ratio)
"""这里每次都要修改成训练的model"""
  #这里修改成训练的断点

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
criterion = nn.L1Loss()  # 假设使用均方误差损失

# 记录文件和检查点路径
if not os.path.exists(f'checkpoint/{file}'):
    os.makedirs(f'checkpoint/{file}')
checkpoint_save_path = os.path.join('../checkpoint', file)
if not os.path.exists(f'trainingResult/{file}'):
    os.makedirs(f'trainingResult/{file}')

def write_to_csv(file_path, epoch, loss):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def save_checkpoint(epoch, iteration, model, optimizer, loss, is_best=False):
    state = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bestloss': loss
    }
    # filename = f"{checkpoint_dir}/checkpoint_epoch_{epoch}_iter_{iteration}.pth"
    # torch.save(state, filename)
    if is_best:
        torch.save(state, f"{checkpoint_save_path}/checkpoint_best.pth")
def load_checkpoint(checkpoint_path, model, optimizer):
    """加载 checkpoint."""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1  # 假设我们保存的 epoch 是已完成的，所以从下一个开始
        best_loss = checkpoint['bestloss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint '{checkpoint_save_path}' (epoch {checkpoint['epoch']})")
        return start_epoch, best_loss
    else:
        print(f"No checkpoint found at '{checkpoint_save_path}'")
        return 0, float('inf')  # 如果没有找到 checkpoint，从头开始
def train(epoch):

    model.train()
    start_time = time.time()

    total_loss = 0  # 用于累积每个 epoch 的总损失
    pbar = tqdm.tqdm(total=len(train_loader), desc=f"Training Epoch {epoch}", leave=True,colour='white')
    for iteration, (masked_labels,labels) in enumerate(train_loader):
        masked_labels,labels = masked_labels,labels
        masked_labels,labels = masked_labels.to(device).to(torch.float32),labels.to(device).to(torch.float32)
        masked_labels = masked_labels.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(masked_labels)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()


        total_loss += loss.item()

        # if (iteration+1)%200 == 0:
        #     iter_loss = loss.item()
        #     write_to_csv(f'experiresult/{file}/train_log_iter.csv', epoch * len(train_loader) + iteration, iter_loss)
        #     """xiugai"""
        pbar.set_description(f"Training Epoch {epoch} [{iteration}/{len(train_loader)}]")
        pbar.update(1)# 更新进度条
    pbar.close()
    average_loss = total_loss / len(train_loader)  # 计算平均损失
    epoch_time = time.time() - start_time
    write_to_csv(f'trainingResult/{file}/train_log.csv', epoch, average_loss)
    wandb.log({"average_loss_train":average_loss,'epoch_usedtime':epoch_time })
    scheduler.step()
    # githubllllkk
    """这里要修改模型的路径"""
    print("epoch:",epoch,'\n',"loss:",average_loss,'\n',"epoch time used:",epoch_time)
def validate(epoch, best_loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(test_loader), desc=f"Training Epoch {epoch}", leave=True, colour='white')
        for iteration, (masked_labels, labels) in enumerate(test_loader):
            masked_labels, labels = masked_labels, labels
            masked_labels, labels = masked_labels.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            masked_labels = masked_labels.unsqueeze(1)
            outputs = model(masked_labels)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pbar.update(1)
    pbar.close()
    avg_loss = total_loss / len(test_loader)
    write_to_csv(f'trainingResult/{file}/val_log.csv', epoch, avg_loss)
    wandb.log({"average_loss_val": avg_loss})
    # 如果是最好的损失，保存为 best checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint(epoch, 'best', model, optimizer, avg_loss, is_best=True)
        print("产生了新的最优结果，模型已经保存!")
    return best_loss

best_loss = float('inf')
num_epochs = args.epochs # 假设训练 1500 个 epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if __name__ == "__main__":
    print(f'The model has {count_parameters(model):,} trainable parameters')
    print(device)
    checkpoint_path = f"{checkpoint_save_path}/checkpoint_best.pth"
    start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer)
    for epoch in tqdm.trange(start_epoch, num_epochs):
        train(epoch)
        if epoch%args.interval==0:
            best_loss = validate(epoch, best_loss)
