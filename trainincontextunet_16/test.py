import torch
import torch.nn as nn
from model.incontextunet import mainUNet

model = mainUNet(sample_num=1)
pretrainedunet_ckpt = torch.load('../checkpoint/maskedunet_typeNum_500maskratio_0.7/checkpoint_best.pth')
pretrainedunet_dict = pretrainedunet_ckpt['model_state_dict']
model_dict = model.state_dict()

# 创建新的参数字典，将前缀添加到键中
load_dict = {}
for k, v in pretrainedunet_dict.items():
    new_k = 'samplesEncoder.' + k
    if new_k in model_dict:
        load_dict[new_k] = v

# 更新模型的状态字典
model_dict.update(load_dict)

for k,v in load_dict.items():
    print(k,v)
