import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata
# 实际上并不需要保存测点的数据，直接生成测点坐标从T中生成测点数据就可以，这样测点的坐标每次都可以自由选择，
# 无论是做voronoi划分还是直接一维输入都可以，而且可以保证两次的坐标索引是相同的很自由

class thermalDataset_vor(Dataset):
    def __init__(self, labels, exp_num,train=True, train_ratio=0.8):
        """
        Custom dataset initializer.
        :param data: The data (e.g., 'T' from your dataset)
        :param labels: The labels or outputs corresponding to the data (e.g., 'o')
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        """


        self.exp_num = exp_num
        self.labels = labels.reshape(labels.shape[0],labels.shape[1],64,64)
        self.train = train
        # 这里是生成坐标索引的现在就是均匀划分
        x = np.linspace(8, 55, 4, dtype=int)
        y = np.linspace(8, 55, 4, dtype=int)
        xv, yv = np.meshgrid(x, y)
        self.points = np.vstack([xv.ravel(), yv.ravel()]).T
        # Determine split sizes
        num_samples_per_class = labels.shape[1]
        num_train = int(num_samples_per_class * train_ratio)
        self.indices = np.arange(num_samples_per_class)

        if self.train:
            self.indices = self.indices[:num_train]
        else:
            self.indices = self.indices[num_train:]

    def __len__(self):
        return len(self.indices) * self.labels.shape[0]  # Total number of samples

    def __getitem__(self, idx):

        class_idx = idx // len(self.indices)
        sample_idx_in_class = self.indices[idx % len(self.indices)]
        sample_label = self.labels[class_idx, sample_idx_in_class]
        #进行划分的测点值
        values = sample_label[self.points[:, 0], self.points[:, 1]]
        # 最后进行这个插值划分的网格
        grid_x, grid_y = np.meshgrid(np.linspace(0, 63, 64), np.linspace(0, 63, 64))
        voronoidata = griddata(self.points, values, (grid_x, grid_y), method='nearest')
        mask = np.zeros_like(sample_label, dtype=np.float32)
        mask[self.points[:, 0], self.points[:, 1]] = 1
        # 新建一个轴用于拼接
        voronoidata_exp = np.expand_dims(voronoidata, axis=0)
        mask_exp = np.expand_dims(mask, axis=0)
        combined_data = np.concatenate([voronoidata_exp, mask_exp], axis=0)



        # 对选取的问答示例进行拼接
        final_prompt = np.zeros((self.exp_num,2,2,64,64))
        indices_except_target = np.setdiff1d(self.indices, sample_idx_in_class)
        # 选四个示例
        random_indices = np.random.choice(indices_except_target, self.exp_num, replace=False)
        examples = self.labels[class_idx, random_indices] #num,64,64


        for i in range(self.exp_num):
            values_ = examples[i][self.points[:, 0], self.points[:, 1]]
            voronoidata_ = griddata(self.points, values_, (grid_x, grid_y), method='nearest')
            voronoidata_exp_ = np.expand_dims(voronoidata_, axis=0)
            examples_ = np.expand_dims(examples[i], axis=0)
            combined_data_ = np.concatenate([voronoidata_exp_,examples_],axis = 0) #2,64,64
            combined_data_exp_ =np.expand_dims(combined_data_,axis=1) #2,1,64,64
            mask_ = np.expand_dims(mask_exp,axis=0) #1,1,64,64
            mask_exp_ = np.tile(mask_, (2, 1, 1, 1))  # 现在形状为 (2, 1, 64, 64)
            final_com_ = np.concatenate([combined_data_exp_,mask_exp_],axis=1) #2,2,64,64
            final_prompt[i] = final_com_


        # 将拓展后的mask和插值后的温度场拼接在一起

        return combined_data,sample_label,final_prompt
dataorigin = np.load('/mnt/d/codespace/DATASETRBF/Heat_Types100_source4_number1000_normalized.npz')
labels = dataorigin['T']

dataset_train = thermalDataset_vor(labels, train=True, train_ratio=0.8,exp_num=5)
dataset_test = thermalDataset_vor(labels, train=False, train_ratio=0.8,exp_num=5)

if __name__ =="__main__":
    train_loader = DataLoader(dataset_train,batch_size=10,shuffle=True)
    for i,(com,labels,prompt) in enumerate(train_loader):
        print(com.shape)
        print(labels.shape)
        print(prompt.shape)
        break

