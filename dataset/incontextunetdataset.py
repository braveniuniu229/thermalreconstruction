import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata
# 实际上并不需要保存测点的数据，直接生成测点坐标从T中生成测点数据就可以，这样测点的坐标每次都可以自由选择，
# 无论是做voronoi划分还是直接一维输入都可以，而且可以保证两次的坐标索引是相同的很自由

class thermalDataset_vor(Dataset):
    def __init__(self, labels, exp_num,maskraito=0.7,mask=True,train=True, train_ratio=0.8):
        """
        Custom dataset initializer.
        :param data: The data (e.g., 'T' from your dataset)
        :param labels: The labels or outputs corresponding to the data (e.g., 'o')
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        """

        self.mask = mask
        self.exp_num =exp_num
        self.labels = labels.reshape(labels.shape[0],labels.shape[1],64,64)
        self.train = train
        if mask:
            assert maskraito is not None ,"give a mask ratio"
            self.mask_ratio =maskraito
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

    def add_random_mask(self, label):
        total_pixels = label.numel()
        num_masked = int(total_pixels * self.mask_ratio)
        mask_indices = np.random.choice(total_pixels, num_masked, replace=False)
        flat_label = label.ravel()  # Flatten the label to 1D for easy indexing
        flat_label[mask_indices] = 0  # Assuming 0 is the masking value
        return flat_label.reshape(label.shape)  # Reshape back to original shape
    def __len__(self):
        return len(self.indices) * self.labels.shape[0]  # Total number of samples

    def __getitem__(self, idx):
        class_idx = idx // len(self.indices)
        sample_idx_in_class = self.indices[idx % len(self.indices)]
        true_label = self.labels[class_idx, sample_idx_in_class]
        #进行划分的测点值
        values = true_label[self.points[:, 0], self.points[:, 1]]
        # 最后进行这个插值划分的网格m
        grid_x, grid_y = np.meshgrid(np.linspace(0, 63, 64), np.linspace(0, 63, 64))
        voronoidata = griddata(self.points, values, (grid_x, grid_y), method='nearest')
        mask = np.zeros_like(true_label, dtype=np.float32)
        mask[self.points[:, 0], self.points[:, 1]] = 1
        # 新建一个轴用于拼接
        voronoidata_exp = np.expand_dims(voronoidata, axis=0)
        mask_exp = np.expand_dims(mask, axis=0)

        # 将拓展后的mask和插值后的温度场拼接在一起
        combined_data = np.concatenate([voronoidata_exp, mask_exp], axis=0)
        indices_except_target = np.setdiff1d(self.indices, [sample_idx_in_class])
        samples_index = np.random.choice(indices_except_target,self.exp_num,replace=False)
        samples = self.labels[class_idx,samples_index] #exp_num,64,64
        samples_ = torch.from_numpy(samples).clone()
        masked_samples = np.zeros_like(samples_)
        if self.mask:
            for i in range(samples_.shape[0]):
                masked_samples[i] = self.add_random_mask(samples_[i])



            return combined_data,true_label,masked_samples
        else:
            return combined_data,true_label,samples

dataorigin = np.load('../data/Heat_Types500_source4_number200fixed_normalized.npz')
labels = dataorigin['T']


dataset_train = thermalDataset_vor(labels, exp_num=1,train=True, mask=True,train_ratio=0.8)
dataset_test = thermalDataset_vor(labels, exp_num=1,train=False, mask=True,train_ratio=0.8)

if __name__ =="__main__":
    train_loader = DataLoader(dataset_train,batch_size=10,shuffle=True)
    for i,(com,labels,samples) in enumerate(train_loader):
        print(com.shape)
        print(labels.shape)
        print(samples.shape)

        break

