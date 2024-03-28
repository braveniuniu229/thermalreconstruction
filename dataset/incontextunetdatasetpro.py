import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata

class thermalDataset_vor(Dataset):
    def __init__(self, labels, exp_num, train=True, mask_ratio=0.8,train_ratio=0.8):
        """
        Custom dataset initializer.
        :param labels: The labels or outputs corresponding to the data
        :param exp_num: Number of samples to draw from the same class except the target
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        """
        self.exp_num = exp_num
        self.labels = labels.reshape(labels.shape[0], labels.shape[1], 64, 64)
        self.train = train

        x = np.linspace(8, 55, 4, dtype=int)
        y = np.linspace(8, 55, 4, dtype=int)
        xv, yv = np.meshgrid(x, y)
        self.points = np.vstack([xv.ravel(), yv.ravel()]).T
        self.mask_ratio= mask_ratio
        num_samples_per_class = labels.shape[1]
        num_train = int(num_samples_per_class * train_ratio)
        self.train_indices = np.arange(num_train)
        self.test_indices = np.arange(num_train, num_samples_per_class)

        if self.train:
            self.indices = self.train_indices
        else:
            self.indices = self.test_indices

    def __len__(self):
        return len(self.indices) * self.labels.shape[0]  # Total number of samples

    def apply_uniform_or_random_mask(self,samples,  mask_type='uniform'):
        """
        Apply either a uniform or a random mask to the samples.

        :param samples: numpy array of shape (n_samples, 64, 64)
        :param mask_ratio: Ratio of pixels to be masked (set to 0)
        :param mask_type: Type of mask to apply ('uniform' or 'random')
        :return: Masked samples with the same shape as input
        """
        n_samples, height, width = samples.shape
        masked_samples = np.copy(samples)  # Copy to avoid modifying the original samples

        if mask_type == 'random':
            # Apply random mask
            n_pixels_to_mask = int(height * width * self.mask_ratio)
            for i in range(n_samples):
                mask_indices = np.random.choice(height * width, n_pixels_to_mask, replace=False)
                mask_indices = np.unravel_index(mask_indices, (height, width))
                masked_samples[i][mask_indices] = 0
        elif mask_type == 'uniform':
            # Apply uniform mask
            step = int(1 / np.sqrt(self.mask_ratio))
            for i in range(n_samples):
                mask_indices = np.array([(x, y) for x in range(0, height, step) for y in range(0, width, step)])
                if mask_indices.size > 0:
                    masked_samples[i][mask_indices[:, 0], mask_indices[:, 1]] = 0

        return masked_samples

    def __getitem__(self, idx):
        class_idx = idx // len(self.indices)
        sample_idx_in_class = self.indices[idx % len(self.indices)]
        true_label = self.labels[class_idx, sample_idx_in_class]

        values = true_label[self.points[:, 0], self.points[:, 1]]
        grid_x, grid_y = np.meshgrid(np.linspace(0, 63, 64), np.linspace(0, 63, 64))
        voronoidata = griddata(self.points, values, (grid_x, grid_y), method='nearest')

        mask = np.zeros_like(true_label, dtype=np.float32)
        mask[self.points[:, 0], self.points[:, 1]] = 1
        voronoidata_exp = np.expand_dims(voronoidata, axis=0)
        mask_exp = np.expand_dims(mask, axis=0)
        combined_data = np.concatenate([voronoidata_exp, mask_exp], axis=0)

        # 对于测试集, 允许从整个数据集中选择额外的样本
        if not self.train:
            combined_indices = np.concatenate([self.train_indices, self.test_indices])
        else:
            combined_indices = self.train_indices
        indices_except_target = np.setdiff1d(combined_indices, [sample_idx_in_class])
        samples_index = np.random.choice(indices_except_target, self.exp_num, replace=False)
        samples = self.labels[class_idx, samples_index] # exp_num,64,64
        masked_samples = self.apply_uniform_or_random_mask(samples)

        return combined_data, true_label, masked_samples

dataorigin = np.load('./data/Heat_Types10000_source4_number10fixed_normalized.npz')
labels = dataorigin['T']

dataset_train = thermalDataset_vor(labels, exp_num=1, train=True, train_ratio=0.8)
dataset_test = thermalDataset_vor(labels, exp_num=1, train=False, train_ratio=0.8)

if __name__ =="__main__":
    train_loader = DataLoader(dataset_train,batch_size=10,shuffle=True)
    for i,(com,labels,m_samples) in enumerate(train_loader):
        print(com.shape)
        print(labels.shape)
        print(m_samples.shape)
        break
