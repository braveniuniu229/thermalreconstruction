import numpy as np
from torch.utils.data import Dataset, DataLoader


class thermalDataset(Dataset):
    def __init__(self, labels, train=True, train_ratio=0.8):
        """
        Custom dataset initializer.
        :param data: The data (e.g., 'T' from your dataset)
        :param labels: The labels or outputs corresponding to the data (e.g., 'o')
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        """



        self.labels = labels
        self.train = train
        x = np.linspace(8, 55, 4, dtype=int)
        y = np.linspace(8, 55, 4, dtype=int)
        xv, yv = np.meshgrid(x, y)
        #这里是选择的测点
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
        sample_label = sample_label.reshape(64,64)
        sample_data = sample_label[self.points[:,0],self.points[:,1]]
        return sample_data, sample_label
dataorigin = np.load('../data/Heat_Types1_source4_number100000fixed_normalized.npz')
labels = dataorigin['T']

dataset_train = thermalDataset(labels, train=True, train_ratio=0.8)
dataset_test = thermalDataset(labels, train=False, train_ratio=0.8)

if __name__ =="__main__":
    trainloader = DataLoader(dataset_train,shuffle=True,batch_size=800)
    for data ,label in trainloader:
        print(data.shape,label.shape)