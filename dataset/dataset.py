import numpy as np
from torch.utils.data import Dataset, DataLoader


class thermalDataset(Dataset):
    def __init__(self, data, labels, train=True, train_ratio=0.8):
        """
        Custom dataset initializer.
        :param data: The data (e.g., 'T' from your dataset)
        :param labels: The labels or outputs corresponding to the data (e.g., 'o')
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        """
        assert data.shape[:2] == labels.shape[:2], "Data and labels must have matching first two dimensions"

        self.data = data
        self.labels = labels
        self.train = train

        # Determine split sizes
        num_samples_per_class = data.shape[1]
        num_train = int(num_samples_per_class * train_ratio)
        self.indices = np.arange(num_samples_per_class)

        if self.train:
            self.indices = self.indices[:num_train]
        else:
            self.indices = self.indices[num_train:]

    def __len__(self):
        return len(self.indices) * self.data.shape[0]  # Total number of samples

    def __getitem__(self, idx):
        class_idx = idx // len(self.indices)
        sample_idx_in_class = self.indices[idx % len(self.indices)]
        sample_data = self.data[class_idx, sample_idx_in_class]
        sample_label = self.labels[class_idx, sample_idx_in_class]
        return sample_data, sample_label
dataorigin = np.load('/mnt/d/codespace/DATASETRBF/Heat_Types100_source4_number1000_normalized.npz')
data = dataorigin['O']
labels = dataorigin['T']

dataset_train = thermalDataset(data, labels, train=True, train_ratio=0.8)
dataset_test = thermalDataset(data, labels, train=False, train_ratio=0.8)

if __name__ =="__main__":
    trainloader = DataLoader(dataset_train,shuffle=True,batch_size=800)
    for data ,label in trainloader:
        print(data.shape,label.shape)