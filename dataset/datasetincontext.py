import numpy as np
from torch.utils.data import Dataset, DataLoader


class thermalDatasetincontext(Dataset):
    def __init__(self, data, labels, train=True, num_samples=4, train_ratio=0.8):
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
        self.num_samples = num_samples

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
        # 把现在的idx排除
        indices_except_target = np.setdiff1d(self.indices, sample_idx_in_class)

        # 选四个示例
        random_indices = np.random.choice(indices_except_target, self.num_samples, replace=False)

        # Get the labels for these random indices
        examples = self.labels[class_idx, random_indices]
        examples = examples.reshape(self.num_samples, 64, 64)
        return sample_data, sample_label,examples
dataorigin = np.load('/mnt/d/codespace/DATASETRBF/Heat_Types100_source4_number1000_normalized.npz')
data = dataorigin['O']
labels = dataorigin['T']

dataset_train = thermalDatasetincontext(data, labels, train=True, train_ratio=0.8,num_samples=4)
dataset_test = thermalDatasetincontext(data, labels, train=False, train_ratio=0.8,num_samples=4)

if __name__ =="__main__":
    trainloader = DataLoader(dataset_train,shuffle=True,batch_size=800)
    for data ,label,samples in trainloader:
        print(data.shape,label.shape,samples.shape)
        break