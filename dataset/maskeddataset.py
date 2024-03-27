import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class MaskedDataset(Dataset):
    def __init__(self, labels,mask_ratio=0.9, train=True, train_ratio=0.8):
        """
        Custom dataset initializer.
        :param data: The data (e.g., 'T' from your dataset)
        :param labels: The labels or outputs corresponding to the data (e.g., 'o')
        :param train: Boolean flag indicating whether this is training data. Default is True.
        :param train_ratio: Ratio of data to be used for training. Default is 0.8.
        """



        self.labels = labels
        self.mask_ratio = mask_ratio
        self.train = train
        # x = np.linspace(8, 55, 4, dtype=int)
        # y = np.linspace(8, 55, 4, dtype=int)
        # xv, yv = np.meshgrid(x, y)
        # #这里是选择的测点
        # self.points = np.vstack([xv.ravel(), yv.ravel()]).T


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
        flat_label = label.view(-1)  # Flatten the label to 1D for easy indexing
        flat_label[mask_indices] = 0  # Assuming 0 is the masking value
        return flat_label.view_as(label)  # Reshape back to original shape



    def __len__(self):
        return len(self.indices) * self.labels.shape[0]  # Total number of samples

    def __getitem__(self, idx):
        class_idx = idx // len(self.indices)
        sample_idx_in_class = self.indices[idx % len(self.indices)]
        sample_label = self.labels[class_idx, sample_idx_in_class]
        sample_label = sample_label.reshape(64, 64)
        sample_label_ = torch.from_numpy(sample_label).clone()  # Clone the data to avoid modifying the original label
        masked_label = self.add_random_mask(sample_label_)
        return masked_label, sample_label
dataorigin = np.load('/mnt/d/codespace/DATASETRBF/Heat_Types1000_source4_number100_normalized.npz')
labels = dataorigin['T']

dataset_train = MaskedDataset(labels, train=True, train_ratio=0.8)
dataset_test = MaskedDataset(labels, train=False, train_ratio=0.8)

if __name__ =="__main__":
    trainloader = DataLoader(dataset_train,shuffle=True,batch_size=800)
    for masked_label, label in trainloader:
        print(masked_label.shape, label.shape)
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(6, 6, figsize=(16, 16), dpi=200)
        fig.tight_layout(pad=3.0)  # Add some padding between figures

        for i in range(6):  # 6 rows of images
            for j in range(3):  # 3 pairs of images per row
                ax1 = axis[i, 2 * j]  # Even index for masked images
                ax2 = axis[i, 2 * j + 1]  # Odd index for original images
                masked_img = masked_label[i * 3 + j].cpu().numpy()
                original_img = label[i * 3 + j].cpu().numpy()

                ax1.imshow(masked_img, vmin=-2, vmax=2, cmap='bwr')
                ax1.set_title('Masked')
                ax1.axis('off')

                ax2.imshow(original_img, vmin=-2, vmax=2, cmap='bwr')
                ax2.set_title('Original')
                ax2.axis('off')

        plt.show()
        break  # Only show the first batch