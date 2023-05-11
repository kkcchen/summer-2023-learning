import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt('deep learning/data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        # self.x = torch.from_numpy(xy[:, 1:])
        # self.y = torch.from_numpy(xy[:, [0]]) 

        #don't convert to tensor
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples
    

# writing custom transforms
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target
    
if __name__ == '__main__':
    dataset = WineDataset(transform=None)
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features)

    composed = torchvision.transforms.Compose([ToTensor(), MulTransform(1.5)])
    dataset = WineDataset(transform=composed)
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features)

    # num_epochs = 2
    # tot_samples = len(dataset)
    # n_iters = math.ceil(tot_samples / 4)

    # for epoch in range(num_epochs):
    #     for i, (inputs, labels) in enumerate(dataloader):
    #         #forward
    #         if (i+1) % 5 == 0:
    #             print(f'epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_iters}, inputs {inputs.shape}')
