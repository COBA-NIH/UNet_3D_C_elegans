from torch.utils.data import Dataset
import skimage
import numpy as np
import torch

class MaddoxDataset(Dataset):
    def __init__(self, data_csv, transform=None):
        self.data = data_csv
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = skimage.io.imread(self.data.iloc[idx, 0]).astype(np.float32)
        mask = skimage.io.imread(self.data.iloc[idx, 1]).astype(np.float32)
        if self.transform:
            sample = {'image': image, 'mask': mask}
            data = self.transform(**sample)
            # return data["image"], data["mask"]
            return data
        else:
            image = torch.from_numpy(image).astype(np.float32)
            image = image[np.newaxis,...]
            mask = torch.from_numpy(mask).astype(np.float32)
            mask = mask[np.newaxis,...]
            return image, mask

        

