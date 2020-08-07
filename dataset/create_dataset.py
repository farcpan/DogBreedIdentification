from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

import numpy as np
import os
import pandas as pd
import sys
import time
 

class DogDataset(Dataset):
    def __init__(self, csv_file_path, image_dir, transform=None):
        self.image_dataframe = pd.read_csv(csv_file_path)
        self.image_dir = image_dir                                             
        self.transform = transform


    def __len__(self):
        return len(self.image_dataframe)


    def __getitem__(self, idx):
        label = self.image_dataframe['breed_id'][idx]
        filename = self.image_dataframe['id'][idx]
        img_name = "{}/{}.jpg".format(self.image_dir, filename)
        print("[{}]: id={}, breed_id={}".format(idx, filename, label))

        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    since = time.time()
    dataset = DogDataset("../labels/labels.csv", "../train", 
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                        ]))

    # split dataset to training and validation
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1)

    # mini-batch size: 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)

    print("-" * 30)
    print("Data Loading Time: {} [sec]".format(time.time() - since))
    print("-" * 30)

    torch.save(train_loader, './train_loader.pth')
    torch.save(val_loader, './validation_loader.pth')

    print("Done.")