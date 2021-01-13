import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
from PIL import Image


class PartialDataSet(Dataset):
    def __init__(self,dataset_path,ratio=100,sample=0,transform=None,random_seed=168):
        class_name = os.listdir(dataset_path)
        img_dir_set = []
        label_set = []
        for i in range(len(class_name)):
            class_path = os.path.join(dataset_path,class_name[i])
            image_name = os.listdir(class_path)
            
            np.random.seed(random_seed)
            random_index = np.random.choice(list(range(len(image_name))),int(len(image_name)/100*ratio))
            if sample != 0:
                random_index = np.random.choice(list(range(len(image_name))),sample)
            #random_index = np.random.choice(list(range(int(len(image_name)*ratio/100))),len(image_name))
            for j in range(len(image_name)):
                img_dir = os.path.join(class_path,image_name[random_index[j%len(random_index)]])
                img_dir_set.append(img_dir)
                label_set.append(i)

        self.transform = transform
        self.imgdir_set = img_dir_set
        self.label_set = label_set

    def __getitem__(self, index):
        img = Image.open(self.imgdir_set[index])
        label = self.label_set[index]
        if self.transform == None:
            img = transforms.CenterCrop(84)(img)
            tensor_data = transforms.ToTensor()(img)
        else:
            tensor_data = self.transform(img)
        return tensor_data,label

    def __len__(self):
        return len(self.imgdir_set)

if __name__ == '__main__':
    dataset_path = 'D:\\假期实验\\MSTAR补充实验\\MSTAR_DataSet\\17DEG_Train'
    dataset = PartialDataSet(dataset_path,5)
    train_loader = DataLoader(dataset,32,True)

    for img,label in train_loader:
        print(img,label)
        break


