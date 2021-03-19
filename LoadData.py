import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
from PIL import Image


class PartialDataSet(Dataset):
    def __init__(self,dataset_path,ratio=100,sample=0,transform=None,random_seed=168,fix_seg=False):
        # If the data is divided by proportion, sample should be 0, and vice versa.
        assert(ratio==0 or sample==0)
        
        class_name = os.listdir(dataset_path)
        img_dir_set = []
        label_set = []
        for i in range(len(class_name)):
            class_path = os.path.join(dataset_path,class_name[i])
            image_name = os.listdir(class_path)
            np.random.seed(random_seed)
            
            # totally random as reported in papers.
            if fix_seg==False:
                # divided data by ratio
                random_index = np.random.choice(list(range(len(image_name))),int(len(image_name)/100*ratio))

                # divided data by sample per class
                if sample != 0:
                    random_index = np.random.choice(list(range(len(image_name))),sample)
            
            # This selection method can choose a wider range of aspect angels as much as possible, and obtain better results than the reported results in our papers.
            else:
                # divided data by ratio
                sample_numbers = int(len(image_name)/100*ratio)
                
                # divided data by sample
                if sample!=0:
                    sample_numbers = sample
                    
                segment_distance = len(image_name)//sample_numbers
                selected_index = np.zeros(sample_numbers,dtype='int')
                random_index = selected_index + np.random.randint(segment_distance)
                for k in range(len(random_index)):
                    random_index[k] += (segment_distance*k)%len(image_name)
                
                
                
                
            # this will draw some same samples to construct a training set which has the same number of sample as the original training set 
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


