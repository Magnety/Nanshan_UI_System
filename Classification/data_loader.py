import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

np.set_printoptions(threshold=100000)


class ImageFolder(data.Dataset):
    def __init__(self, data_path,name_path, image_size=(128, 128), mode='train',augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.name_path = name_path
        self.centers = os.listdir(self.name_path)
        self.patient_paths = []

        for center in self.centers:
            self.center_path = self.name_path+'/'+center
            self.patient_paths.extend(list(map(lambda x: os.path.join(self.center_path , x), os.listdir(self.center_path))))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        #self.transform = transform
        self.augmentation_prob = augmentation_prob
        self.data_path = data_path





        print("image count in {} path :{}".format(self.mode, len(self.patient_paths)))

    def __getitem__(self, index):
        center_name = self.patient_paths[index].split('/')[-2]
        patient_name = self.patient_paths[index].split('/')[-1]
        image_path = self.data_path+ '/' +center_name+'/'+ patient_name + '/swe_100_image/horizontal_view'
        image_names = os.listdir(image_path)
        #print(image_path,"-------",len(image_names))
        source = open(self.data_path + '/' +center_name+'/'+ patient_name + '/label.txt')  # 打开源文件
        label = source.read()  # 显示所有源文件内容
        #print(img_length)
        #print(center_name, " ", patient_name)
        #print(image_names)
        img_out_names = [image_names[0],image_names[1]]
        #print(img_out_names)
        img_length = len(img_out_names)*3
        out_img = torch.zeros((img_length,self.image_size[0],self.image_size[1]))
        #print(out_img.shape)
        #print(image_paths)
        i=0
        Transform1 = []
        if self.augmentation_prob == 0:
            CropRange = self.image_size[0]
        else :
            CropRange = random.randint(self.image_size[0] - 10, self.image_size[0] + 10)
        Transform1.append(T.CenterCrop((CropRange, CropRange)))
        Transform1 = T.Compose(Transform1)
        p_transform = random.random()
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:

            for imagename in img_out_names:
                image = Image.open(image_path + '/' + imagename)
                aspect_ratio = image.size[1] / image.size[0]
            Transform2 = []
            RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1 / aspect_ratio
            Transform2.append(T.RandomRotation((RotationDegree, RotationDegree)))
            RotationRange = random.randint(-10, 10)
            Transform2.append(T.RandomRotation((RotationRange, RotationRange)))
            Transform2 = T.Compose(Transform2)
            ShiftRange_left = random.randint(0, 20)
            ShiftRange_upper = random.randint(0, 20)
            ShiftRange_right = random.randint(0, 20)
            ShiftRange_lower = random.randint(0, 20)
            flip_random = random.random()
            Transform3 = []
            Transform3.append(T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02))
            Transform3 = T.Compose(Transform3)
        for imagename in img_out_names:
            #print(imagename)

            image = Image.open(image_path+'/'+imagename)
            if (self.mode == 'train') and p_transform <= self.augmentation_prob:
                image = Transform1(image)
                image = Transform2(image)
                ShiftRange_right = image.size[0] - ShiftRange_right
                ShiftRange_lower = image.size[1] - ShiftRange_lower
                image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
                if  flip_random< 0.5:
                    image = F.hflip(image)
                if flip_random < 0.5:
                    image = F.vflip(image)
                #image = Transform3(image)
            Transform = []
            Transform.append(T.Resize(self.image_size))
            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)
            image = Transform(image)
            #print("image_shape:",image.shape)
            out_img[i*3+0, :, :] = image[0, :, :]
            out_img[i*3+1, :, :] = image[1, :, :]
            out_img[i*3+2, :, :] = image[2, :, :]
            i+=1
        label_np = np.array(float(label), np.float)
        label_tensor = torch.from_numpy(label_np)
        label_tensor = label_tensor.long()
            # image = torch.unsqueeze(image, dim=0)
        #print(out_img.shape)
        return out_img, label_tensor,self.patient_paths[index].split('/')[-1]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.patient_paths)


def get_loader(data_path,name_path, image_size, batch_size,  num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(data_path=data_path, name_path=name_path, image_size=image_size, mode=mode,
                          augmentation_prob=augmentation_prob)
    if augmentation_prob==0:
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
