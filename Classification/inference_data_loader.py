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
    def __init__(self, data_path,root, image_size=(128, 128), mode='train',augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.patient_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        #self.transform = transform
        self.augmentation_prob = augmentation_prob
        self.data_path = data_path

        print("image count in {} path :{}".format(self.mode, len(self.patient_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        #data_path = '/home/ubuntu/liuyiyao/Multi_modal_Image/dataset/get_pseudo_3d_view_resample'

        """Reads an image from a file and preprocesses it and returns."""
        patient_name = self.patient_paths[index].split('/')[-1]
        patient_name_split = patient_name.split('_')
        if len(patient_name_split) == 3:
            patient_name = patient_name_split[0] + '_' + patient_name_split[1]

        patient_path = self.data_path+ '/' + patient_name + '/img'
        image_paths = os.listdir(patient_path)
        source = open(self.data_path + '/' + patient_name + '/label.txt')  # 打开源文件
        label = source.read()  # 显示所有源文件内容

        #print(img_length)
        img_out_names = ['1.bmp', '2.bmp', '3.bmp','4.bmp', '5.bmp', '6.bmp','7.bmp', '8.bmp', '9.bmp']
        img_length = len(img_out_names)
        out_img = torch.zeros((img_length,self.image_size[0],self.image_size[1]))
        #print(out_img.shape)
        #print(image_paths)
        i=0
        p_transform = random.random()
        for imagename in image_paths:
            if imagename in img_out_names:
                #print(imagename)
                image = Image.open(patient_path+'/'+imagename)
                aspect_ratio = image.size[1] / image.size[0]
                Transform = []
                CropRange = random.randint(110, 130)
                Transform.append(T.CenterCrop((CropRange, CropRange)))
                Transform = T.Compose(Transform)
                image = Transform(image)
                Transform = []

                if (self.mode == 'train') and p_transform <= self.augmentation_prob:
                    RotationDegree = random.randint(0, 3)
                    RotationDegree = self.RotationDegree[RotationDegree]
                    if (RotationDegree == 90) or (RotationDegree == 270):
                        aspect_ratio = 1 / aspect_ratio
                    Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
                    RotationRange = random.randint(-10, 10)
                    Transform.append(T.RandomRotation((RotationRange, RotationRange)))
                    Transform = T.Compose(Transform)
                    image = Transform(image)
                    ShiftRange_left = random.randint(0, 20)
                    ShiftRange_upper = random.randint(0, 20)
                    ShiftRange_right = image.size[0] - random.randint(0, 20)
                    ShiftRange_lower = image.size[1] - random.randint(0, 20)
                    image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
                    if random.random() < 0.5:
                        image = F.hflip(image)
                    if random.random() < 0.5:
                        image = F.vflip(image)
                    Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)
                    image = Transform(image)
                    Transform = []
                Transform.append(T.Resize(self.image_size))
                Transform.append(T.ToTensor())
                Transform = T.Compose(Transform)
                image = Transform(image)
                #print("image_shape:",image.shape)
                out_img[i,:,:] = image[0,:,:]
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


def get_loader(data_path,image_path, image_size, batch_size,  num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(data_path=data_path, root=image_path, image_size=image_size, mode=mode,
                          augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
