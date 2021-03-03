import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import is_image_file, load_img
import glob
import random
import os
import numpy as np
from PIL import Image
import cv2
import pickle
from tqdm import tqdm

class MICROSCOPY(data.Dataset):
    def __init__(self, root, training=True):
        super(MICROSCOPY, self).__init__()

        self.files_A = []
        self.files_B = []

        self.root_train = os.path.join(root, 'MICCAI3/train/')
        self.root_test = os.path.join(root, 'MICCAI3/test/')
        self.root_label = os.path.join(root, 'MICCAI3/label/')
        self.training = training

        if not os.path.isdir(self.root_train):
            print('path not exist')
            assert False
        
        self.image_train = glob.glob(self.root_train + '/*.png')
        self.image_test = glob.glob(self.root_test + '/*.png')
        self.image_label = glob.glob(self.root_label + '/*.png')

        if self.training:
            for i in tqdm(range(len(self.image_train))):
                self.files_A.append(self.image_train[i])
                self.files_B = None
        else:
            for i in tqdm(range(len(self.image_test))):
                self.files_A.append(self.image_test[i])
                self.files_B.append(self.image_label[i])

        transform_list = [# transforms.Resize(256, Image.BICUBIC), 
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(180, resample=False, expand=False, center=None),
                transforms.ToTensor()]
                # transforms.Normalize((0.5,0.5), (0.5,0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Normal image
        img_A = cv2.imread(self.files_A[index], 0)
        normal_img_A = torch.tensor(img_A).float()/128.0 - 1.0
        normal_img_A = torch.unsqueeze(normal_img_A, 0)
        if self.files_B is not None:
            img_B = cv2.imread(self.files_B[index], 0)
            normal_img_B = self.transform(Image.fromarray(np.uint8(img_B)))
            return normal_img_A, normal_img_B

        return normal_img_A


    def __len__(self):
        return len(self.files_A)

    def fill_blank(self, crop_img_A):

        # h, w, c = crop_img_A.shape
        # s = max(h, w)

        # l_h = int((s-h)/2)
        # l_w = int((s-w)/2)

        # blank = np.zeros((s, s, 3))

        # blank[l_h:l_h+h, l_w:l_w+w, :] = crop_img_A

        blank = cv2.resize(crop_img_A, (256, 256), interpolation=cv2.INTER_CUBIC)

        return blank

    def rotate_and_flip(self, image, angle, center=None, scale=1.0, flip=2):

        if flip!=2:
            image = cv2.flip(image, flip)

        (h, w) = image.shape[:2]
    
        if center is None:
            center = (w / 2, h / 2)
    
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
    
        return rotated

class NATURAL(data.Dataset):
    def __init__(self, root, training=True):
        super(NATURAL, self).__init__()

        self.files_A = []

        self.root = os.path.join(root, 'NATURAL/')
        self.training = training

        if not os.path.isdir(self.root):
            print('path not exist')
            assert False
        
        self.image = glob.glob(self.root + '/images/*.jpg')

        if self.training:
            for i in tqdm(range(10000)):
                self.files_A.append(self.image[i])
        else:
            self.files_A = self.image[10000:]

        transform_list = [transforms.Resize(256, Image.BICUBIC), 
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(180, resample=False, expand=False, center=None),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Normal image
        img_A = cv2.imread(self.files_A[index])

        if self.training:
            normal_img = self.fill_blank(img_A)

            angle = random.randint(-180,180)
            flip  = random.randint(-1,2)
            normal_img = self.rotate_and_flip(normal_img, angle, flip=flip)
        else:
            normal_img = self.fill_blank(img_A)

        normal_img = self.transform(Image.fromarray(np.uint8(normal_img)))

        return normal_img


    def __len__(self):
        return len(self.files_A)

    def fill_blank(self, crop_img_A):
        blank = cv2.resize(crop_img_A,(224,224),interpolation=cv2.INTER_CUBIC)

        return blank

    def rotate_and_flip(self, image, angle, center=None, scale=1.0, flip=2):
        if flip!=2:
            image = cv2.flip(image, flip)

        (h, w) = image.shape[:2]
    
        if center is None:
            center = (w / 2, h / 2)
    
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
    
        return rotated


