import os
import glob
from PIL import Image, ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import cv2


class SegDataset(data.Dataset):
    def __init__(self, image_files: list, mask_files: list, trainsize, augmentations, subset = 'training'):
        self.trainsize = trainsize
        self.subset = subset
        self.augmentations = augmentations
        self.images = image_files
        self.gts = mask_files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.subset == 'testing':
            print("Testing data set require augmentations to be False")
            self.augmentations = False
        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_files, mask_files, batchsize, trainsize, shuffle=True, num_workers=1, pin_memory=True, augmentation=False, subset = 'training'):
    dataset = SegDataset(image_files, mask_files, trainsize, augmentation, subset = subset)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    main_data_dir = os.getcwd() + "/Data set"
    seg_data_dir = main_data_dir + "/Seg_Task_Data"

    train_seg_data_dir = seg_data_dir + "/Train"
    train_seg_img_dir = train_seg_data_dir + "/Img"
    train_seg_img_files = glob.glob(train_seg_img_dir + "/*")
    train_seg_mask_dir = train_seg_data_dir + "/Mask"
    train_seg_mask_files = glob.glob(train_seg_mask_dir + "/*")

    test_seg_data_dir = seg_data_dir + "/Test"
    test_seg_img_dir = test_seg_data_dir + "/Img"
    test_seg_img_files = glob.glob(test_seg_img_dir + "/*")
    test_seg_mask_dir = test_seg_data_dir + "/Mask"
    test_seg_mask_files = glob.glob(test_seg_mask_dir + "/*")

    train_loader = get_loader(image_files=train_seg_img_files, mask_files=train_seg_mask_files, batchsize=32, trainsize=300)
    train_features, train_labels = next(iter(train_loader))
    img = train_features[0]
    img = torch.permute(img, (2, 1, 0))
    plt.subplot(1, 2, 1)
    plt.imshow((img.numpy()).astype(np.uint8))
    plt.subplot(1, 2, 2)
    plt.imshow(torch.permute(train_labels[0], (2, 1, 0)).numpy().astype(np.uint8))
    plt.show()