import os
import numpy as np
import skimage.io as io

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset

from nnutils import mesh_net
from utils import image as img_util

resnet_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# img_transforms = transforms.Compose([transforms.Resize(image_size),  # this resizes based on min dimension, not max
#                                      # transforms.CenterCrop(image_size),
#                                      # transforms.RandomHorizontalFlip(0.5),
#                                      transforms.ToTensor(),
#                                      resnet_transform])

# def preprocess_image(img_path):
#     im = Image.open(img_path) # PIL image
#     if im.mode == 'L':
#         gray = im.split()[0]
#         im = Image.merge('RGB', (gray, gray, gray))
#     transformed = img_transforms(im)
#     return transformed # 3 X H X W, which is required for resnet

def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path) / 255.

    # if grayscale, convert to RGB
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

    img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    # necessary preprocessing for resnet
    img = torch.tensor(img, dtype=torch.float)
    img = resnet_transform(img)

    # random flip
    if np.random.rand(1) > 0.5:
        img = torch.flip(img, (2,))

    return img

class CUBDataset(Dataset):
    # def __init__(self, data_dir, type="train"):
    #     self.data_dir = data_dir
    #
    #     if type == "train":
    #         img_list_file = open(os.path.join(data_dir, "categorized_by_order_train.txt"), "r")
    #     else:
    #         img_list_file = open(os.path.join(data_dir, "categorized_by_order_val.txt"), "r")
    #     img_list = img_list_file.readlines()
    #
    #     # each line: images/200.Common_Yellowthroat/Common_Yellowthroat_0010_190572.jpg: 7
    #     self.imgs = []
    #     for line in img_list:
    #         path, label = line.strip().split(': ')
    #         self.imgs.append((path, int(label)))

    def __init__(self, data_dir, X, y):
        self.data_dir = data_dir
        self.X = X
        self.y = y

    def __len__(self):
        # return len(self.imgs)
        return len(self.X)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.data_dir, self.imgs[idx][0])
        # img = preprocess_image(img_path)
        # return img, self.imgs[idx][1]

        img_path = os.path.join(self.data_dir, self.X[idx])
        img = preprocess_image(img_path)
        return img, self.y[idx]

class Classifier(nn.Module):
    def __init__(self, input_shape, n_classes, n_blocks=4, nz_feat=100):
        super(Classifier, self).__init__()
        self.encoder = mesh_net.Encoder(input_shape, n_blocks=n_blocks, nz_feat=nz_feat)
        self.linear = nn.Linear(nz_feat, n_classes)
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, img):
        img_feat = self.encoder.forward(img)
        x = self.linear(img_feat)
        x = self.log_softmax(x)
        return x
