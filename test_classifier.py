"""
Demo of CMR.

Note that CMR assumes that the object has been detected, so please use a picture of a bird that is centered and well cropped.

Sample usage:

python demo.py --name bird_net --num_train_epoch 500 --img_path misc/demo_data/img1.jpg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

# from absl import flags, app
import numpy as np
import skimage.io as io

import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset

from nnutils import test_utils
from nnutils import mesh_net
from utils import image as img_util
from data import cub as cub_data


# flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
# ###
# # flags.DEFINE_string('img_path', 'misc/demo_data/img1.jpg', 'Image to run')
# # flags.DEFINE_integer('num_train_epoch', 500, "num")
# ###
# flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')
#
# opts = flags.FLAGS

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

opts = {'img_size': 256}


def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path) / 255.
    
    # if grayscale
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

    return img

data_dir = "../cvpr18-inaturalist-transfer/data/cub_200/"
class CUBDataset(Dataset):
    def __init__(self, data_dir, type="train"):
        self.data_dir = data_dir

        if type == "train":
            img_list_file = open(os.path.join(data_dir, "categorized_by_order_train.txt"), "r")
        else:
            img_list_file = open(os.path.join(data_dir, "categorized_by_order_val.txt"), "r")
        img_list = img_list_file.readlines()

        #each line: images/200.Common_Yellowthroat/Common_Yellowthroat_0010_190572.jpg: 7
        self.imgs = []
        for line in img_list:
            path, label = line.strip().split(': ')
            self.imgs.append((path, int(label)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.imgs[idx][0])
        try:
            img = preprocess_image(img_path)
        except IndexError as err:
            print("ERROR IN PREPROCESS")
            imagee = io.imread(img_path)
            print(imagee.shape)
            print("issue with path", img_path)
            print("===============================\n\n")
            torch.save(imagee, "inst_with_issue.pt")
            print(err)
            raise
        img = torch.tensor(img, dtype=torch.float)
        # img = io.imread(img_path)
        return img, self.imgs[idx][1]

# based on order
classes = ('Anseriformes', 'Apodiformes', 'Caprimulgiformes', 'Charadriiformes', 'Coraciiformes',
           'Cuculiformes', 'Gaviiformes', 'Passeriformes', 'Pelecaniformes', 'Piciformes',
           'Podicipediformes', 'Procellariiformes', 'Suliformes')

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



train_dataset = CUBDataset(data_dir, "train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
test_dataset = CUBDataset(data_dir, "test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=True)

classifier = Classifier((opts['img_size'], opts['img_size']), len(classes)).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

print("Starting training")
running_loss = 0.0
losses = []
start_time = time.time()
total_steps = 0
#for epoch in range(opts.num_pretrain_epochs, opts.num_epochs):
for epoch in range(500):
    epoch_start_time = time.time()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        '''
        if (i+1) % 92 == 0: # print every 92 minibatches - will have 2 in an epoch
            print("{}, {}: loss: {:.4f}".format(epoch+1, i+1, running_loss/100))
            losses += [running_loss]
            running_loss = 0.0
        '''
        total_steps += 1

    print("epoch {}: {:.2f}s \t{:.2f}s total".format(epoch+1, time.time() - epoch_start_time, time.time()-start_time))
    print("\tloss: {}".format(running_loss/(i+1)))
    losses += [running_loss]
    running_loss = 0.0

   
    if (epoch + 1) % 50 == 0:
        print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch + 1, total_steps))
        save_filename = 'classifier_checkpoint_{}.pth'.format(epoch+1)
        save_path = os.path.join("./checkpoints", save_filename)
        torch.save(classifier.cpu().state_dict(), save_path)
        classifier.to(device)

print(losses)
    # if (epoch + 1) % opts.save_epoch_freq == 0:
    #     print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
    #     self.save('latest')
    #     self.save(epoch + 1)


# if __name__ == '__main__':
#     opts.batch_size = 1
#     app.run(main)
