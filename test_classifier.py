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
from torchvision import transforms
from torch.utils.data import Dataset

from nnutils import test_utils
from nnutils import mesh_net
from utils import image as img_util
from data import cub as cub_data

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# transforms
image_size = 256
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

def undo_resnet_preprocess(image_tensor):
    # takes in an image of shape 3 X H X W
    image_tensor = image_tensor.clone()
    # image_tensor.narrow(1,0,1).mul_(.229).add_(.485)
    image_tensor[0].mul_(.229).add_(.485)
    image_tensor[1].mul_(.224).add_(.456)
    image_tensor[2].mul_(.225).add_(.406)
    return image_tensor

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
        img = preprocess_image(img_path)
        return img, self.imgs[idx][1]

# based on order
classes = ('Anseriformes', 'Apodiformes', 'Caprimulgiformes', 'Charadriiformes', 'Coraciiformes',
           'Cuculiformes', 'Gaviiformes', 'Passeriformes', 'Pelecaniformes', 'Piciformes',
           'Podicipediformes', 'Procellariiformes', 'Suliformes')

num_insts = torch.tensor([240, 240, 165, 1364, 300, 292, 60, 7900, 110, 408, 240, 239, 231], dtype=torch.float)
class_weights = 1 / num_insts
class_weights = class_weights / sum(class_weights) * len(classes)

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

curr = "test"
checkpoint_folder = "./checkpoints_16-10"

train_dataset = CUBDataset(data_dir, "train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=False)
train_size = len(train_dataset)
test_dataset = CUBDataset(data_dir, "test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=False)
test_size = len(test_dataset)

classifier = Classifier((image_size, image_size), len(classes)).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss(torch.tensor(class_weights).to(device))  # reweight for training
criterion_unweighted = torch.nn.NLLLoss()

if curr == "train":
    print("Starting training")
    losses_train = []
    start_time = time.time()
    total_steps = 0

    for epoch in range(500):
        epoch_start_time = time.time()
        epoch_loss_train = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss_train += loss.item()
            '''
            if (i+1) % 92 == 0: # print every 92 minibatches - will have 2 in an epoch
                print("{}, {}: loss: {:.4f}".format(epoch+1, i+1, running_loss/100))
                losses += [running_loss]
                running_loss = 0.0
            '''
            total_steps += 1

        print("\nepoch {}: {:.2f}s \t{:.2f}s total".format(epoch+1, time.time() - epoch_start_time, time.time()-start_time))
        # print("\nepoch {}: {:.2f}s".format(epoch+1, time.time() - epoch_start_time))
        print("\ttrain loss: {}".format(epoch_loss_train/(i+1)))
        losses_train += [epoch_loss_train]

        if (epoch + 1) % 50 == 0:
            print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch + 1, total_steps))
            save_filename = 'classifier_checkpoint_{}.pth'.format(epoch+1)
            save_path = os.path.join(checkpoint_folder, save_filename)
            torch.save(classifier.cpu().state_dict(), save_path)
            classifier.to(device)

    print("=== train loss ===")
    print(losses_train)
    torch.save(torch.tensor(losses_train), os.path.join(checkpoint_folder, "losses_train.pt"))

else:
    print("Starting testing")
    losses_test = []
    losses_test_unweighted = []
    start_time = time.time()
    classifier.eval()
    with torch.no_grad():
        for epoch in range(50, 501, 50):
            print("\nCheckpoint", epoch)
            checkpoint_path = os.path.join(checkpoint_folder, "classifier_checkpoint_{}.pth".format(epoch))
            
            #classifier = Classifier((opts['img_size'], opts['img_size']), len(classes)).to(device)
            classifier.load_state_dict(torch.load(checkpoint_path))
            #classifier.eval()

            hit_inst = 0
            total_inst = 0
            epoch_loss_test = 0.0
            epoch_loss_test_unweighted = 0.0
            class_hits = [0] * len(classes)
            class_totals = [0] * len(classes)

            epoch_start_time = time.time()
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
                loss_unweighted = criterion_unweighted(outputs, labels)
                epoch_loss_test += loss.item()
                epoch_loss_test_unweighted += loss_unweighted.item()

                pred = outputs.argmax(1)

                for j in range(len(labels)):
                    class_totals[labels[j]] += 1
                    if labels[j] == pred[j]:
                        class_hits[labels[j]] += 1

            print("\nepoch {}: {:.2f}s \t{:.2f}s total".format(epoch, time.time() - epoch_start_time,
                                                               time.time() - start_time))
            print("\ttest loss: {} | {}\t accuracy: {:.4f}".format(epoch_loss_test / (i + 1), epoch_loss_test_unweighted / (i + 1),
                                                                   sum(class_hits) / sum(class_totals)))
            for i in range(len(classes)):
                print("{}: {:.3f}".format(i, (class_hits[i] / class_totals[i]) if class_totals[i] > 0 else -1), end=", ")
            print()
            losses_test += [epoch_loss_test]
            losses_test_unweighted += [epoch_loss_test_unweighted]


    print("=== test loss ===")
    print(losses_test)
    print("=== test loss (unweighted) ===")
    print(losses_test_unweighted)
    test_losses = torch.tensor((losses_test, losses_test_unweighted))
    torch.save(test_losses, os.path.join(checkpoint_folder, "losses_test.pt"))
