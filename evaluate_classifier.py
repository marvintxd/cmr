from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from classifier_model import CUBDataset, Classifier

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

image_size = 256

data_dir = "../cvpr18-inaturalist-transfer/data/cub_200/"
labelled_images_path = "../cvpr18-inaturalist-transfer/data/cub_200/categorized_by_order_images.txt"
f = open(labelled_images_path, 'r')
lines = f.readlines()
f.close()
X = []
y = []
for line in lines:
    path, label = line.strip().split(': ')
    X.append(path)
    y.append(int(label))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

# based on order
classes = ('Anseriformes', 'Apodiformes', 'Caprimulgiformes', 'Charadriiformes', 'Coraciiformes',
           'Cuculiformes', 'Gaviiformes', 'Passeriformes', 'Pelecaniformes', 'Piciformes',
           'Podicipediformes', 'Procellariiformes', 'Suliformes')

num_insts = torch.tensor([240, 240, 165, 1364, 300, 292, 60, 7900, 110, 408, 240, 239, 231], dtype=torch.float)
class_weights = 1 / num_insts
class_weights = class_weights / sum(class_weights) * len(classes)

checkpoint_folder = "./checkpoints_18-10_wd0"

test_dataset = CUBDataset(data_dir, X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=False)
test_size = len(test_dataset)

classifier = Classifier((image_size, image_size), len(classes)).to(device)
criterion = torch.nn.NLLLoss(torch.tensor(class_weights).to(device))  # reweight for training
criterion_unweighted = torch.nn.NLLLoss()

if True:
    print("Starting testing")
    losses_train = []
    losses_test = []
    losses_test_unweighted = []
    accuracy = []
    start_time = time.time()
    total_steps = 0

    classifier.eval()
    with torch.no_grad():
        for epoch in [150, 200]:
            print("\nCheckpoint", epoch)
            checkpoint_path = os.path.join(checkpoint_folder, "classifier_checkpoint_{}.pth".format(epoch))
            classifier.load_state_dict(torch.load(checkpoint_path))

            epoch_test_start_time = time.time()

            epoch_loss_test = 0.0
            epoch_loss_test_unweighted = 0.0

            preds_all = []
            labels_all = []

            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
                loss_unweighted = criterion_unweighted(outputs, labels)
                epoch_loss_test += loss.item()
                epoch_loss_test_unweighted += loss_unweighted.item()

                pred = outputs.argmax(1)

                preds_all.append(pred)  # list of tensors
                labels_all.append(labels)

            preds_all_t = torch.cat(preds_all)
            labels_all_t = torch.cat(labels_all)

            print("\ntest: {:.2f}s \t{:.2f}s total".format(time.time() - epoch_test_start_time,
                                                               time.time() - start_time))
            
            epoch_accuracy = sum(preds_all_t == labels_all_t).item() / len(labels_all_t)

            print("\ttest loss: {} | {}\t accuracy: {:.4f}".format(epoch_loss_test / (i + 1),
                                                                   epoch_loss_test_unweighted / (i + 1),
                                                                  epoch_accuracy))
            print('\nClassification Report\n')
            print(classification_report(labels_all_t.cpu(), preds_all_t.cpu(), target_names=classes))
            print('\n')
            losses_test += [epoch_loss_test]
            losses_test_unweighted += [epoch_loss_test_unweighted]
            accuracy += [epoch_accuracy]

