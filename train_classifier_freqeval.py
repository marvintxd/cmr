from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from classifier_model import CUBDataset, Classifier

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def undo_resnet_preprocess(image_tensor):
    # takes in an image of shape 3 X H X W
    image_tensor = image_tensor.clone()
    # image_tensor.narrow(1,0,1).mul_(.229).add_(.485)
    image_tensor[0].mul_(.229).add_(.485)
    image_tensor[1].mul_(.224).add_(.456)
    image_tensor[2].mul_(.225).add_(.406)
    return image_tensor

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

checkpoint_folder = "./checkpoints_23-10_wd0"

# train_dataset = CUBDataset(data_dir, "train")
train_dataset = CUBDataset(data_dir, X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=False)
train_size = len(train_dataset)
# test_dataset = CUBDataset(data_dir, "test")
test_dataset = CUBDataset(data_dir, X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=False)
test_size = len(test_dataset)

classifier = Classifier((image_size, image_size), len(classes)).to(device)
criterion = torch.nn.NLLLoss(torch.tensor(class_weights).to(device))  # reweight for training
criterion_unweighted = torch.nn.NLLLoss()

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
# lr: pytorch default = 0.001, cmr default = 0.0001

# if curr == "train":
if True:
    print("Starting training")
    losses_train = []
    losses_test = []
    losses_test_unweighted = []
    accuracy = []
    start_time = time.time()
    total_steps = 0

    min_loss_test_epoch = 0
    min_loss_test_unweighted_epoch = 0
    max_accuracy_epoch = 0

    for epoch in range(500):
        epoch_start_time = time.time()
        epoch_loss_train = 0.0
        classifier.train()
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

        print("\nepoch {}: {:.2f}s \t{:.2f}s total".format(epoch + 1, time.time() - epoch_start_time,
                                                           time.time() - start_time))
        # print("\nepoch {}: {:.2f}s".format(epoch+1, time.time() - epoch_start_time))
        print("\ttrain loss: {}".format(epoch_loss_train / (i + 1)))
        losses_train += [epoch_loss_train]

        if (epoch + 1) % 50 == 0:
            print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch + 1, total_steps))
            save_filename = 'classifier_checkpoint_{}.pth'.format(epoch + 1)
            save_path = os.path.join(checkpoint_folder, save_filename)
            torch.save(classifier.cpu().state_dict(), save_path)
            classifier.to(device)

        epoch_test_start_time = time.time()
        classifier.eval()
        with torch.no_grad():
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

            print('\n')
            losses_test += [epoch_loss_test]
            losses_test_unweighted += [epoch_loss_test_unweighted]
            accuracy += [epoch_accuracy]

            if epoch_loss_test < losses_test[min_loss_test_epoch]:
                print('lowest test loss: saving the model at the end of epoch {:d}, iters {:d}'.format(epoch + 1, total_steps))
                save_filename = 'classifier_checkpoint_lowest_test_loss.pth'.format(epoch + 1)
                save_path = os.path.join(checkpoint_folder, save_filename)
                torch.save(classifier.state_dict(), save_path)
                min_loss_test_epoch = epoch

            if epoch_loss_test_unweighted < losses_test_unweighted[min_loss_test_unweighted_epoch]:
                print('lowest test loss (unweighted): saving the model at the end of epoch {:d}, iters {:d}'.format(epoch + 1, total_steps))
                save_filename = 'classifier_checkpoint_lowest_test_loss_unweighted.pth'.format(epoch + 1)
                save_path = os.path.join(checkpoint_folder, save_filename)
                torch.save(classifier.state_dict(), save_path)
                min_loss_test_unweighted_epoch = epoch

            if epoch_accuracy > accuracy[max_accuracy_epoch]:
                print('highest accuracy: saving the model at the end of epoch {:d}, iters {:d}'.format(
                    epoch + 1, total_steps))
                save_filename = 'classifier_checkpoint_highest_accuracy.pth'.format(epoch + 1)
                save_path = os.path.join(checkpoint_folder, save_filename)
                torch.save(classifier.state_dict(), save_path)
                max_accuracy_epoch = epoch

    print("=== train loss ===")
    print(losses_train)
    torch.save(torch.tensor(losses_train), os.path.join(checkpoint_folder, "losses_train.pt"))
    print("=== test loss ===")
    print(losses_test)
    print("=== test loss (unweighted) ===")
    print(losses_test_unweighted)
    test_losses = torch.tensor((losses_test, losses_test_unweighted))
    torch.save(test_losses, os.path.join(checkpoint_folder, "losses_test.pt"))
    torch.save(accuracy, os.path.join(checkpoint_folder, "accuracy.pt"))

    print("min test loss: epoch", min_loss_test_epoch + 1)
    print("min test loss (unweighted): epoch", min_loss_test_unweighted_epoch + 1)
    print("max accuracy: epoch", max_accuracy_epoch + 1)
