from collections import OrderedDict
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm

import copy
import matplotlib.pyplot as plt
import numpy as np
import os 
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from model import model
from util import get_device


def disp_image(data_loader, index):
    for i, (inputs, labels) in enumerate(data_loader):
        if i == index:
            inputs = inputs[:,torch.arange(inputs.shape[1]-1,-1,-1),:]
            images = inputs.numpy()

            print("Label: {}".format(labels[i]))
            plt.imshow(np.transpose(images[i], [1, 2, 0]))
            break


def load_DataLoader(path_train_loader, path_validation_loader, debug=False):
    start = time.time()
    t_loader = torch.load(path_train_loader)
    v_loader = torch.load(path_validation_loader)
    elapsed_time = time.time() - start
    return t_loader, v_loader, elapsed_time


def get_transfer_learning_model(num_classes):
    c = nn.CrossEntropyLoss()   # loss function
    opt = optim.SGD(target_model.parameters(), lr=0.001, momentum=0.9) # optimizer
    lr = lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)   # learning rate scheduler
    return c, opt, lr


def train_model(
    model, 
    device, 
    train_data_loader,
    valid_data_loader, 
    criterion, optimizer, scheduler, num_epochs=5):
    """
    training

    Parameters
    --------------
    model : DogClassificationModel
        Network model to be trained.
    device : device
        cuda or cpu
    train_data_loader : dataloader
        dataloader for training
    valid_data_loader : dataloader
        dataloader for validation
    criterion : 
        Loss function.
    optimizer :
        Optimizer.
    scheduler : 
        Learning rate scheduler.
    num_epochs : int
        The number of epochs.

    Returns
    --------------
    model : DogClassificationModel
        Trained model.
    """
    since = time.time()
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        bar = tqdm(total = len(train_data_loader))
        bar.set_description("Epoch: {}/{}".format(epoch+1, num_epochs))

        """
        Training Phase
        """
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for j, (inputs, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()
            tmp_loss_item = 0.0

            # training
            with torch.set_grad_enabled(True):
                outputs = model(inputs.to(device))
                torch.cuda.empty_cache()

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels.to(device))

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                tmp_loss_item = loss.item()

            # statistics
            running_loss += tmp_loss_item * inputs.size(0)
            running_corrects += torch.sum(preds.to('cpu') == labels.data)

            # progress bar
            bar.update(1)
            tmp_loss = float(running_loss / (j+1)) / 32         # 32: mini-batch size
            tmp_acc = float(running_corrects // (j+1)) / 32
            bar.set_postfix(OrderedDict(loss=tmp_loss, acc=tmp_acc))

        # update learning rate scheduler
        scheduler.step()

        dataset_size = len(train_data_loader.dataset) 
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        """
        Validation Phase
        """
        model.eval()  # Set model to validation mode

        val_running_loss = 0.0
        val_running_corrects = 0

        # Iterate over data.
        for inputs, labels in val_data_loader:
            val_inputs = inputs.to(device)
            val_labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.no_grad():
                val_outputs = model(val_inputs)
                _, preds = torch.max(val_outputs, 1)
                loss = criterion(val_outputs, val_labels)

            # statistics
            val_running_loss += loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(preds == val_labels.data)

        dataset_size = len(val_data_loader.dataset) 
        val_epoch_loss = val_running_loss / dataset_size
        val_epoch_acc = val_running_corrects.double() / dataset_size

        print('VALIDATION  Loss: {:.4f} Acc: {:.4f}'.format(val_epoch_loss, val_epoch_acc))                
        print("Elapsed time: {} [sec]".format(time.time() - since))

        # deep copy the model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    num_classes = 121
    print("Start Training.")

    # device
    device = get_device.get_device()   # cuda or cpu
    print("device: {}".format(device))

    # loading dataset
    print("Dataset load: Start.")
    since = time.time()
    train_data_loader, val_data_loader, t = load_DataLoader(
        "./Dataset/train_loader.pth", "./Dataset/validation_loader.pth")
    print("Done. {} [sec]".format(time.time() - since))

    # model
    #   downloading pretrained model
    base_model = models.resnext101_32x8d(pretrained=True)
    # model for dog classification
    target_model = model.DogClassificationModel(model=base_model, num_classes=num_classes, mean=0.5, std=0.25)

    # loss function, optimizer and lr
    criterion, opt, exp_lr_scheduler = get_transfer_learning_model(num_classes)

    # training
    print("Training start.")
    since = time.time()
    trained_model = train_model(
        model=model_ft, device=device, 
        train_data_loader=train_data_loader, valid_data_loader=val_data_loader,
        criterion=criterion, 
        optimizer=opt, scheduler=exp_lr_scheduler, num_epochs=5)
    print("Done. {} [sec]".format(time.time() - since))

    # save model
    torch.save(trained_model.state_dict(), "./trained_model")
