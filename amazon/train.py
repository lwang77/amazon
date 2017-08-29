#resnet34 with sgd and pytorch, on GPU
#this one will only train the first 35000 images (set shuffle=False in dataloader)

import numpy as np # linear algebra
import scipy as scipy
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import torch
import torch.nn as nn
from torchvision import models

import shutil
from PIL import Image

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer

from resnet import Resnet as Net
from dataset import TrainDataset as TrainDataset


IMG_PATH = 'train-jpg/'
IMG_EXT = '.jpg'
TRAIN_DATA = 'train_v2.csv'

split = 35000

#data flips and such
def augment(x, u=0.75):
    if random.random()<u:
        if random.random()>0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)
        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=1)

    x = randomFlip(x, u=0.5)
    x = randomTranspose(x, u=0.5)
    x = randomContrast(x, limit=0.2, u=0.5)
    #x = randomSaturation(x, limit=0.2, u=0.5),
    x = randomFilter(x, limit=0.5, u=0.2)
    return x


#personalized loss
class PenalizedMultiLabelSoftMarginLoss(nn.modules.Module):
    def __init__(self, penalty=1):
        super(PenalizedMultiLabelSoftMarginLoss, self).__init__()
        self.penalty = penalty
    def forward(self, input, target):
        
        #THE INFAMOUS THING THAT TRIPPED ME UP omg...................
        #clamped_input = input.clamp(min=0)
        #loss = target * (1 / (clamped_input.neg().exp() + 1)).log() + self.penalty * (1 - target) * (clamped_input.neg().exp() / (clamped_input.neg().exp() + 1)).log()

        loss = target * (1 / (input.neg().exp() + 1)).log() + self.penalty * (1 - target) * (input.neg().exp() / (input.neg().exp() + 1)).log()
        return loss.mean().neg()

# loss ----------------------------------------
def multi_criterion(logits, labels):
    loss = PenalizedMultiLabelSoftMarginLoss(penalty=4)(logits, Variable(labels))
    return loss


#reshape images to 128x128
transformations = transforms.Compose([transforms.ToTensor()])
dset_train = TrainDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations,                                     transform=[
                                        lambda x: augment(x))


#pytorch's custom dataloader class
train_loader = DataLoader(dset_train,
                          batch_size=64,
                          shuffle=False,
                          num_workers=1, # 1 for CUDA
                         pin_memory=True # CUDA only
                        )


model = Net(pretrained=True);


optimizer = torch.optim.SGD([{'params': base_params},
                              {'params': model.fc.parameters(), 'lr': 0.01}], lr=0.001, momentum=0.9)


def lr_scheduler(optimizer, epoch):
    """Epochs 1-8: 0.01
       Epochs 9-12: 0.001
       Epochs 13-16: 0.0001
    """
    #learning rate for top fc layer
    if epoch == 9 or epoch == 13:
        print("changing learning rate")
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = old_lr / 10
    return optimizer

#returns average loss and average accuracy over entire epoch
#if phase == train, also backprop. otherwise just return validation accuracy / loss
def train(epoch):
    #used to calculate accuracy and loss for the epoch
    running_loss = 0.0
    running_corrects = 0
    
    #sets model to training mode
    model.train(True)
    
    #used for calculating epoch loss
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        
        loss.backward()
        optimizer.step()
        
        #predictions vs labels. calculate accuracy and loss.
        preds = output.data.cpu().numpy()
        #can change thresholds later
        preds = ((preds>0.2).astype(int))
        labels = target.data.cpu().numpy().astype(int)
                        
        running_loss += loss.data[0]
        running_corrects += np.sum(preds == labels.data)
        
        #print out information in middle of batch
        #need to multiply (batch_idx+1) * len(data) * 17 because that's the number of labels
        #for a one batch of inputs. 17 labels per picture, len(data) is batch_size. no. of batches 
        #so far = (batch_idx + 1)
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAccuracy: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], 
                running_corrects / (17 * (batch_idx + 1) * len(data))))
        
        #will be set to last batch_idx
        num_batches = batch_idx + 1
        
    epoch_loss = running_loss / num_batches
    epoch_acc = running_corrects / (17 * split)
    return (epoch_loss, epoch_acc)

for epoch in range(1, 17):
    optimizer = lr_scheduler(optimizer, epoch)
    epoch_loss, epoch_acc = train(epoch)
    opt_save_name = ''.join(['optimizier_epoch_', str(epoch), '_state_dict.pt'])
    model_save_name = ''.join(['model_epoch_', str(epoch), '_state_dict.pt'])
    torch.save(optimizer.state_dict(), opt_save_name)
    torch.save(model.state_dict(), model_save_name)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))







