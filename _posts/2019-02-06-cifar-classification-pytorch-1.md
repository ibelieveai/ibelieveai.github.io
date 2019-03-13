---
title: "Building CNN on CIFAR-10 dataset using PyTorch: 1"
excerpt: "Classifying CIFAR images using convnets"
author: Praneeth Bellamkonda
date: 2019-02-006
tags: [Deep Learning, CNN, Pytorch, Computer Vision]
header:
    image: "/images/densenet/deeplearning.jpg"
---

{% include base_path %}
{% include toc %}

## The CIFAR-10 dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks

## Test for CUDA
Since these are larger (32x32x3) images, it may prove useful to speed up your training time by using a GPU. CUDA is a parallel computing platform and CUDA Tensors are the same as typical Tensors, only they utilize GPU's for computation.


```python
import torch
import numpy as np
import time

# check if cuda is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
  print("CUDA is not available. Training on CPU")
else:
  print("CUDA available. Training on GPU")
```

    CUDA available. Training on GPU
    

## Loading the Dataset
Downloading may take a minute. We load in the training and test data, split the training data into a training and validation set, then create DataLoaders for each of these sets of data.


```python
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

#number of subprocess to use for data loading
num_workers = 0

# how many samples per batch to load
batch_size = 20

# percentage of training set to use as validaion
valid_size = 0.2

# convert data to a normalized torch tensor

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

#choose the training and test datasets

train_data = datasets.CIFAR10('data', train=True, download= True, transform=transforms)
test_data = datasets.CIFAR10('data', train=False, download= True, transform=transforms)

# Create validation dataset

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)

split = int(np.floor(valid_size*num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
```

      0%|          | 0/170498071 [00:00<?, ?it/s]

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar-10-python.tar.gz
    

    100%|█████████▉| 170418176/170498071 [02:43<00:00, 899116.15it/s]

    Files already downloaded and verified
    

## Visualize a Batch of Training Data


```python
import matplotlib.pyplot as plt
%matplotlib inline

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
```


```python
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]]) 
```


![png](images/cifar/output_6_0.png)


## Define the Network Architecture


```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    
    # Convolutional Layer
    self.conv1 = nn.Conv2d(3,16,3, padding=1)
    self.conv2 = nn.Conv2d(16,32,3,padding=1)
    self.conv3 = nn.Conv2d(32,64,3,padding=1)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(64*4*4,120)
    self.fc2 = nn.Linear(120, 60)
    self.fc3 = nn.Linear(60,10)
    self.dropout = nn.Dropout(0.25)
    
  def forward(self, x):
    
    x =  self.pool(F.relu(self.conv1(x)))
    x =  self.pool(F.relu(self.conv2(x)))
    x =  self.pool(F.relu(self.conv3(x)))
    x =  x.view(-1,64*4*4)
    x =  self.dropout(x)
    x = F.relu(self.fc1(x))
    x =  self.dropout(x)
    x = F.relu(self.fc2(x))
    x =  self.dropout(x)
    x = self.fc3(x)
    return x
  
  
model = Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
  model.cuda()

```

    Net(
      (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=1024, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=60, bias=True)
      (fc3): Linear(in_features=60, out_features=10, bias=True)
      (dropout): Dropout(p=0.25)
    )
    

## Specify Loss Function and Optimizer


```python
import torch.optim as optim
# Specify loss function
criterion = nn.CrossEntropyLoss()
# Specify optmizer
optimizer = optim.SGD(model.parameters(),lr=0.01, momentum =0.9)
```

## Train the Network

Remember to look at how the training and validation loss decreases over time; if the validation loss ever increases it indicates possible overfitting.


```python
#number of epochs to train the model
n_epochs = 50

valid_loss_min = np.Inf #to track the changes in the validation loss

for epoch in range(1,n_epochs+1):
  start_time  = time.time()
  #keep track of training and validation loss
  train_loss = 0 
  valid_loss = 0
  
  ################
  #Training model#
  ################
  
  model.train()
  for data,target in train_loader:
    if train_on_gpu:
      data, target = data.cuda(), target.cuda()
    
    #Clear the gradients of all optimized variables
    optimizer.zero_grad()
    #Forward pass
    output = model(data)
    #calculate loss
    loss = criterion(output,target)
    #backward pass
    loss.backward()
    #perform optimizer step
    optimizer.step()
    #update training loss
    train_loss +=loss.item()*data.size(0)
    
  #####################
  #Validate the model #
  #####################
  
  model.eval()
  for data,target in valid_loader:
    if train_on_gpu:
      data, target = data.cuda(), target.cuda()
    # perform forward pass on validation data
    
    output = model(data)
    # calculate the loss
    loss = criterion(output,target)
    # update average validation loss
    valid_loss += loss.item()*data.size(0)
    
  
  # calculate average loss
  train_loss = train_loss/len(train_loader.dataset)
  valid_loss = valid_loss/len(valid_loader.dataset)
  
  #calculate time for each epoch 
  end_time = time.time()
  delta = start_time-end_time
  
  # print training / validation metrics
  print('Epoch:',epoch,' took ',round(delta,3),' secs','----Training loss:',round(train_loss,3),'----Validation loss:',round(valid_loss,3))
  

  #save model if validation loss is decreased
  
  if valid_loss <= valid_loss_min:
        print('Validation loss decreased (',round(valid_loss_min,3),')--->',round(valid_loss,3),' savind model as model_cifar.pt')
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

```

    Epoch: 1  took  -23.389  secs ----Training loss: 0.938 ----Validation loss: 0.226
    Validation loss decreased ( inf )---> 0.226  savind model as model_cifar.pt
    Epoch: 2  took  -22.683  secs ----Training loss: 0.942 ----Validation loss: 0.243
    Epoch: 3  took  -22.611  secs ----Training loss: 0.96 ----Validation loss: 0.218
    Validation loss decreased ( 0.226 )---> 0.218  savind model as model_cifar.pt
    Epoch: 4  took  -22.692  secs ----Training loss: 0.96 ----Validation loss: 0.24
    Epoch: 5  took  -23.031  secs ----Training loss: 0.989 ----Validation loss: 0.227
    Epoch: 6  took  -22.69  secs ----Training loss: 0.999 ----Validation loss: 0.239
    Epoch: 7  took  -22.694  secs ----Training loss: 1.059 ----Validation loss: 0.243
    Epoch: 8  took  -22.736  secs ----Training loss: 1.048 ----Validation loss: 0.247
    Epoch: 9  took  -22.777  secs ----Training loss: 1.065 ----Validation loss: 0.234
    Epoch: 10  took  -22.793  secs ----Training loss: 1.062 ----Validation loss: 0.272
    Epoch: 11  took  -22.77  secs ----Training loss: 1.1 ----Validation loss: 0.263
    Epoch: 12  took  -22.741  secs ----Training loss: 1.173 ----Validation loss: 0.253
    Epoch: 13  took  -22.766  secs ----Training loss: 1.164 ----Validation loss: 0.281
    Epoch: 14  took  -22.697  secs ----Training loss: 1.2 ----Validation loss: 0.26
    Epoch: 15  took  -22.647  secs ----Training loss: 1.272 ----Validation loss: 0.284
    Epoch: 16  took  -22.616  secs ----Training loss: 1.343 ----Validation loss: 0.316
    Epoch: 17  took  -22.591  secs ----Training loss: 1.388 ----Validation loss: 0.328
    

```python
model.load_state_dict(torch.load('model_cifar.pt'))
```

## Test the Trained Network
Test your trained model on previously unseen data! A "good" result will be a CNN that gets around 70% (or more, try your best!) accuracy on these test images.


```python
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
```

    Test Loss: 1.128265
    
    Test Accuracy of airplane: 70% (705/1000)
    Test Accuracy of automobile: 77% (771/1000)
    Test Accuracy of  bird: 42% (426/1000)
    Test Accuracy of   cat: 58% (585/1000)
    Test Accuracy of  deer: 59% (594/1000)
    Test Accuracy of   dog: 43% (438/1000)
    Test Accuracy of  frog: 70% (708/1000)
    Test Accuracy of horse: 70% (708/1000)
    Test Accuracy of  ship: 74% (746/1000)
    Test Accuracy of truck: 79% (795/1000)
    
    Test Accuracy (Overall): 64% (6476/10000)
    

## What are our model's weaknesses and how might they be improved?

Model performs better on vehicles than on animals. As animals vary in color and size to improve the model either we can increase the animal pictures in the dataset or increase the number of neurons so that our model understands the complex patterns inside the animal images.

## Visualize Sample Test Results


```python
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
```


![png](images/cifar/output_18_0.png)

## Resources & Acknowledgements
[https://www.udacity.com/course/deep-learning-pytorch--ud188](https://www.udacity.com/course/deep-learning-pytorch--ud188)
[http://cs231n.github.io/convolutional-networks/#layers](http://cs231n.github.io/convolutional-networks/#layers)






