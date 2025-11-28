#!/usr/bin/env python3
"""
   model.py

   UNSW ZZEN9444 Neural Networks and Deep Learning

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from config import device

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################

"""
This function applies the necessary transformations to the input images
based on the specified mode (train or test).

There are three data transformations that occur to the dataset images before
both training and testing:

1. Resizing the image to 300x300 pixels.
This is required because the selected pre-trained classification model -
RegNetY_800MF - model expects input images of this size.

2. Convert the image to a tensor.
This is necessary because the PyTorch model expects input in the form of tensors.

3. Normalize the image with the specified mean and standard deviation.
Normalization is important for ensuring that the input data has a consistent
scale and distribution, helping the convergence of the model during training.
The values selected for normalization are based on the dataset. Specifically,
the mean and standard deviation of the training set were calculatedand used
to normalize both the training and test datasets before training occurs.
Based off experimentation results, the actual dataset's normalization values
produced highest test accuracy compared to ImageNet's normalization values or
no normalization.

The transform function takes in the parameter "mode" which specifies whether the 
transformation to be applied is being used for training or testing.
"""

def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((300, 300)), # RegNetY_800MF expects 300x300 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4806, 0.4338, 0.3918],
                               std=[0.2422, 0.2343, 0.2270]) # Dataset normalization values
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.Resize((300, 300)),  # RegNetY_800MF expects 300x300 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4806, 0.4338, 0.3918],
                               std=[0.2422, 0.2343, 0.2270]) # Dataset normalization values
        ])

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
"""
This function defines the image classification model architecture using a
pre-trained image classification CNN pytorch model RegNet_Y_800MF. The model
has been pre-trained on ImageNet, and is a powerful feature extractor, as per
the documented performance on the ImageNet dataset.

This pre-trained model was selected after extensive experimentation, using different
pre-trained image classification pytorch models and hyperparameter tuning. 

Both the model architecture and the model weights are utilised in this function.
The model has been defined in such a way to be fine-tuned (no freezing of any
model layers). When this model is trained, all weights in all layers will be updated.

The final layer of the model (the fully connected layer) has been replaced to match 
the number of output classes (8 classes) for the specific dataset that will be used. 

The Network function takes in the input "nn.Module" which is a base class for Pytorch
models.
"""

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Specifying the model and weights of the pre-trained classification model RegNet_Y_800MF
        weights = models.RegNet_Y_800MF_Weights.IMAGENET1K_V2 # Pre-trained weight values
        self.regnet = models.regnet_y_800mf(weights=weights) # Pre-trained model definition
       
        # Determining the number of input features that RegNet_Y_800MF has in the final layer
        num_features = self.regnet.fc.in_features 
        # Replacing the final classifier layer to match the number of output classes
        self.regnet.fc = nn.Linear(num_features, 8)

    def forward(self, input):
        return self.regnet(input)

net = Network()

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
"""
An optimizer function helps update the model parameters during training to 
minimize the loss function. The AdamW optimizer is used for training the model, 
based on experimentation results. 

The learning rate is a hyperparameter that controls how much to change the model in 
response to the estimated error each time the model weights are updated. 
Based on experimentation results, a learning rate of 0.0001 was selected.

Weight decay is a regularization technique used to prevent overfitting by adding a 
penalty to the loss function based on the size of model weights. 
Based on experimentation results, a weight decay of 0.01 was selected.

The loss function is the type of error function that will be aimed to be minimised. 
The loss function selected for training is CrossEntropyLoss, which is suitable for 
multi-class classification tasks.
"""
optimizer = optim.AdamW(net.parameters(), lr=0.0001, weight_decay=0.01)

loss_func = nn.CrossEntropyLoss()
############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
###########################################################################

def weights_init(m):
    return

############################################################################
#######              Metaparameters and training options              ######
############################################################################

"""
The metaparameters and training options are defined here.

- train_val_split: This parameter defines the proportion of the dataset to include 
in the training and validation split. Including a validation set helps monitor the 
model's performance during training and prevent overfitting. The selected value for this
parameter is 0.8 to ensure a good balance between training and validation data.

- batch_size: This parameter defines the number of samples that will be processed 
together in one iteration during training. A larger batch size can speed up training 
but requires more memory, which was a hardware constraint when training. 
Based on experimentation and hardware constraints, the selected value for this parameter is 64.

- epochs: This parameter defines the number of times the entire training dataset will be 
passed through the model during training. More epochs can lead to better performance but 
also increase the chance of overfitting. 
Based on experimentation, the selected value for this parameter is 20.

A learning rate scheduler was implemented to help adjust the learning rate during training to 
improve convergence. In this case, the scheduler a CosineAnnealingLR was implemented. The parameter
eta_min was set to 1e-6 to allow for model fine-tuning.
"""

dataset = "./data"
train_val_split = 0.8
batch_size = 64
epochs = 20

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)