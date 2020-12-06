import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
# import torch.nn.functional as F
import utility.processimage
# import matplotlib.pyplot as plt
# import numpy as np
import json
from collections import OrderedDict

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



def create_model(arch, hidden_units):
    '''
        Creates a pretrained model using VGG19 or Densenet161 (archarch, hidden_units) and returns the model
    '''

    # Load a pretrained network (vgg19 or densenet161)
    print("Creating model ...")

    # Load a pretrained model
    if arch.lower() == "vgg13":
        model = models.vgg13(pretrained=True)
        input_features = 25088
    elif arch.lower() == "densenet161":
        model = models.densenet161(pretrained=True)
        input_features = 2208
    else:
        print("Model architecture: {}, is not supported as yet. \n Try vgg19 or densenet161".format(arch.lower()))
        return 0

    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create our classifier to replace the current one in the model

    model.classifier = nn.Sequential(OrderedDict([
                          ('dropout1', nn.Dropout(p=0.2)),
                          ('fc1', nn.Linear(input_features, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_units, 1000)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    print("Done creating the model\n")
    return model



def train_model(model, train_dataloaders, valid_dataloaders, optimizer, epochs, use_gpu):
    '''
        Train model with loss function, optimizer, dataloaders, epochs, GPU. Outputs loss and accuracy

    '''
    print("Training model...\n")
    device = torch.device('cuda' if use_gpu else 'cpu')
    model.to(device)
    # loss function
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        running_loss = 0
        steps = 0
        print_every = 5
        for images, labels in train_dataloaders:
            steps +=1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in valid_dataloaders:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_dataloaders):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_dataloaders):.3f}")
                running_loss = 0
                model.train()



def save_model(model, train_datasets, epochs, optimizer):
    '''
        Saves model to checkpoint.pth file in specifed directory
    '''
    print("Saving model...")

    # Save the train image dataset
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {  'model' : model,
                    'model.classifier': model.classifier,
                    'model.class_to_idx' : model.class_to_idx,
                    'epochs': epochs,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer,
                    'optimizer.state_dict': optimizer.state_dict()
                    }

    torch.save(checkpoint, 'checkpoint.pth')
    print("model Saved to checkpoint.pth")
