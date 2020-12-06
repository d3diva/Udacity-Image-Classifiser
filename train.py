# Imports here
import torch
from torch import nn as nn
from torch import optim as optim
import tmodel
import utility.dataloader
import argparse


# Create the parser
parser = argparse.ArgumentParser(description="Trained a deep neural network")
# Data directory with image files to train with
parser.add_argument('data_directory', default='flowers',
                    help="Path to the image files to train with.")
# Drectory to the image files to train with
parser.add_argument('--save_dir', default='./',
                    help="Path to save the checkpoint")
# Choose the architecture
parser.add_argument('--arch', default="vgg13",
                    help="The architecture to train the model with. Can be: vgg13 or densenet161")
# Set Learning Rate
parser.add_argument('--learning_rate', type=float, default="0.001",
                    help="Learning rate")
# Set Hidden Units
parser.add_argument('--hidden_units', type=int, default=512,
                    help="Units in the hidden layer")
# Set Training Epochs
parser.add_argument('--epochs', type=int, default=4,
                    help="Number of training epochs")
# Set Batch size
parser.add_argument('--batch_size', type=int, default=32,
                    help="The size of the batches for training")
# 5. Choose the GPU for training
parser.add_argument('--gpu', default=False, action='store_true',
                    help="GPU for training.")

pargs = parser.parse_args()
data_directory = pargs.data_directory
save_directory = pargs.save_dir
arch = pargs.arch
learning_rate = pargs.learning_rate
hidden_units = pargs.hidden_units
epochs = pargs.epochs
batch_size = pargs.batch_size
gpu = pargs.gpu

# Create the data loaders
train_dataloaders, valid_dataloaders, test_dataloaders, train_datasets = utility.dataloader.load_image_data(data_directory, batch_size)

# Create the model
model = tmodel.create_model(arch, hidden_units)

# If model is created procide with training
if model != 0:

    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    # Train model with validation
    tmodel.train_model(model, train_dataloaders, valid_dataloaders, optimizer, epochs, gpu)

    # Save model
    tmodel.save_model(model, train_datasets, epochs, optimizer)
    
