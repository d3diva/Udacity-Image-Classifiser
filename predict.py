import pmodel
import torchvision
from torchvision import datasets, transforms, models
import utility.jsonloader
import argparse

# Load a saved model and predict image

# Create the parser
parser = argparse.ArgumentParser(description="Load a Neural Network")
# Data directory
parser.add_argument('data_directory',
                    help="Path to the image files")
# Path to checkpoint.pth
parser.add_argument('checkpoint',
                    help="Path to checkpoint.pth file.")
# Number ot top classes
parser.add_argument('--top_k', default=1, type=int,
                    help="Most likley classes to return for the predictions")
# Path to json file with category names
parser.add_argument('--category_names', default = './cat_to_name.json',
                    help="json file with path to load category names")
# GPU True or False
parser.add_argument('--gpu', default=False, action='store_true',
                    help="gpu - True/False")

# Collect the arguments
pargs = parser.parse_args()
data_directory = pargs.data_directory
checkpoint = pargs.checkpoint
top_k = pargs.top_k
category_name = pargs.category_names
use_gpu = pargs.gpu

# Load the model
model = pmodel.load_checkpoint(checkpoint)

# Load the content of the json file
catonames = utility.jsonloader.load_json(category_name)

# Predict

classes = pmodel.predict(catonames, data_directory, model, top_k)


top_flowers = [catonames[i] for i in classes]
class_flowers = list(zip(classes, top_flowers ))
print(class_flowers)
print('Classes and Top flower :')
for f in class_flowers:
    print( f"Class : {f[0]} " + "| "
          f"Flower : {f[1]} ")