import torch
import torchvision
from torchvision import models
from PIL import Image
import utility.processimage


def load_checkpoint(checkpoint):
    '''
        Loads model from checkpoint.pth, returns model
    '''
    print("Loading model...")

    # Load model
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    model.classifier = checkpoint['model.classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
    epochs = checkpoint['epochs']
    model.eval()

    return model

model = load_checkpoint('checkpoint.pth')


def predict(catonames, image_path, model, topk):
    '''
        Predict the class of image using a trained model.

    '''
    # model.to('cpu')
    device = 'cpu'

    model.eval()
    model.to(device)

    img = utility.processimage.process_image(image_path)
    img = img.unsqueeze(0)
    inputs = img.to(device)
    output = model.forward(inputs)
    p = torch.exp(output)

    pred = torch.topk(p, topk)[0].tolist()[0]
    index = torch.topk(p, topk)[1].tolist()[0]
    item = []
    for i in range(len(model.class_to_idx.items())):
        item.append(list(model.class_to_idx.items())[i][0])

    classes = []
    for i in range(topk):
        classes.append(item[index[i]])
    
    return classes
