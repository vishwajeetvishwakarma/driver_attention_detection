from fastai.vision.all import * 
from fastai import *  
from PIL import Image
from torchvision import models,transforms,datasets
import torch 
import torch.nn as nn
def label_func(x): return x.parent.name    
def load_model_vgg():
    model = load_learner('./export.pkl')
    return model

def load_model_resnet(): 
    device = 'cpu'
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 10) #No. of classes = 10
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load('model_driver_resnet.pth', map_location=torch.device('cpu')))
    return model_ft


class_dict = {0 : "safe driving",
              1 : "texting - right",
              2 : "talking on the phone - right",
              3 : "texting - left",
              4 : "talking on the phone - left",
              5 : "operating the radio",
              6 : "drinking",
              7 : "reaching behind",
              8 : "hair and makeup",
              9 : "talking to passenger"}

output_label = {'c0': 'normal driving',
'c1': 'texting - right',
'c2': 'talking on the phone - right',
'c3': 'texting - left',
'c4': 'talking on the phone - left',
'c5': 'operating the radio',
'c6': 'drinking',
'c7':' reaching behind',
'c8': 'hair and makeup',
'c9': 'talking to passenger'}

def predict_with_image(image, model):
    image = Image.open(image).convert('RGB')
    if model == 'resnet50':
        model_ft = load_model_resnet()
        transform = transforms.Compose([transforms.Resize((400, 400)),
                                 transforms.RandomRotation(10),
                                 transforms.ToTensor()])
        model_ft.eval()
        image = transform(image)
        image = image.unsqueeze(0)
        output = model_ft(image)
        proba = nn.Softmax(dim=1)(output)
        proba = [round(float(elem),4) for elem in proba[0]]
        label = f'model predicted uploaded image is` {class_dict[proba.index(max(proba))]}` with probability `{max(proba)}`'
        return label
    else: 
        model_ft = load_model_vgg()
        prediction  = model_ft.predict(np.array(image))
        label = f'model predicted uploaded image is `{output_label[prediction[0]]}` with probability `{prediction[2].max()}`'
        return label