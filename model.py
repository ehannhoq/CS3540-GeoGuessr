import torch
import torch.nn as nn
from torchvision import models, transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_model():
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential( # type: ignore
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.Sigmoid(),
        nn.Linear(256, 4)
    )
    return model


def parse_prediction(prediction):
    return {
        'ghana': prediction[0],
        'japan': prediction[1],
        'malaysia': prediction[2],
        'united_states': prediction[3],
    }