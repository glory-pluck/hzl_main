import sys
import torch.nn as nn
import torch
from torchvision.models import resnet101
from tqdm import tqdm
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Identity() # remove final fully connected layer
        # Freeze model parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
    def forward(self, x):
        x = self.resnet(x)
        return x#[batch,2048]
    
def featureExtract(data_loader, device):
    model = MyModel()
    model.to(device)
    model.eval()
    all_features = []
    all_labels = []
    for step, data in enumerate(data_loader):
        # print(len(data[0][0]))
        images,labels= data
        feature = model(images.to(device))
        all_features.extend(feature.data.cpu().numpy())
        all_labels.extend(labels.data.cpu().numpy())
    return all_features,all_labels
# model = MyModel()
# print(model(torch.ones(3,3,128,128)))
    

