
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import optim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA_PATH = 'images/train'
TEST_DATA_PATH = 'images/test'

num_epoch = 5
batch_size = 10
learning_rate = 0.001
num_classes = 4 

transform = transforms.Compose([
    transforms.Resize((224,224), antialias=True),   
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)

train_loaded = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loaded = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

#classes = ('sway-back posture','forward-head posture','correct posture', 'hunched posture')

#image resolution = 1920x1080
#image size = 1472x3264 
#new image size = 664x428, after transforms 224, 224

#input_size = Input(shape = 1472, 3264, 3)

#(W-F + 2P)/S + 1
#INPUT - FILTER + 2 * PADDING/ STRIDE + 1
#3264x1427 - 32x32 + 2*0 /0 + 1  

class PostureNet(nn.Module):
       def __init__(self, num_classes= 4): 
                super(PostureNet, self).__init__()
                self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1), 
                        nn.BatchNorm2d(5), 
                        nn.ReLU(),
                        nn.MaxPool2d(2,2)
                )
                self.layer2 = nn.Sequential(
                       nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, stride=1),
                       nn.BatchNorm2d(8),
                       nn.ReLU(),
                       nn.MaxPool2d(2,2)
                )
                self.layer3 = nn.Sequential(
                       nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
                       nn.BatchNorm2d(16),
                       nn.ReLU(),
                       nn.MaxPool2d(2,2)
                )
                self.layer4 = nn.Sequential(
                       nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
                       nn.BatchNorm2d(32),
                       nn.ReLU(),
                       nn.MaxPool2d(2,2)
                )
                self.layer5 = nn.Sequential(
                       nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
                       nn.BatchNorm2d(64),
                       nn.ReLU(),
                       nn.MaxPool2d(2,2)
                )
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(1600, 300)
                self.fc2 = nn.Linear(300, num_classes)

       def forward(self, x):
              x = self.layer1(x)
              x = self.layer2(x)
              x = self.layer3(x)
              x = self.layer4(x)
              x = self.layer5(x)
              x = self.flatten(x)
              x = self.fc1(x)
              x = self.fc2(x)
              return x 

model = PostureNet(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
       print(f"Epoch[{epoch + 1}/{num_epoch}]")
       for batch_index, (data, targets) in enumerate(tqdm(train_loaded)):
              data = data.to(device)
              targets = targets.to(device)

              scores = model(data)
              loss = criterion(scores, targets)

              optimizer.zero_grad()
              loss.backward()

              optimizer.step()

def check_accuracy(loader, model):
 
       num_correct = 0
       num_samples = 0
       # Set the model to evaluate 
       model.eval()

       with torch.no_grad():
              for inputs, labels in loader:
                     inputs = inputs.to(device)
                     labels = labels.to(device)

                     scores = model(inputs)
                     _,predictions = torch.max(scores.data, 1)
                     
                     num_correct += (predictions == labels).sum().item()
                     num_samples += (labels.size(0))

              accuracy = float(num_correct) / float(num_samples) * 100
              print(f"Got {num_correct}/{num_samples} with accuracy {accuracy: .2f}%")
              
       model.train()

'''model = PostureNet()
x = torch.FloatTensor()
with torch.no_grad():
       x = x.transpose()
       print(model(x))
       traced_cell = torch.jit.trace(model, (x))
torch.jit.save(traced_cell, "model_classification.pth")''' #THIS STILL REQUIRES ADJUSTMENTS WITH TRANSPOSING TO THE CORRECT DIMENSIONS 

check_accuracy(train_loaded, model)
check_accuracy(test_loaded, model)


        