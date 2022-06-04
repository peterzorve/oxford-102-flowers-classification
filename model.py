
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class Flowers_Classifier(nn.Module):
    def __init__(self):
        super(Flowers_Classifier, self).__init__()   

        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32,  kernel_size=5,  stride=1,  padding=0)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=5,  stride=2,  padding=0)  
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=5,  stride=1,  padding=0)                                          

        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1)

        self.fc1 = nn.Linear(32*46*46, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 102)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1( F.relu( self.conv1(x) ))
        x = self.pool1( F.relu( self.conv2(x) ))
        x = self.pool1( F.relu( self.conv3(x) ))

        # print(x.shape)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        x = self.softmax(x)

        return x

# fake_data = torch.rand((32, 1, 128, 128))
# model = Flowers_Classifier()

# print(model.forward(fake_data))