import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 6, 5) # 0
        self.conv3 = nn.Conv2d(6, 16, 5) # 2
        self.conv5 = nn.Conv2d(16, 120, 5) # 4
    
        self.fc6 = nn.Linear(120, 84) # 6
        self.fc7 = nn.Linear(84, 10) # 8
    
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv5(x))
        x = x.view(-1, 120)
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x
    
class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet2, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 6, 5) # 0
        self.conv3 = nn.Conv2d(6, 16, 5) # 2
        self.conv5 = nn.Conv2d(16, 120, 5) # 4
            
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.LeakyReLU(),
            nn.Linear(84, 10)
            )
    
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.LeakyReLU(),
            nn.Linear(84, 10)
            )
        
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv5(x))
        x = x.view(-1, 120)
        
        return self.fc1(x), self.fc2(x)


class LeNet_rep(nn.Module):
    def __init__(self):
        super(LeNet_rep, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 6, 5) # 0
        self.conv3 = nn.Conv2d(6, 16, 5) # 2
        self.conv5 = nn.Conv2d(16, 120, 5) # 4
    
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv5(x))
        x = x.view(-1, 120)
        return x


class MLP(nn.Module):
    def __init__(self, out=10):
        super(MLP, self).__init__()
        
        self.fc6 = nn.Linear(120, 84) # 6
        self.fc7 = nn.Linear(84, out) # 8
        
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x

    
if __name__ == "__main__":
    a = torch.randn((1,1,32,32))
    b = LeNet2()
    c = b(a, True)
    print(c.shape)
    
    d = b(a, classify=False)
    print(d.shape)
    