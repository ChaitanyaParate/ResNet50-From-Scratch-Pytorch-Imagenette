import torch
import torch.nn as nn

class conv1(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        return self.conv(x)
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels = 64,mid_channels = 128, out_channels = 256,stride =1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out += self.skip(x)

        out = self.relu(out)

        return out
    
class classification(nn.Module):
    def __init__(self, input_dim):
        super(classification, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 10),

        )
    def forward(self, x):
        return self.linear(x)

class ResNet(nn.Module):
    def __init__(self, in_channels = 64 ,mid_channels = 64, out_channels = 256, num_classes = 10, stride = 2):
        super().__init__()
        self.conv1 = conv1()

        self.flow = nn.ModuleList()
        self.classifier = None

        for _ in range(3):
            self.flow.append(Bottleneck(in_channels,mid_channels,out_channels,stride))
            in_channels = out_channels
            stride = 1
        out_channels = out_channels*2
        mid_channels = mid_channels*2
        stride = 2

        for _ in range(4):
            self.flow.append(Bottleneck(in_channels, mid_channels, out_channels,stride))
            in_channels = out_channels
            stride = 1

        out_channels = out_channels*2
        mid_channels = mid_channels*2
        stride = 2

        for _ in range(6):
            self.flow.append(Bottleneck(in_channels, mid_channels, out_channels,stride))
            in_channels = out_channels
            stride = 1

        out_channels = out_channels*2
        mid_channels = mid_channels*2
        stride = 2

        for _ in range(3):
            self.flow.append(Bottleneck(in_channels, mid_channels, out_channels, stride))
            in_channels = out_channels
            stride = 1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   
        
        self.fc = nn.Linear(out_channels, num_classes)
        

    def forward(self, x):

        x = self.conv1(x)
        
        for flo in self.flow:
            x = flo(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)

        if self.classifier is None:
            input_dim = x.shape[1]
            
            self.classifier = classification(input_dim).to(x.device)

        x = self.classifier(x)

        return x

def test():
    model = ResNet(num_classes=1000)
    x = torch.randn(3, 3, 224, 224)
    print(model(x).shape) 
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total number of parameters: {total_params}")

if __name__ == "__main__":
    test()