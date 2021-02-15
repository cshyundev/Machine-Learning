import torch
import torchvision as tr
import torchvision.transforms as transforms 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary


# set hyperparameters

lr = 0.01 # reduce slowely
batch_size = 16
n_epoch = 40

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data : cifar10

transforms = {
    'train' : transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]),
    'test' : transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
}

trainset = tr.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transforms['train'])
testset = tr.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transforms['test'])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# build model : vgg19 

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.layer4 = nn.Sequential(
             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)            
        )
        self.layer5 = nn.Sequential(
             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)            
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(50176,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.fc3 = nn.Linear(2048,10)
        self.softmax = nn.Softmax()

    def forward(self,x):
        print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        print(x.shape)
        x = x.view(-1)
        print(x.shape)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(x)

        print(x.shape)

        return x


# net = VGG19().to(device=device)
net = tr.models.vgg19(pretrained=True).to(device)

summary(net, batch_size=batch_size, input_size=(3,224,224))

optimizer = torch.optim.Adam(net.parameters(), lr= lr)
criterion = nn.CrossEntropyLoss() 

for epoch in range(20):
    running_loss = 0.
    for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.to(device) 
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    print('[%d] loss: %.3f' % (epoch + 1, running_loss)  )

print('Finished Training')


# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Acc : %d %%' % (100* correct / total))


        










