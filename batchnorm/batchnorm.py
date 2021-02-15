import torch
import torchvision as tr
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
import time 

# We will compare convolutional neural network model with batchnormalization and without it
#dataset : MNIST 
#optimizer : Adam
#epoch : 20

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu") 
# set hyperparameter
batch_size = 200
lr = 1e-4
n_epochs = 20

# Load data 

trainset = tr.datasets.MNIST(root='./mnist', train=True, download=False, transform=tr.transforms.ToTensor())
testset = tr.datasets.MNIST(root='./mnist', train=False, download=False, transform=tr.transforms.ToTensor())


trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(testset, shuffle=False, batch_size=batch_size)



# Build CNN model without batchnormalization : vanilla model 

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()  
        self.pool = nn.MaxPool2d(2,2)
        self.conv1_1 = nn.Conv2d(1,32,kernel_size=3,stride=1 ,padding=1)
        self.conv1_2 = nn.Conv2d(32,32,kernel_size=3,stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(64,128, kernel_size=3, stride=1 , padding=1)
        self.conv3_2 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128*3*3,512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128,10)

    def forward(self,x):
        # first block : conv1_1 => conv1_2 => maxpool
        x = F.relu(self.conv1_1(x))
        x = self.pool(F.relu(self.conv1_2(x)))
        #second block : conv2_1 => conv2_2 => maxpool
        x = F.relu(self.conv2_1(x))
        x = self.pool(F.relu(self.conv2_2(x)))
        # third block : conv3_1=> conv3_2 => conv3_2 => conv3_2 => maxpool 
        x = F.relu(F.relu(self.conv3_1(x)))
        x = F.relu(F.relu(self.conv3_2(x)))
        x = F.relu(F.relu(self.conv3_2(x)))
        x = self.pool(F.relu(self.conv3_2(x)))
        # fourth block(fully connected layer) 
        x = x.view(-1, 128*3*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x) ,dim=1)

        return x



# Build CNN model with batchnormalization. apply it once each block

class CnnWithBN(nn.Module):
    def __init__(self):
        super(CnnWithBN,self).__init__()  
        self.pool = nn.MaxPool2d(2,2)
        self.conv1_1 = nn.Conv2d(1,32,kernel_size=3,stride=1 ,padding=1)
        self.conv1_2 = nn.Conv2d(32,32,kernel_size=3,stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64,128, kernel_size=3, stride=1 , padding=1)
        self.conv3_2 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*3*3,512)
        # self.bn_4 = nn.BatchNorm2d(512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128,10)

    def forward(self,x):
        # first block : conv1_1 => conv1_2 => maxpool
        x = F.relu(self.conv1_1(x))
        x = self.bn_1(x)
        x = self.pool(F.relu(self.conv1_2(x)))
        #second block : conv2_1 => conv2_2 => maxpool
        x = F.relu(self.conv2_1(x))
        x = self.bn_2(x)
        x = self.pool(F.relu(self.conv2_2(x)))
        # third block : conv3_1=> conv3_2 => conv3_2 => conv3_2 => maxpool 
        x = F.relu(F.relu(self.conv3_1(x)))
        x = self.bn_3(x)
        x = F.relu(F.relu(self.conv3_2(x)))
        x = F.relu(F.relu(self.conv3_2(x)))
        x = self.pool(F.relu(self.conv3_2(x)))
        # fourth block(fully connected layer) 
        x = x.view(-1, 128*3*3)
        x = F.relu(self.fc1(x))
        # x = self.bn_4(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x) ,dim=1)

        return x

# Build cnn model between layers
class CnnFBN(nn.Module):
    def __init__(self):
        super(CnnFBN,self).__init__()  
        self.pool = nn.MaxPool2d(2,2)
        self.conv1_1 = nn.Conv2d(1,32,kernel_size=3,stride=1 ,padding=1)
        self.conv1_2 = nn.Conv2d(32,32,kernel_size=3,stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64,128, kernel_size=3, stride=1 , padding=1)
        self.conv3_2 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*3*3,512)
        self.bn_4_1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn_4_2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,10)

    def forward(self,x):
        # first block : conv1_1 => conv1_2 => maxpool
        x = F.relu(self.conv1_1(x))
        x = self.bn_1(x)
        x = F.relu(self.conv1_2(x))
        x =self.bn_1(x) 
        x = self.pool(x)
        #second block : conv2_1 => conv2_2 => maxpool
        x = F.relu(self.conv2_1(x))
        x = self.bn_2(x)
        x = self.bn_2(F.relu(self.conv2_2(x)))
        x = self.pool(x)
        # third block : conv3_1=> conv3_2 => conv3_2 => conv3_2 => maxpool 
        x = F.relu(F.relu(self.conv3_1(x)))
        x = self.bn_3(x)
        x = F.relu(F.relu(self.conv3_2(x)))
        x = self.bn_3(x)
        x = F.relu(F.relu(self.conv3_2(x)))
        x = self.bn_3(x)
        x = F.relu(self.conv3_2(x))
        x = self.bn_3(x)
        x = self.pool(x)
        # fourth block(fully connected layer) 
        x = x.view(-1, 128*3*3)
        x = F.relu(self.fc1(x))
        x = self.bn_4_1(x)
        x = F.relu(self.fc2(x))
        x = self.bn_4_2(x)
        x = F.softmax(self.fc3(x) ,dim=1)
        return x


def train(model, device):
    print('Train start')
    start = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters() ,lr =lr)
    for epoch in range(n_epochs):
        
        running_loss = 0.
        for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device) 

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward() 
                optimizer.step()

                running_loss += loss.item()
                # if i % 100 == 99:
                #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100)  )
                running_loss = 0.0

    end = time.time()
    print('Finished Training, training tiem : %f ' % (end- start) )


def evaluate(model):
    print('Evaltuation start')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Acc : %.4f %%' % (100* correct / total))


# print(device)

cnn = Cnn()
cnn = cnn.to(device=device)
# summary(cnn, (1,28,28), batch_size)

cnnbn = CnnWithBN()
cnnbn = cnnbn.to(device=device)
# summary(cnnbn, (1,28,28), batch_size)

cnnfbn = CnnFBN()
cnnfbn = cnnfbn.to(device=device)
# summary(cnnfbn, (1,28,28), batch_size)


train(cnn)
train(cnnbn)
train(cnnfbn)

evaluate(cnn)
evaluate(cnnbn)
evaluate(cnnfbn)





