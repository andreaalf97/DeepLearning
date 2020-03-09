
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Using ``torchvision``, it’s extremely easy to load CIFAR10.

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.5 is the mean and std
)

trainSet = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=80,
    shuffle=True,
    num_workers=2
)

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
)

# DEFINING A CONVOLUTIONAL NEURAL NETWORK

# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

class Net(nn.Module):

    def __init__(self):
        
        super(Net, self).__init__()
        
        self.do1 = nn.Dropout2d(p=0.1)
        self.do2 = nn.Dropout2d(p=0.25)
        self.do3 = nn.Dropout2d(p=0.5) 
        self.do4 = nn.Dropout(p=0.5)
        
        self.conv1 = nn.Conv2d(3, 96, 5,padding=2)

        self.conv2 = nn.Conv2d(96, 128, 5,padding=2)
        
        self.conv3 = nn.Conv2d(128,256, 5, padding=2)

        self.fc1 = nn.Linear(2304, 2048)

        self.fc2 = nn.Linear(2048, 10)
        
        self.pool = nn.MaxPool2d(3, 2)

    def forward(self, x):
        
        x = self.do1(x)
        x = self.pool(F.relu(self.do2(self.conv1(x))))
        x = self.pool(F.relu(self.do2(self.conv2(x))))
        x = self.pool(F.relu(self.do3(self.conv3(x))))
        x = x.view(-1, 2304)
        x = self.do4(x)
        x = F.relu(self.fc1(x))
        x = self.do4(x)
        x = F.relu(self.fc2(x))
        return x



use_gpu = torch.cuda.is_available()

net = Net()

if use_gpu:
    net = net.cuda()

# Define a Loss function and optimizer
# Let's use a Classification Cross-Entropy loss and SGD with momentum.
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9
)

#optimizer =optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# TRAIN THE NETWORK

# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(trainLoader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
       
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
       
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')



