import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__=='__main__':
    # Convert images to tensor, then normalize them
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

    # Load a training and a testing dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # List of classes to work on
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Neural net class with 3 convolutional layers and 3 linear layers
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # Three channel input (RGB) and 6 convolutional channel output with a 5x5 kernel size (the size of the window matrix
            # which is multiplied to perform convolutions)
            self.conv1 = nn.Conv2d(3, 6, 5)

            # Pooling function which generalises the finer detais of the convolutions and allows for recognition of tilted and 
            # differently scaled images. It passes a 2x2 kernel with a stride(pixel per step movement) of 2x2. Maxpool keeps only the
            # maximum value from all kernels. This also results in decrease in size of each channel(down-sampling: it's a good thing).
            self.pool = nn.MaxPool2d(2, 2)

            # 6 input channels and 16 output channels with kernel size 5x5
            self.conv2 = nn.Conv2d(6, 16, 5)

            # Flattening layer: converts channel input into linear output after the tensors in the last layer(conv2) are resized in the
            self.fc1 = nn.Linear(16 * 5 * 5, 120)

            #two linear layers for complexity
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            
            # Resizing of input tensors for flattening of the 2d channels into 1d outputs
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    #Neural net object
    net = Net()

    # Loss and optimizasion functions: Cross Entrpy loss and Schocastic Gradient Descent 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)          # Momentum decides effect of past steps on the current one

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Seperate Data = (inputs, labels)
            inputs, labels = data

            # Initialize optimizer gradients to zero
            optimizer.zero_grad()
            
            # Get predicted outputs using forward (called automatically when net is given inputs)
            outputs = net(inputs)
            # Calculate loss from outputs and labels
            loss = criterion(outputs, labels)
            # Calculate gradients from loss dx/d(loss)
            loss.backward()
            # Optimize parameters using gradients from loss
            optimizer.step()

            # Prints stats every 2000 iterations
            running_loss += loss.item()
            if i % 2000 == 1999:
                print("Epoch %d, Iteration %5d, Loss: %.3f" %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print("Training Complete. Model saved.")
