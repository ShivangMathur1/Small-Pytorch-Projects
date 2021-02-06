import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

# Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #layers: 
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':

    # Load dataset to train on
    train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

    # Hyperparameters
    epochs = 3
    lr = 0.001

    # Declaring model, optimizer
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Learning
    for epoch in range(epochs):
        for data in trainset:
            # Data is a batch of (feature, label)
            x, y = data
            net.zero_grad()
            
            # Forward, backward,  optimiztion
            output = net(x.view(-1, 28*28))
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()

    # Save model
    torch.save(net.state_dict(), "./mnist_net.pth")
    print("Model Saved")