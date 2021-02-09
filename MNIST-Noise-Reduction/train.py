import torch as T
import torch.optim as optim
from torch.nn import Module, Conv2d, MaxPool2d, CrossEntropyLoss 
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(Module):
    def __init__(self, lr):
        super(Autoencoder, self).__init__()

        # Layers
        self.encode1 = Conv2d(1, 32, 3)
        self.encode2 = Conv2d(32, 32, 3)
        self.pool = MaxPool2d(2, 2)
        self.decode1 = Conv2d(32, 32, 3)
        self.decode2 = Conv2d(32, 32, 3)
        self.output = Conv2d(32, 1, 3)


        self.loss = CrossEntropyLoss()
        self.optimizer = optim.Adadelta(self.parameters(), lr=lr)
    
    def forward(self, inputs):
        print(inputs.shape, inputs.type())
        x = self.pool(F.relu(self.encode1(inputs)))
        x = self.pool(F.relu(self.encode2(x)))
        x = F.relu(F.interpolate(self.decode1(x).squeeze(1), scale_factor=2, mode='nearest'))
        print(x.shape)
        x = F.relu(F.interpolate(self.decode2(x), scale_factor=2, mode='nearest'))
        x = F.sigmoid(self.output(x))

        return x


if __name__ == '__main__':

    # Load dataset
    trainset = datasets.MNIST('', train=True, download=True ,transform = transforms.Compose([transforms.ToTensor()]))
    train = T.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)


    # Hyperparameters
    epochs = 1#50
    lr = 0.001
    noiseFactor = 0.5

    net = Autoencoder(lr)
    net = net.double()

    losses = []

    # Training
    for epoch in range(epochs):
        for i, batch in enumerate(train, 0):
            # Adding noise to the MNIST image batch
            x, y = batch
            xNoisy = x + noiseFactor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
            xNoisy = np.clip(xNoisy, 0., 1.)

            net.optimizer.zero_grad()
            
            print(xNoisy.type())

            xPredicted = net(xNoisy.double())
            loss = net.loss(xPredicted, x.squeeze(1))
            loss.backward()
            net.optimizer.step()

            losses.append(loss.item())
        print("Epoch:", epoch, "Loss:", losses[epoch])

    T.save(net.state_dict(), './MNIST-denoising.pth')
    print('Model Saved')
        

