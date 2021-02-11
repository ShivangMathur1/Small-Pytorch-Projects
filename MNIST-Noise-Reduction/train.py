import torch as T
import torch.optim as optim
from torch.nn import Module, Conv2d, MaxPool2d, BCELoss 
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm

class Autoencoder(Module):
    def __init__(self, lr=0.002):
        super(Autoencoder, self).__init__()

        # Layers
        self.encode1 = Conv2d(1, 32, 3, padding=1)
        self.encode2 = Conv2d(32, 32, 3, padding=1)
        self.pool = MaxPool2d(2, 2)
        self.decode1 = Conv2d(32, 32, 3, padding=1)
        self.decode2 = Conv2d(32, 32, 3, padding=1)
        self.output = Conv2d(32, 1, 3, padding=1)


        self.loss = BCELoss()
        self.optimizer = optim.Adadelta(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)
    
    def forward(self, inputs):
        x = F.relu(self.encode1(inputs))
        x = self.pool(x)
        x = F.relu(self.encode2(x))
        x = self.pool(x)
        x = self.decode1(x)
        x = F.relu(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.decode2(x)
        x = F.relu(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = F.sigmoid(self.output(x))

        return x


if __name__ == '__main__':

    # Load dataset
    trainset = datasets.MNIST('', train=True, download=True ,transform = transforms.Compose([transforms.ToTensor()]))
    train = T.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)


    # Hyperparameters
    epochs = 50
    lr = 0.003
    noiseFactor = 0.5

    net = Autoencoder(lr)
    net = net.double()

    

    # Training
    for epoch in range(epochs):
        losses = []
        for i, batch in enumerate(tqdm(train, 0)):
            # Adding noise to the MNIST image batch
            x, _ = batch
            xNoisy = x + noiseFactor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
            xNoisy = np.clip(xNoisy, 0., 1.).to(net.device)
            x = x.to(net.device)

            net.optimizer.zero_grad()

            xPredicted = net(xNoisy)
            loss = net.loss(xPredicted.double(), x.double()).to(net.device)
            loss.backward()
            net.optimizer.step()

            losses.append(loss.item())
        print("Epoch:", epoch, "Loss:", np.mean(losses))

    T.save(net.state_dict(), './MNIST-denoising.pth')
    print('Model Saved')
        

