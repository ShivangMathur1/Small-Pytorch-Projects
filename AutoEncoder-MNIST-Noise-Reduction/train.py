import torch as T
from torch.optim import optim
from torch.nn import Linear, Module, Conv2d
import torch.nn.functional as F
from torchvision import transforms, datasets

class Autoencoder(Module):
    def __init__(self, lr):
        super(Autoencoder, self).__init__()

        # Layers
        self.encode1 = Conv2d(1, 32, 3)
        self.encode2 = Conv2d(32, 32, 3)
        self.pool = F.max_pool2d(2, 2)
        self.decode1 = Conv2d(32, 32, 3)
        self.decode2 = Conv2d(32, 32, 3)
        self.upsample = F.upsample(2, 2)
        self.output = Conv2d(32, 1, 3)


        self.loss = F.mse_loss()
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
    
    def forward(self, inputs):
        x = self.pool(self.encode1(inputs))
        x = self.pool(self.encode2(x))
        x = self.upsample(self.decode1(x))
        x = self.upsample(self.decode2(x))
        x = self.output(x )


if __name__ == '__main__':

    # Load dataset
    train = datasets.MNIST('', train=True, download=True ,transform = transforms.Compose([transforms.ToTensor()]))
    trainset = T.utils.data.DataLoader(train, batch_size=512)


    # Hyperparameters
    epochs = 10
    lr = 0.001
