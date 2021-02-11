import torch as T
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from train import Autoencoder

net = Autoencoder()
net.load_state_dict(T.load('./MNIST-denoising.pth'))
net = net.double()

testset = datasets.MNIST('', train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = T.utils.data.DataLoader(testset, batch_size=256, shuffle=True)

for i, batch in enumerate(test, 0):
    x, _ = batch
    xNoisy = x + 0.2 * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    xNoisy = np.clip(xNoisy, 0., 1.).to(net.device)
    x = x.to(net.device)

    X = net(xNoisy).cpu().detach().numpy()
    break
plt.imshow(X[0].reshape(28, 28))
plt.show()
'''
losses = []
for i, batch in enumerate(test, 0):
    x, _ = batch
    xNoisy = x + 0.2 * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    xNoisy = np.clip(xNoisy, 0., 1.).to(net.device)
    x = x.to(net.device)
    
    xPredicted = net(xNoisy)
    loss = net.loss(xPredicted.double(), x.double()).to(net.device)
    losses.append(loss.item())
    
print(losses)
'''