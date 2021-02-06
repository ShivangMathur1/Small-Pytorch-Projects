import torch
from torchvision import transforms, datasets
from train import Net
import matplotlib.pyplot as plt

# Create dataset to test on
test =  datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

# Load Model
net = Net()
net.load_state_dict(torch.load('./mnist_net.pth'))


# Individual image testing
# '''
img = 0
x, y = 0, 0
for data in testset:
    x, y = data
    break

plt.imshow(x[img].view(28,28))
plt.show()

print(torch.argmax(net(x[img].view(-1, 28*28))[0]).item())

'''

correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        x, y = data
        output = net(x.view(-1, 28*28))
        for index, i in enumerate(output):
            if(torch.argmax(i) == y[index]):
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 4) * 100, "%")
'''