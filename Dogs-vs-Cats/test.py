import torch
from train import Net
import numpy as np
from tqdm import tqdm

correct = 0
total = 0

# Load training data
trainingData = np.load("Training_Data.npy", allow_pickle=True)

#Further processing training data
x = torch.Tensor([i[0] for i in trainingData]).view(-1, 50, 50)
x = x/255.0
y = torch.Tensor([i[1] for i in trainingData])

# Divide dataset into training and testing sets
testFraction = 0.1
size = int(len(x)*testFraction)
print(size)

testX = x[-size:]
testY = y[-size:]

# Load model
net = Net()
net.load_state_dict(torch.load("./DogsVsCats_net.pth"))

# Start testing
with torch.no_grad():
    for i in tqdm(range(len(testX))):
        realClass = torch.argmax(testY[i])
        predictedClass = torch.argmax(net(testX[i].view(-1, 1, 50, 50)))

        if predictedClass == realClass:
            correct += 1
        total += 1

print("Accuracy: ", round(correct/total, 4))
