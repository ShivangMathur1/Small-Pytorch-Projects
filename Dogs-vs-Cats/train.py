import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2
import numpy as np
from tqdm import tqdm

# Neural network for cats vs dogs
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers: Images go from 1 to 32 to 64 to 128 channels. Kernel size 5x5
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        # To determine the flattened shape of convolutional layer output tensor by passing
        # dummy data through the network
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self.toLinear = None
        self.convs(x)

        # Linear layers: from flattened convolutional layer to a 512 neuron dense layer to 
        # a one-hot output layer with 2 outputs
        self.fc1 = nn.Linear(self.toLinear, 512)
        self.fc2 = nn.Linear(512, 2)

    # Does a feed forward of convolutions with maxpool functions and also finds flattened shape
    # of output tensor if not already determined
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        
        if self.toLinear is None:
            self.toLinear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    # Simple forward funtion
    def forward(self, x):
        x = self.convs(x)               # Convolutional layers
        x = x.view(-1, self.toLinear)   # FLattening
        x = F.relu(self.fc1(x))         # Linear layer
        x = self.fc2(x)                 # Output layer
        
        return F.softmax(x, dim=1)
if __name__ == '__main__':
    # Preprocessing (optional if Training_Data.npy is already genereated)
    # ----------------------------------------------------------------------------------------------
    # Flag to decide whether to generate dataset from images or not
    build_data = False
    # Class to generate training data from the dataset
    class DvsC():
        SIZE = 50
        CATS = 'PetImages/Cat'
        DOGS = 'PetImages/Dog'
        LABELS = {CATS: 0, DOGS: 1}
        trainingData = []
        catCount = 0
        dogCount = 0

        # function that does all the preprocessing to make the inputs(images) into trainable data for
        # the neural network. Includes resizing all images to same size, grayscaling, and grouping 
        # images with the output value in one-hot format
        def makeTrainingData(self):
            for label in self.LABELS:
                print(label)
                for f in tqdm(os.listdir(label)):   # For loading screen
                    try:
                        # Grayscaling, colour not needed for cats vs dogs, resizing to 50x50 size and
                        # grouping images and output (in one-hot) together
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.SIZE, self.SIZE))
                        self.trainingData.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                        if label == self.CATS:
                            self.catCount += 1
                        elif label == self.DOGS:
                            self.dogCount += 1
                    # For ignoring corrupt images and non-image files
                    except Exception as e:
                        pass
            
            # Shuffle training data for efficient learning
            np.random.shuffle(self.trainingData)
            # Saving dataset to disk
            np.save('Training_Data.npy', self.trainingData)
            print('Cats: ', self.catCount, 'Dogs: ', self.dogCount)

    # Build data only if needed
    if build_data:
        pre = DvsC()
        pre.makeTrainingData()
    # -------------------------------------------------------------------------------------------

    # Real Program starts
    
    # Set device to GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
    else:
        device = torch.device("cpu:0")
        print("Using CPU")
    
    # Load training data
    trainingData = np.load("Training_Data.npy", allow_pickle=True)

    # Hyperparameters
    batchSize = 100
    lr = 0.001
    epochs = 3

    # Initializations
    net = Net().to(device);
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lossFunction = nn.MSELoss()

    #Further processing training data
    x = torch.Tensor([i[0] for i in trainingData]).view(-1, 50, 50)
    x = x/255.0
    y = torch.Tensor([i[1] for i in trainingData])

    # Divide dataset into training and testing sets
    testFraction = 0.1
    size = int(len(x)*testFraction)
    print(size)

    trainX = x[:-size]
    trainY = y[:-size]

    # Training starts
    loss = []
    for epoch in range(epochs):
        for i in tqdm(range(0, len(trainX), batchSize)):
            # Generate batches
            batchX = trainX[i:i+batchSize].view(-1, 1, 50, 50).to(device)
            batchY = trainY[i:i+batchSize].to(device)

            # Initialize, forward, loss, backward and optimize
            net.zero_grad()
            outputs = net(batchX)
            loss = lossFunction(outputs, batchY)
            loss.backward()
            optimizer.step()

    print(loss)
    print("Model Saved")
    torch.save(net.state_dict(), "./DogsVsCats_net.pth")