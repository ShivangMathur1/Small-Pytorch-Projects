import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Deep Q Network Class
class DQN(nn.Module):
    def __init__(self, lr, inputDims, fc1Dims, fc2Dims, nActions):
        super(DQN, self).__init__()
        
        # self.nActions = 1

        print("input_dims ", inputDims[0], " n_actions ",nActions) # +nActions
        self.fc1 = nn.Linear(*inputDims, fc1Dims)
        self.fc2 = nn.Linear(fc1Dims, fc2Dims)
        self.fc3 = nn.Linear(fc2Dims, nActions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        if T.cuda.is_available():
            print("Using CUDA")
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, observation):
        x = T.tensor(observation).float().to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Agent Class 
class Agent(object):
    def __init__(self, gamma, epsilon, lr, inputDims, batchSize, nActions, memSize=1000000, epsilonFinal=0.01, epsilonDecrease=0.996):
        self.actionSpace = [i for i in range(nActions)]
        self.DQN = DQN(lr, inputDims, 256, 256, nActions)
        
        # Hyper parameters
        self.memCounter = 0
        self.memSize = memSize
        self.nActions = nActions
        self.epsilon = epsilon
        self.epsilonDecrease = epsilonDecrease
        self.epsilonFinal = epsilonFinal
        self.batchSize = batchSize
        self.gamma = gamma

        # Memory storage for sampling inputs
        self.stateMemory = np.zeros((memSize, *inputDims))
        self.newStateMemory = np.zeros((memSize, *inputDims))        
        self.actionMemory = np.zeros((memSize, nActions), dtype=np.uint8)
        self.actionStateMemory = np.zeros((memSize, inputDims[0]+nActions))
        self.rewardMemory = np.zeros(memSize)
        self.qMemory = np.zeros(memSize)
        self.terminalMemory = np.zeros(memSize, dtype=np.uint8)

    # Storage of state, action, reward and termination values
    def store(self, state, action, reward, state_, terminal):
        index = self.memCounter % self.memSize
        self.stateMemory[index] = state
        actions = np.zeros(self.nActions)
        actions[action] = 1.0
        self.actionMemory[index] = actions
        self.rewardMemory[index] = reward
        self.terminalMemory[index] = terminal
        self.newStateMemory[index] = state_
        self.memCounter += 1

    # Exploration vs Exploitation
    def choose(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.actionSpace)
        else:
            actions = self.DQN(observation)
            action = T.argmax(actions).item()

        return action

    # Actual learning
    def learn(self):
        # If one batch is available
        if self.memCounter > self.batchSize:
            self.DQN.optimizer.zero_grad()
            
            # Generate batch
            maxMem = self.memCounter if self.memCounter < self.memSize else self.memSize
            batch = np.random.choice(maxMem, self.batchSize)
            stateBatch = T.Tensor(self.stateMemory[batch]).to(self.DQN.device)
            actionBatch = self.actionMemory[batch]
            rewardBatch = T.Tensor(self.rewardMemory[batch]).to(self.DQN.device)
            terminalBatch = T.Tensor(self.terminalMemory[batch]).to(self.DQN.device)
            newStateBatch = T.Tensor(self.newStateMemory[batch]).to(self.DQN.device)

            # Forward DQN for this and the next state
            # Update target Q values of the whole batch
            # Qtarget = reward + gamma*(q-value_for_best_action)
            # We get index 0 as the max function returns a tuple (value, index)
            batchIndex = np.arange(self.batchSize, dtype=np.int32)
            
            qEval = self.DQN.forward(stateBatch).to(self.DQN.device)[actionBatch]
            qNext = self.DQN.forward(newStateBatch).to(self.DQN.device)
            qTarget = rewardBatch + self.gamma*T.max(qNext, dim=1)[0]

            # Backpropagate the loss and Optimize
            loss = self.DQN.loss(qEval, qTarget).to(self.DQN.device)
            loss.backward();
            self.DQN.optimizer.step()
    
    def updateEpsilon(self):
        # Update exploration chance
        self.epsilon = self.epsilon * self.epsilonDecrease if self.epsilon > self.epsilonFinal else self.epsilonFinal