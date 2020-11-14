import gym
from Lunar_lander_agent import Agent
import torch as T
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ =='__main__':
    env = gym.make('LunarLander-v2')
    brain = Agent(gamma=1.98, epsilon=1.0, batchSize=32, nActions=4, inputDims=[8], lr=0.01)

    scores = []
    epsHistory = []
    episodes = 500

    for i in range(episodes):
        
        score = 0
        state = env.reset()
        done = False

        while not done:
            action = brain.choose(state)
            stateNew, reward, done, info = env.step(action)
            
            if i % 10 == 0:
                env.render()

            score += reward
            brain.store(state, action, reward, stateNew, done)
            brain.learn()
            state = stateNew
        
        scores.append(score)
        epsHistory.append(brain.epsilon)

        brain.updateEpsilon()
        if i % 10 == 0 and i > 0:
            avgScore = np.mean(scores[-100:])
            print('Episode: ', i, '\tScore: ', score, '\tAverage Score: %.3f' % avgScore, 'Epsilon %.3f' % brain.epsilon)
        else:
            print('Episode: ', i, 'Score: ', score)
        time.sleep(1)

    T.save(brain.DQN.state_dict(), 'lunar-model.pt')

    x = [i + 1 for i in range(episodes)]
    plt.plot(x, scores)
    plt.show()