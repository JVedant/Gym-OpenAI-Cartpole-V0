#import libraries
import numpy as np
import random
from tensorflow.keras import Sequential
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, linear

# Defining a Class for Deep Q learning
class DQN:

    # initialize using the action and state size
    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .996
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    # building and compiling a neural network of n (here n=3) layers 
    def build_model(self):
        
        model = Sequential([
            Dense(150, input_dim=self.state_space, activation=relu),
            Dense(120, activation=relu),
            Dense(self.action_space, activation=linear)])

        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model
    
    #saving a model
    def save_model(self):
        self.model.save('Cartpole-model')

    # creating checkpoints (adding in memory)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # predicting the next action from current state
    def act(self, state):

        # checking for exploration or exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


    def replay(self):

        # if batchsize is greater is less than memory size then exit
        if len(self.memory) < self.batch_size:
            return

        
        # fetching data from minibatch and assigning to states, actions, rewards, next_states and dones
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Remove single-dimensional entries from states and next_states.
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        # setting target values
        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        # training the model
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        # reducing the epsilon to reduce exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay