import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from ddpgarm import Actor, Critic, OUNoise

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DDPG():

    '''Define DDPG agent class'''

    def __init__(self, n_state, n_action, num_agents, seed):

        '''Initialize the agent

        Params:
            n_state         : dimension of state space
            n_action        : dimension of action
            lr              : learning rate of the network
            tau             : constant soft update
            seed            : random seed number
        '''

        # assign vars from input
        self.n_state = n_state
        self.n_action = n_action
        self.lr = 0.0001
        self.tau = 0.001
        self.seed = random.seed(seed)

        # assign the rest of the vars from constants
        self.batch_size = 256
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 1e-6
        self.eps_min = 0.01
        self.targetUpdateNet = 20

        # set actor and critic network
        self.act_local = Actor(self.n_state, self.n_action, seed).to(device)
        self.act_target = Actor(self.n_state, self.n_action, seed).to(device)
        self.act_opt = optim.Adam(self.act_local.parameters(), lr=self.lr)

        self.crit_local = Critic(self.n_state, self.n_action, seed).to(device)
        self.crit_target = Critic(self.n_state, self.n_action, seed).to(device)
        self.crit_opt = optim.Adam(self.crit_local.parameters(), lr=self.lr*3,
                                   weight_decay=0)

        # Setup noise process
        self.noise = OUNoise(n_action, seed)

        # Setup tuple for replay buffer
        self.experience_replay = deque(maxlen=int(1e6))
        labels = ['state', 'action', 'reward', 'state_', 'done']
        self.experience = namedtuple('Experience', field_names=labels)
        self.t_step = 0     # counter for update

    def step(self, state, action, reward, state_, done):

        '''Learn for every step fulfilled by targetUpdateNet after appending
           memory experience

        Params:
            state (array_like)  : current state
            action (array_like) : action taken
            reward (array_like) : reward for the specific action taken
            state_ (array_like) : new state after action is executed
            done (array_like)   : status of episode (finished or not)
        '''

        # Append the experience
        exp = self.experience(state, action, reward, state_, done)
        self.experience_replay.append(exp)

    def check_learn(self):

        '''Check the learning condition while updating the t_step
           for target update network
        '''
        self.t_step += 1
        if (len(self.experience_replay) > self.batch_size):
            if (self.t_step % self.targetUpdateNet == 0):
                experience_batch = self.sample_replay()
                self.learn(experience_batch)

    def select_action(self, states):

        '''Returns actions based on the current state of the environment
           using the current policy

        Params:
            state (array_like)  : current state
            eps (float)         : epsilon for epsilon-greedy action

        Return:
            action selected
        '''

        states = torch.from_numpy(states).float().to(device)
        self.act_local.eval()
        with torch.no_grad():
            actions = self.act_local(states).cpu().data.numpy()
        self.act_local.train()
        actions += self.epsilon*self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        '''Reset the noise after updating epsilon'''
        self.epsilon = max(self.epsilon - self.eps_decay, self.eps_min)
        self.noise.reset()

    def sample_replay(self):

        '''Take random sample of experience from the batches available within
           the replay buffer

        Return:
            tuple of states, actions, rewards, next states and dones
        '''

        experiences = random.sample(self.experience_replay, self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        states_ = torch.from_numpy(np.vstack([e.state_ for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, states_, dones)

    def learn(self, experience_batch):

        '''Setup the qnet to learn from qnet_local and use the qnet_target as
           the Q_target to learn from

        Params:
            experiences_batch (array_like)  : a batch of memory replay tuples
        '''

        states, actions, rewards, states_, dones = experience_batch

        actions_ = self.act_target(states_)
        Q_targets_ = self.crit_target(states_, actions_)

        Q_targets = rewards + (self.gamma * Q_targets_ * (1 - dones))
        Q_expect = self.crit_local(states, actions)

        crit_loss = F.mse_loss(Q_targets, Q_expect)
        self.crit_opt.zero_grad()
        crit_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.crit_local.parameters(), 1)
        self.crit_opt.step()

        # update actor
        actions_pred = self.act_local(states)
        actor_loss = -self.crit_local(states, actions_pred).mean()
        self.act_opt.zero_grad()
        actor_loss.backward()
        self.act_opt.step()

        # soft update
        self.soft_update(self.act_local, self.act_target)
        self.soft_update(self.crit_local, self.crit_target)

    def soft_update(self, local, target):

        '''Carry out the soft update of the network using the constant tau

        Params:
            local (PyTorch model)   : qnet_local model
            target (PyTorch model)  : qnet_target model
        '''

        for local_param, target_param in zip(local.parameters(),
                                             target.parameters()):
            target_param.data.copy_(self.tau * local_param.data + \
                                    (1.0-self.tau) * target_param.data)
