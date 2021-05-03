from dqn import DQN
import torch
from util.epsilon import Epsilon
from util.replay_mem import ReplayMemory
from pathlib import Path
import copy
import numpy as np


class Agent:

    def __init__(self, state_shape, action_count, device, *, path=None, epsilon=True, **hyper):
        super().__init__()

        self.device = device
        self.action_count = action_count
        self.state_shape = state_shape

        # Construct target, and online net
        if path is None:
            self._target = DQN(state_shape, action_count).to(device)
        else:
            self._target = torch.load(path).to(device)

        if epsilon:
            self.epsilon = Epsilon(start=hyper['EPSILON_START'],
                                   end=hyper['EPSILON_END'],
                                   decay=hyper['EPSILON_DECAY'])
        else:
            self.epsilon = Epsilon(
                hyper['EPSILON_END'], hyper['EPSILON_END'], 1)

        self._online = copy.deepcopy(self._target).to(device)
        for p in self._online.parameters():
            p.requires_grad = True
        for p in self._target.parameters():
            p.requires_grad = False

        self.step = 0
        self.memory = ReplayMemory(hyper['REPLAY_MEMORY_SIZE'])
        self.batch_size = hyper['BATCH_SIZE']
        self.gamma = hyper['GAMMA']
        self.optimizer = torch.optim.Adam(
            self._online.parameters(), lr=hyper['LEARNING_RATE'])
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.save_dir = hyper['SAVE_DIR']
        self.sync_every = hyper['SYNC_EVERY']
        self.save_every = hyper['SAVE_EVERY']
        self.learn_every = hyper['LEARN_EVERY']
        self.burnin = hyper['BURNIN']

    def save(self):
        filename = f"dqn{int(self.step // self.save_dir)}.chkpt"
        save_path = Path.joinpath(Path.cwd(), self.save_dir, filename)
        torch.save(self._target, save_path)
        print(f'DQN saved to {save_path} at step {self.step}')

    def reset_epsilon(self):
        self.epsilon.reset()

    def learn(self):
        if self.step % self.sync_every == 0:
            self.sync_Q_target()
        if self.step % self.save_every == 0:
            self.save()
        if self.step < self.burnin:
            return None, None
        if self.step % self.learn_every != 0:
            return None, None

        state, action, next_state, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self._target.load_state_dict(self._online.state_dict())

    def td_estimate(self, state, action):
        action = action.view(-1)
        current_Q = self.online(state)
        return current_Q[np.arange(0, self.batch_size), action]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.online(next_state)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.target(next_state)[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def remember(self, state, action, next_state, reward, done):
        state = torch.tensor(state.__array__()).to(self.device)
        action = torch.tensor([action]).to(self.device)
        next_state = torch.tensor(next_state.__array__()).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        done = torch.tensor([done]).to(self.device)

        self.memory.push(state, action, next_state, reward, done)

    def recall(self):
        batch = self.memory.sample(self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def target(self, X):
        return self._target(X)

    def online(self, X):
        return self._online(X)

    def act(self, state):
        '''
        Given a state, choose an epsilon-greedy action and update value of step.
        '''
        self.step += 1
        if np.random.rand() < next(self.epsilon):
            return self.explore()
        else:
            return self.exploit(state)

    def explore(self):
        '''Agent explores by taking a random action, and remembering it'''
        return np.random.randint(self.action_count)

    def exploit(self, state):
        '''Agent exploits by choosing the action that will maximize Q-value'''
        state_array = state.__array__()
        state_tensor = torch.tensor(state_array)
        state_tensor = state_tensor.to(self.device).view(-1, *self.state_shape)
        return self._target.act(state_tensor).item()

    @property
    def exploration_rate(self):
        '''Exploration rate is controlled by decaying exponential'''
        return self.epsilon.value
