from gym import Wrapper
from gym.spaces import Discrete


class AutoFire(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._lives = 5
        self.action_space = Discrete(3)
        self._fire = False

    def step(self, action):
        if self._fire:
            obs, reward, done, info = self.env.step(1)
            self._fire = False
        elif action > 0:
            obs, reward, done, info = self.env.step(action + 1)
        else:
            obs, reward, done, info = self.env.step(action + 1)

        if info['ale.lives'] < self._lives:
            self._lives = info['ale.lives']
            self._fire = True
        return obs, reward, done, info
