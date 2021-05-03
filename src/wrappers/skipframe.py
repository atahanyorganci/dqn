from gym import Wrapper


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        '''Return only `skip`-th frame'''
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        '''Repeat action, and sum reward'''
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
