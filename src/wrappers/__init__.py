import gym
from gym.wrappers import FrameStack
from wrappers.skipframe import SkipFrame
from wrappers.grayscale import GrayScaleObservation
from wrappers.resize import ResizeObservation
from wrappers.autofire import AutoFire


def make_env(env_name):
    env = gym.make(env_name)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = AutoFire(env)
    return env
