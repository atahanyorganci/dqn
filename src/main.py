import gym
import time
import click
from torch._C import device
from wrappers import make_env
import torch
from dqn import DQN
from train import train_agent
from gym.utils.play import play
import numpy as np
from gym.envs.classic_control import rendering


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        return rgb_array

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--env_name', default='BreakoutDeterministic-v4',
              help='Name of the gym environment.')
@click.option('--it_count', default=0,
              help='Bound number of ')
@click.option('--delay', default=0.1,
              help='Delay duration to set the game speed.')
@click.option('--zoom', default=3,
              help='Adjusts the size of the game window.')
def random_agent(env_name, it_count, delay, zoom):
    viewer = rendering.SimpleImageViewer()
    env = gym.make(env_name)
    observation = env.reset()

    iteration = 0
    while iteration <= it_count:
        rgb = env.render('rgb_array')
        upscaled = repeat_upsample(rgb, zoom, zoom)
        viewer.imshow(upscaled)

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
        if it_count > 0:
            iteration += 1
        time.sleep(delay)
    env.close()


@cli.command()
@click.option('--env_name', default='BreakoutDeterministic-v4',
              help='Name of the gym environment.')
@click.option('--fps', default=30,
              help='FPS to set the game speed.')
@click.option('--zoom', default=3,
              help='Adjusts the size of the game window.')
def keyboard_agent(env_name, fps, zoom):

    play(gym.make(env_name), fps=fps, zoom=zoom)


@cli.command()
@click.argument('model')
@click.option('--env_name', default='BreakoutDeterministic-v4',
              help='Name of the gym environment.')
@click.option('--delay', default=0.1,
              help='Delay duration to set the game speed.')
@click.option('--zoom', default=3,
              help='Adjusts the size of the game window.')
@click.option('-c', '--continuous', is_flag=True, default=False,
              help='Continue playing after death.')
def observe(model, env_name, delay, zoom, continuous):
    viewer = rendering.SimpleImageViewer()
    env = make_env(env_name)
    ACTION_COUNT = env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dqn = torch.load(model, map_location=device).cpu()

    while True:
        obs = env.reset()
        beginning_episode = True
        total_rew = 0
        while True:
            if beginning_episode:
                action = 1
                beginning_episode = False
            else:
                action = dqn.act(torch.Tensor(obs.__array__()).unsqueeze(0))

            obs, rew, done, info = env.step(action)
            total_rew += rew

            rgb = env.render('rgb_array')
            upscaled = repeat_upsample(rgb, zoom, zoom)
            viewer.imshow(upscaled)
            time.sleep(delay)

            if done:
                obs = env.reset()
                beginning_episode = True
                break
        if continuous:
            print(f'Dead! Reward: {total_rew}.')
        else:
            ans = input(f'Dead! Reward: {total_rew}. Continue? [y/n] ').lower()
            if ans == 'n':
                break
    env.close()


@cli.command()
@click.option('--env', help='Environment name.', default='BreakoutDeterministic-v4')
@click.option('--model', help='If provided model will continue to be trained')
@click.option('--epsilon', is_flag=True, help='Controls whether epsilon is reset')
def train(env, model, epsilon):
    train_agent(env, model_path=model, epsilon=True)


if __name__ == '__main__':
    # Keyboard Global Attributes
    human_agent_action = 0
    human_wants_restart = False
    human_sets_pause = False
    cli()
