import gym
import time
import click
from wrappers import make_env
import torch
from dqn import DQN
from train import train_agent


@click.group()
def cli():
    pass


@cli.command()
@click.option('--env_name', default='Breakout-v0',
              help='Name of the gym environment.')
@click.option('--it_count', default=0,
              help='Bound number of ')
@click.option('--delay', default=0.0,
              help='Delay duration to set the game speed.')
def random_agent(env_name, it_count, delay):
    env = gym.make(env_name)
    observation = env.reset()

    iteration = 0
    while iteration <= it_count:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
        if it_count > 0:
            iteration += 1
        time.sleep(delay)
    env.close()


@cli.command()
@click.option('--env_name', default='Breakout-v0',
              help='Name of the gym environment.')
@click.option('--delay', default=0.1,
              help='Delay duration to set the game speed.')
def keyboard_agent(env_name, delay):
    env = gym.make(env_name)
    ACTIONS = env.action_space.n
    SKIP_CONTROL = 0

    def key_press(key, mod):
        global human_agent_action, \
            human_wants_restart, \
            human_sets_pause
        if key == 0xff0d:
            human_wants_restart = True
        if key == 32:
            human_sets_pause = not human_sets_pause
        a = int(key - ord('0'))
        if a <= 0 or a >= ACTIONS:
            return
        human_agent_action = a

    def key_release(key, mod):
        global human_agent_action
        a = int(key - ord('0'))
        if a <= 0 or a >= ACTIONS:
            return
        if human_agent_action == a:
            human_agent_action = 0

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    def rollout(env):
        global human_agent_action, \
            human_wants_restart, \
            human_sets_pause
        human_wants_restart = False
        obser = env.reset()
        skip = 0
        total_reward = 0
        total_timesteps = 0
        while True:
            if not skip:
                a = human_agent_action
                total_timesteps += 1
                skip = SKIP_CONTROL
            else:
                skip -= 1

            obser, r, done, info = env.step(a)
            if r != 0:
                print('reward %0.3f' % r)
            total_reward += r
            window_still_open = env.render()
            if not window_still_open:
                return False
            if done or human_wants_restart:
                break
            while human_sets_pause:
                env.render()
                time.sleep(delay)
            time.sleep(delay)
        print((f'timesteps {total_timesteps} '
               f'reward {total_reward:0.2f}'))

    print(f'ACTIONS={ACTIONS}')
    print('Press keys 1 2 3 ... to take actions 1 2 3 ...')
    print('No keys pressed is taking action 0')

    while True:
        window_still_open = rollout(env)
        if window_still_open == False:
            break


@cli.command()
@click.argument('model')
@click.option('--env_name', default='Breakout-v0',
              help='Name of the gym environment.')
@click.option('--delay', default=0.02,
              help='Delay duration to set the game speed.')
@click.option('--record',
              help='If supplied actions of the agent will be recorded into given file')
def observe(model, env_name, delay):
    env = make_env(env_name)
    ACTION_COUNT = env.action_space.n
    dqn = torch.load(model).cpu()

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
            env.render()
            time.sleep(delay)

            if done:
                obs = env.reset()
                beginning_episode = True
                break
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
