import torch
from wrappers import make_env
from agent import Agent
from pathlib import Path
import itertools
from util.logger import MetricLogger


hyper = {
    'REPLAY_MEMORY_SIZE': 12000,
    'BATCH_SIZE': 32,
    'GAMMA': 0.9,
    'LEARNING_RATE': 1e-3,
    'MOMENTUM': 0.95,
    'EPSILON_START': 1,
    'EPSILON_END': 0.2,
    'EPSILON_DECAY': 0.999995,
    'SAVE_EVERY': 1e4,
    'BURNIN': 1e4,
    'LEARN_EVERY': 3,
    'SYNC_EVERY': 100,
    'SAVE_DIR': 'models',
}


def train_agent(env_name, *, model_path=None, epsilon=False):
    global hyper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'PyTorch using {device} device')

    env = make_env(env_name)
    ENV_SHAPE = env.observation_space.shape
    ACTION_COUNT = env.action_space.n

    agent = Agent(ENV_SHAPE, ACTION_COUNT, device,
                  path=model_path, epsilon=epsilon, **hyper)

    save_dir = Path(hyper['SAVE_DIR'])
    episodes = 4000

    for i in itertools.count():
        agent.reset_epsilon()
        logger = MetricLogger(save_dir)
        print(f'Running Epoch {i+1}')
        for e in range(episodes):
            state = env.reset()
            while True:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, next_state, reward, done)
                q, loss = agent.learn()
                logger.log_step(reward, loss, q)
                state = next_state
                if done:
                    break
            logger.log_episode()
            if e % 20 == 0:
                logger.record(episode=e,
                              epsilon=agent.exploration_rate,
                              step=agent.step)
