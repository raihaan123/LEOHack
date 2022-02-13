from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import numpy as np


# Reference - https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html



## State vector - [x, y, vx, vy, theta, omega]
## Action vector - [F_x, F_y, tau]

## Reward function - 0.1 * Time elapsed + 0.9 * Fuel Consumption


# RL Environment Wrapper
class SatelliteSystem(Environment):

    def __init__(self):
        super().__init__()

    def states(self):
        # Continuous State space has 6 dimensions - [x, y, theta, vx, vy, omega]
        return dict(type='float', shape=(6,))

    def actions(self):
        # Continuous Action space has 3 dimensions - [F_x, F_y, tau]
        return dict(type='float', shape=3)


    # Optional additional steps to close environment
    def close(self):
        super().close()

    # From the environment
    def reset(self, init_state=np.random.random(size=(6,))):
        state = init_state
        return state


    def execute(self, state, actions):
        next_state = np.random.random(size=(6,))
        terminal = False  # Always False if no "natural" terminal state
        reward = np.random.random()
        return next_state, terminal, reward



class RL_Model:
    def __init__(self, env):
        agent = Agent.create(
            agent='ppo', environment=env, batch_size=10, learning_rate=1e-3
        )

        runner = Runner(
            agent=agent,
            environment=env
        )
    
    def start(self):
        runner.run(num_episodes=200)
        runner.run(num_episodes=100, evaluation=True)

        runner.close()


