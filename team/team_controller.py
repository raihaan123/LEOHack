# =================================================================
# Controller choice --> 'PID', 'RL'
controller_choice   = 'RL'
training            = True
agent_path          = 'agent_model'
# =================================================================

from sat_controller import SatControllerInterface, sat_msgs
from datetime import datetime, timedelta
import numpy as np

# Additional imports - PID solution
from pid import PID
from plotter import state_space
import sys

# Additional imports - RL solution
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner


# RL Environment Wrapper - interacts with the simulation api
class SatelliteSystem(Environment):
    def __init__(self):
        super().__init__()
        self.state = np.zeros(6)
        self.action = np.zeros(3)
        self.terminal = False
        self.reward = 0

        # Create a thrust command message
        self.control = sat_msgs.ControlMessage()

    # State space definition
    def states(self):
        return dict(type='float', shape=(6,))

    # Action space definition
    def actions(self):
        return dict(type='float', shape=3)

    def close(self):
        super().close()

    def reset(self, init_state):
        self.state = init_state
        self.time = 0
        self.fuel = 0
        self.reward = 0
        return self.state


    # Defining the reward function
    def rewarder(self, errors, fuel):

        # Reward function - norm of the error
        reward = np.linalg.norm(errors)
        return reward


class RL_Model:
    def __init__(self, env):
        self.env = env

        agent = self.agent = Agent.create(
            agent='ppo', environment=env, batch_size=10, learning_rate=1e-3
        )

        try:
            agent.restore_model(directory=agent_path)
        except:
            print('No previous model found!')

        # Probably not gonna use standard runner
        self.runner = Runner(
            agent=agent,
            environment=env
        )
    
    def start(self):
        runner.run(num_episodes=200)
        runner.run(num_episodes=100, evaluation=True)
        runner.close()



class TeamController(SatControllerInterface):
    """ Team control code """

    def team_init(self):
        """ Runs any team based initialization """

        self.counter = 0

        # Update team info
        team_info = sat_msgs.TeamInfo()
        team_info.teamName = "Group 16"
        team_info.teamID = 1111

        # Mission performance histories - N x 6 array of errors and N x 3 array of actions
        self.errors         = np.zeros((0,6))
        self.actions        = np.zeros((0,3))


        # ====================== RL Controller ============================
        # =================================================================

        # Initialize RL environment wrapper, agent and model
        self.RL_satellite = Environment.create(
            environment=SatelliteSystem,
            max_episode_timesteps=500
        )

        self.RL_model = RL_Model(self.RL_satellite)
        self.total_reward = 0


        # ====================== PID Controller ===========================
        # =================================================================

        ### Solving --> U =  Kp * E + Ki * int_E + Kd * dE
        ### State       X = [x, y, theta, v_x, v_y, theta_dot]
        ### Control     U = [F_x, F_y, tau]

        # Gain Dims = [6 states x 3 outputs]
        K_P = np.array([[-3, 0, 0, -2, 0, 0],           [0, -3, 0, 0, -2, 0],               [0, 0, -0.2, 0, 0, -0.5]])
        K_I = np.array([[0.001, 0, 0, 0.001, 0, 0],     [0, 0.001, 0, 0, 0.001, 0],         [0, 0, 0, 0, 0, 0]])
        K_D = np.array([[0, 0, 0, 0, 0, 0],             [0, 0, 0, 0 ,0 ,0],                 [0, 0, 0, 0 ,0 ,0]])

        pid_params =  {"p_gain": K_P, "i_gain": K_I, "d_gain": K_D,
                                "antiwindup": False, "max_error_integral": 1.0}

        self.PID_controller = PID(pid_params)
        self.dt = 0.05

        # Return team info
        return team_info
    

    def compute_errors(self):
        dead_sat_state  = self.dead_sat_state
        system_state    = self.system_state
        satellite_state = self.satellite_state

        # Dead satellite state
        dead_x          = dead_sat_state.pose.x
        dead_y          = dead_sat_state.pose.y        
        dead_theta      = dead_sat_state.pose.theta

        dead_vx         = dead_sat_state.twist.v_x
        dead_vy         = dead_sat_state.twist.v_y
        dead_omega      = dead_sat_state.twist.omega

        # Local offsets and tolerancing internally
        x_tolerance     = 0.05
        y_offset        = 0.25
        y_tolerance     = 0.05
        theta_tolerance = 0.1

        vx_tolerance    = 0.001
        vy_tolerance    = 0.001
        omega_tolerance = 0.001

        # Targets
        x_target        = dead_x + y_offset * np.cos(dead_theta - np.pi / 2)
        y_target        = dead_y + y_offset * np.sin(dead_theta - np.pi / 2)
        theta_target    = dead_theta
        
        vx_target       = dead_vx
        vy_target       = dead_vy
        omega_target    = dead_omega

        # Errors
        x_error         = satellite_state.pose.x        - x_target
        y_error         = satellite_state.pose.y        - y_target        
        theta_error     = satellite_state.pose.theta % (np.pi) - theta_target % (np.pi)
        
        vx_error        = satellite_state.twist.v_x     - vx_target
        vy_error        = satellite_state.twist.v_y     - vy_target
        omega_error     = satellite_state.twist.omega   - omega_target

        # Verbosey stuff
        self.logger.info(f' omega_error = {omega_error}, theta_error = {theta_error}'
                         f' x_error = {x_error},         y_error = {y_error}'
                         f' vx_error = {vx_error},       vy_error = {vy_error}')

        # Concantenate errors
        return np.array([x_error, y_error, theta_error, vx_error, vy_error, omega_error])
        

    def team_run(self, system_state, satellite_state, dead_sat_state):
        """ Takes in a system state, satellite states """

        # For computing errors later
        self.system_state    = system_state
        self.satellite_state = satellite_state
        self.dead_sat_state  = dead_sat_state

        # Get timedelta from elapsed time
        self.elapsed_time = system_state.elapsedTime.ToTimedelta()
        self.logger.info(f'Elapsed time: {self.elapsed_time}')

        self.counter += 1
        self.logger.info(f'Counter value: {self.counter}')

        # Access controller api through RL agent lol
        control = self.RL_model.env.control

        # Compute errors
        errors = self.compute_errors()

## ====================== Controller Switch ===========================


        if controller_choice == 'PID':
            actions = self.PID_controller.step(e = errors, delta_t = self.dt)

            # Applying the control actions
            [control.thrust.f_x, control.thrust.f_y, control.thrust.tau] = actions


        elif controller_choice == 'RL':
            environment = self.RL_model.env
            agent       = self.RL_model.agent

            # Agent identifies action from state
            actions = agent.act(states=errors, independent=True, deterministic=True)

            # Janky way to calculate reward
            reward = environment.rewarder(self.compute_errors(), satellite_state.fuel)

            # Update the agent
            agent.observe(terminal=False, reward=reward)

            # Aggregate the rewards
            self.total_reward += reward

            # Save model every 1000 steps
            if self.counter % 1000 == 0:
                agent.save_model(agent_weights)
                self.logger.info(f'Saved model at step {self.counter}')
                self.logger.info(f'Total Reward: {self.reward}')


## ======================================================================

        # Adding to histories
        self.errors = np.vstack((self.errors, errors))
        self.actions = np.vstack((self.actions, actions))

        # More verbosey stuff!
        print(f"Control action = {actions}")

        # Matplotlib to plot phase portraits on every 100th step
        if self.counter % 100 == 0:
            state_space(self.errors)

        self.prev_time = self.elapsed_time

        return control
    


    def team_reset(self) -> None:
        # Run any reset code
        pass
