## Controller choice --> 'PID', 'RL', 'MPC'
controller_choice = 'PID'

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

# Team code is written as an implementation of various methods
# within the the generic SatControllerInterface class.
# If you would like to see how this class works, look in sat_control/sat_controller

# Specifically, init, run, and reset

# RL Environment Wrapper - interacts with the Sim object
class SatelliteSystem(Environment):
    def __init__(self):
        super().__init__()
        self.state = np.zeros(6)
        self.action = np.zeros(3)
        self.time = 0
        self.fuel = 0
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

    def reset(self, init_state=np.random.random(size=(6,))):
        self.state = init_state
        self.time = 0
        self.fuel = 0
        self.reward = 0
        return self.state

    def execute(self, state, actions):
        self.state = state
        self.action = actions

        self.reward = 0.1 * self.time + 0.9 * self.fuel
        return self.state, self.terminal, self.reward



class RL_Model:
    def __init__(self, env):
        self.env = env

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



class TeamController(SatControllerInterface):
    """ Team control code """

    def team_init(self):
        """ Runs any team based initialization """
        # Run any initialization you need

        # Example of persistant data
        self.counter = 0

        # Example of logging
        self.logger.info("Initialized :)")
        self.logger.warning("Warning...")
        self.logger.error("Error!")

        # Update team info
        team_info = sat_msgs.TeamInfo()
        team_info.teamName = "Group 16"
        team_info.teamID = 1111

        # Mission performance histories - N x 6 array of errors and N x 3 array of actions
        self.errors         = np.zeros((0,6))
        self.actions        = np.zeros((0,3))

        # Initialize RL environment wrapper, agent and model
        self.RL_satellite = Environment.create(
            environment=SatelliteSystem,
            max_episode_timesteps=500
        )


        self.RL_model = RL_Model(self.RL_satellite)


        # =================================================================
        # ====================== PID Controller ===========================
        # =================================================================

        ### Solving --> U =  Kp * E + Ki * int_E + Kd * dE
        ### State       X = [x, y, theta, v_x, v_y, theta_dot]
        ### Control     U = [F_x, F_y, tau]

        # Gain Dims = [6 states x 3 outputs]
        K_P = np.array([[-5, 0, 0, -15, 0, 0],          [0, -5, 0, 0, -15, 0],             [0, 0, -7, 0, 0, -3]])
        K_I = np.array([[-1, 0, 0, -1, 0, 0],         [0, -1, 0, 0, -1, 0],                  [0, 0, 0.0, 0, 0, 0]])
        K_D = np.array([[0, 0, 0, 0, 0, 0],             [0, 0, 0, 0 , 0, 0],                 [0, 0, 0, 0 ,0 ,0]])

        pid_params =  {"p_gain": K_P, "i_gain": K_I, "d_gain": K_D,
                                "antiwindup": True, "max_error_integral": 0.5}

        self.controller = PID(pid_params)
        self.dt = 0.05

        # Return team info
        return team_info
        
    

    def team_run(self, system_state, satellite_state, dead_sat_state):
        """ Takes in a system state, satellite states """

        # Get timedelta from elapsed time
        self.elapsed_time = system_state.elapsedTime.ToTimedelta()
        self.logger.info(f'Elapsed time: {self.elapsed_time}')

        self.counter += 1
        self.logger.info(f'Counter value: {self.counter}')

        control = self.RL_model.env.control

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

        # Distance to dead satellite
        dist2dead       = np.sqrt( (satellite_state.pose.x - dead_x)**2 + (satellite_state.pose.y - dead_y)**2 )

        # v

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

        error = np.array([x_error, y_error, theta_error, vx_error, vy_error, omega_error])
        
        # Velocity 
        satellite_vel = np.sqrt(satellite_state.twist.v_x**2 + satellite_state.twist.v_y **2)


        # Ensure that there is n overspeed anywhere - therefore we need to cap the thrust prior to arrival at the keepout zone
        # Define a keepout 'horizon'? Or a safety margin

        # Find which spatial region the satellite is in ---> Beyond kepout outer, vmax = 1, beyond keepout inner, vmax=1 but slowing to 0.2, or inside keepout, vmax = 0.2
        
        sphere_radius = 0.5
        keepout_dist = 0.5 # Keepout zone margin (beyond the sphere)


        if (dist2dead < sphere_radius + keepout_dist + 0.3):
            if (dist2dead  < sphere_radius + 0.3):               # Inside sphere
                print(f"inside_sphere")
                target_v = 0.05
                x_target = x_target
                x_setpoints = np.linspace(satellite_state.pose.x, x_target, num=10)             # Define x_target above

                y_target = y_target
                y_setpoints = np.linspace(satellite_state.pose.y, y_target, num=10)             # Define y_target above

            else:  
                print(f"in keepout dist")
                target_v = 0.2                 # Keepout inner
                x_target = x_target + sphere_radius * np.cos(dead_theta - np.pi / 2)
                x_setpoints = np.linspace(satellite_state.pose.x, x_target, num=10)             # Define x_target above 

                y_target = y_target + sphere_radius * np.sin(dead_theta - np.pi / 2)
                y_setpoints = np.linspace(satellite_state.pose.y, y_target, num=10)             # Define y_target above

        else:  
            print(f"outside shpere")
            target_v = 1.0                   # Outside keepout outer
            x_target = x_target + (sphere_radius + keepout_dist) * np.cos(dead_theta - np.pi / 2)
            x_setpoints = np.linspace(satellite_state.pose.x, x_target, num=10)             # Define x_target above 

            y_target = y_target + (sphere_radius + keepout_dist) * np.sin(dead_theta - np.pi / 2)
            y_setpoints = np.linspace(satellite_state.pose.y, y_target, num=10)             # Define y_target above


        # Direction vector from satellite to target
        direction_vec = np.array([x_target - satellite_state.pose.x, y_target - satellite_state.pose.y])
        direction_vec = direction_vec / np.linalg.norm(direction_vec)

        target_vx = target_v * direction_vec[0]
        target_vy = target_v * direction_vec[1]

        print(x_setpoints)
        print(y_setpoints)
        # calculate velocity/distance gradient
        r_target = np.sqrt(x_target**2 + y_target**2)
        # grad = (target_v - satellite_vel) / (dist2dead - r_target)
        # print(f'grad:{grad}')


        # Find sequence of x, vx setpoints that follow the trajectory modelled by the gradient in v/x space starting at the current satellite position - make it relative to the dead satellite position
        x_setpoints = np.linspace(satellite_state.pose.x, x_target, num=10)[8]             # Define x_target above
        y_setpoints = np.linspace(satellite_state.pose.y, y_target, num=10)[8]             # Define x_target above
        
        vx_setpoints = np.linspace(satellite_state.twist.v_x, target_vx, num=10)[8]        # Define target_v above
        vy_setpoints = np.linspace(satellite_state.twist.v_y, target_vy, num=10)[8]        # Define target_v above

        # PID 
        # Errors
        x_error         = satellite_state.pose.x        - x_setpoints
        y_error         = satellite_state.pose.y        - y_setpoints        
        theta_error     = satellite_state.pose.theta % (np.pi) - theta_target % (np.pi)
        
        vx_error        = satellite_state.twist.v_x     - vx_setpoints
        vy_error        = satellite_state.twist.v_y     - vy_setpoints
        omega_error     = satellite_state.twist.omega   - omega_target

        error = np.array([x_error, y_error, theta_error, vx_error, vy_error, omega_error])

        control_actions = self.controller.step(e = error, delta_t = self.dt)

        [control.thrust.f_x, control.thrust.f_y, control.thrust.tau] = control_actions

        self.errors = np.vstack((self.errors, error))
        # self.actions = np.vstack((self.actions, control_actions))

        # Matplotlib to plot vy errors vs x errors - both located in self.errors
        if self.counter % 10 == 0:
            state_space(self.errors)


        self.prev_time = self.elapsed_time

        print(f"F_x = {control.thrust.f_x}; F_y = {control.thrust.f_y}, Tau = {control.thrust.tau}")

        return control
    


    def team_reset(self) -> None:
        # Run any reset code
        pass