## Controller choice --> 'PID', 'RL'
controller_choice = 'PID'


from sat_controller import SatControllerInterface, sat_msgs
from datetime import datetime, timedelta
import numpy as np

# Additional imports
from pid import PID
from plotter import state_space
import sys

from tensorforce import Environment
from RL import RL_Model

# Team code is written as an implementation of various methods
# within the the generic SatControllerInterface class.
# If you would like to see how this class works, look in sat_control/sat_controller

# Specifically, init, run, and reset

class SatelliteSystem(Environment):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state = np.zeros(6)
        self.action = np.zeros(3)
        self.time = 0
        self.fuel = 0
        self.terminal = False
        self.reward = 0

    def states(self):
        return dict(type='float', shape=(6,))

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
        self.time += 1
        self.fuel -= 0.1
        self.reward = 0.1 * self.time + 0.9 * self.fuel
        return self.state, self.terminal, self.reward


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

        self.has_docked     = False

        # # Initialize RL environment wrapper, agent and model
        # self.RL_satellite = Environment.create(
        #     environment=SatelliteSystem, max_episode_timesteps=500
        # )


        # self.RL_model = RL_Model(self.RL_satellite)


        
        ### Solving --> U =  Kp * E + Ki * int_E + Kd * dE
        ### State       X = [x, y, theta, v_x, v_y, theta_dot]
        ### Control     U = [F_x, F_y, tau]

        # Gain Dims = [6 states x 3 outputs]
        K_P = np.array([[-3, 0, 0, -2, 0, 0],           [0, -3, 0, 0, -2, 0],               [0, 0, -0.2, 0, 0, -0.5]])
        K_I = np.array([[0.001, 0, 0, 0.001, 0, 0],     [0, 0.001, 0, 0, 0.001, 0],         [0, 0, 0, 0, 0, 0]])
        K_D = np.array([[0, 0, 0, 0, 0, 0],             [0, 0, 0, 0 ,0 ,0],                 [0, 0, 0, 0 ,0 ,0]])

        pid_params =  {"p_gain": K_P, "i_gain": K_I, "d_gain": K_D,
                                "antiwindup": False, "max_error_integral": 1.0}

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

        # Create a thrust command message
        control = sat_msgs.ControlMessage()

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

        error = np.array([x_error, y_error, theta_error, vx_error, vy_error, omega_error])

        # If using PID, run this
        if controller_choice == 'PID':
            control_actions = self.controller.step(e = error, delta_t = self.dt)

        else:
            # If using RL, run this
            control_actions = self.RL_model.step(error, self.elapsed_time)

        # More verbosey stuff!
        print(f"control action = {control_actions}")
        

        # control.thrust.f_x = control_actions[0]
        # control.thrust.f_y = control_actions[1]
        # control.thrust.tau = control_actions[2]
        [control.thrust.f_x, control.thrust.f_y, control.thrust.tau] = control_actions
        
        # Flags
        pose_check      = False
        theta_check     = False

        # Check position tolerance
        if abs(x_error) < x_tolerance and abs(y_error) < y_tolerance:           
            pose_check = True
        else:                                                                   
            pose_check = False

        # Check angular tolerance
        if abs(theta_error) < theta_tolerance and abs(omega_error) < omega_tolerance:     
            theta_check = True
        else:                                                                   
            theta_check = False

        # Add error vector as a row to the self.errors matrix
        self.errors = np.vstack((self.errors, error))
        # Same with actions
        self.actions = np.vstack((self.actions, control_actions))

        # Final post-docking processing
        if pose_check and theta_check:      
            print(pose_check, theta_check)
            self.has_docked = True

            # Matplotlib to plot vy errors vs x errors - both located in self.errors
            state_space(self.errors)



        self.prev_time = self.elapsed_time

        return control
    

    



    def team_reset(self) -> None:
        # Run any reset code
        pass