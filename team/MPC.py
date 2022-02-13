import numpy as np
import do_mpc


class MPC():

    def __init__(self, type="continuous"):
        
        
        self.model = do_mpc.model.Model(type)


        
        phi = self.model.set_variable(var_type='_x', var_name='phi', shape=(3,1))
        dphi = self.model.set_variable(var_type='_x', var_name='dphi', shape=(3,1))
        
        # Inputs to system
        phi_input = self.model.set_variable(var_type='_u', var_name='phi_m', shape=(3,1))
        
        mass = self.model.set_variable('parameter', 'mass')

        