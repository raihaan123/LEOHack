import numpy as np

class PID:

    def __init__(self, controller_parameters):
        self.k_p     = controller_parameters['p_gain']   # proportional gain
        self.k_d     = controller_parameters['d_gain']   # derivative gain   
        self.k_i     = controller_parameters['i_gain']   # integral gain
        
        self.antiwindup         = controller_parameters['antiwindup'] 
        self.max_error_integral = controller_parameters['max_error_integral']

        self.past_error         = 0.0
        self.error_sum          = 0.0
        self.curr_error_sign   = None
        self.prev_error_sign   = None


    def step(self, e, delta_t):
        """ Takes error 'e' and provides control action 'u'. 
            'delta_t' should be time interval since the last control step (used for estimating derivative term). 
        """

        self.error_sign     = np.sign(e)
        self.error_sum      += e

        if self.antiwindup: self.antiwindup_measures()

        prop_term    = np.dot(self.k_p, e)
        der_term     = np.dot(self.k_d, ((e - self.past_error) / delta_t))
        int_term     = np.dot(self.k_i, self.error_sum)

       # print(f"prop_term = {prop_term}\nder_term = {der_term}\nint_term = {int_term}\n")

        u            = prop_term + der_term + int_term

        self.past_error         = e
        self.prev_error_sign    = self.error_sign

        # print(f"error_sum = {self.error_sum}")
        # print(f"e = {e}")

        return u


    def antiwindup_measures(self):

        # If error changes sign, reset error integral
        # if (self.prev_error_sign is not None) and (self.prev_error_sign != self.error_sign):
        #     self.error_sum = 0.0
        if np.all(self.prev_error_sign is not None): #and (self.prev_error_sign != self.error_sign):
            self.error_sum[self.prev_error_sign != self.error_sign] = 0.0
            #print(f"Reset_Error - {self.error_sum}")
        # # Limits absolute value of error sum to be below the set threshold
        # if (np.abs(self.error_sum) > self.max_error_integral):
        #     self.error_sum = self.error_sign * self.max_error_integral
        #if (np.abs(self.error_sum) > self.max_error_integral):
        self.error_sum[np.abs(self.error_sum) > self.max_error_integral] = (self.error_sign * self.max_error_integral)[[np.abs(self.error_sum) > self.max_error_integral]]
