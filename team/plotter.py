# Functions to Plot stuffs!

import matplotlib.pyplot as plt
import numpy as np


def state_space(errors):
    """
    Plot the phase portraits of the satellite.
    """
    # Extract time series - same number of rows as errors matrix, starting at zero in increments of 0.05
    print(errors.shape)
    # n elements, from 0 increasing in 0.05
    ts = [i*0.05 for i in range(0, errors.shape[0])]

    # Find r from x and y error values
    r = np.sqrt(errors[:,0]**2 + errors[:,1]**2)

    # Speed vs radial distance
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].plot(r, errors[:,3], 'b-')
    ax[0].set_xlabel('Radial Distance (m)')
    ax[0].set_ylabel('X component of velocity (m/s)')
    ax[0].set_title('X Velocity vs Radial Distance')
    ax[0].grid(True)

    # Theta vs radial distance
    ax[1].plot(r, errors[:,2], 'b-')
    ax[1].set_xlabel('Radial Distance (m)')
    ax[1].set_ylabel('Theta (rad)')
    ax[1].set_title('Theta vs Radial Distance')
    ax[1].grid(True)

    # Theta vs time
    ax[2].plot(ts, errors[:,2], 'b-')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Theta (rad)')
    ax[2].set_title('Theta vs Time')
    ax[2].grid(True)
    fig.savefig('phase_portraits.png')
