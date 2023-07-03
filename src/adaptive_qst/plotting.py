import numpy as np
import matplotlib.pyplot as plt
from adaptive_qst.max_info import Posterior

class PlotOneQubit:
    
    def get_components(state):
        return 2 * np.real(state[0, 1]), 2 * np.imag(state[1, 0]), 2 * np.real(state[0, 0]) - 1
    
    def plot_state(state, projection, ax):
        comp_1 = ord(projection[0]) - ord('x')
        comp_2 = ord(projection[1]) - ord('x')

        components = PlotOneQubit.get_components(state)
        ax.scatter(components[comp_1], components[comp_2], s = 50, color = "red")

    
    ###projection = {'xy', 'xz', 'yz'}
    def plot_posterior(posterior: Posterior, projection, ax):
        comp_1 = ord(projection[0]) - ord('x')
        comp_2 = ord(projection[1]) - ord('x')

        components = np.array([PlotOneQubit.get_components(state) for state in posterior.particle_states])

        ax.set_xlabel(projection[0])
        ax.set_ylabel(projection[1])
    
        return ax.scatter(components[:, comp_1], components[:, comp_2], c = posterior.particle_weights, cmap = 'viridis')

        
        
                                                     
                                                     
                                                     
                                           


