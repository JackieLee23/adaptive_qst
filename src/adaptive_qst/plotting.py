import numpy as np
import matplotlib.pyplot as plt
from adaptive_qst.max_info import Posterior

class PlotOneQubit:
    
    def get_components(states):
        return np.array([2 * np.real(states[:, 0, 1]), 
                         2 * np.imag(states[:, 1, 0]), 
                         2 * np.real(states[:, 0, 0]) - 1])
    
    def plot_states(states, projection, ax, size = 10):
        comp_1 = ord(projection[0]) - ord('x')
        comp_2 = ord(projection[1]) - ord('x')

        components = PlotOneQubit.get_components(states)
        ax.scatter(components[comp_1], components[comp_2], s = size, color = "red")

    
    ###projection = {'xy', 'xz', 'yz'}
    def plot_posterior(posterior: Posterior, projection, ax):
        comp_1 = ord(projection[0]) - ord('x')
        comp_2 = ord(projection[1]) - ord('x')

        components = PlotOneQubit.get_components(posterior.particle_states)

        ax.set_xlabel(projection[0])
        ax.set_ylabel(projection[1])
    
        return ax.scatter(components[comp_1], components[comp_2], c = posterior.particle_weights, cmap = 'viridis')

        
        
                                                     
                                                     
                                                     
                                           


