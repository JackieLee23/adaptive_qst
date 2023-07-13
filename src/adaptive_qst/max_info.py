import numpy as np
from numpy import cos, sin, exp, pi, sqrt, log, transpose, conjugate
from numpy.linalg import cholesky

import matplotlib.pyplot as plt

from qiskit.quantum_info import random_statevector, random_density_matrix
from qiskit.quantum_info import state_fidelity
from qiskit import QuantumCircuit

from scipy import stats
from scipy.optimize import minimize



def entropy(probs):
    return -np.sum(probs * np.log(probs), axis = -1)
    
def orient_measure(configuration):
    theta, phi = configuration
    return np.array([[cos(theta/2), sin(theta/2)*exp(-phi * 1j)],
                     [-sin(theta/2), cos(theta/2)*exp(-phi * 1j)]])

def get_adjoint(matrix):
    return conjugate(transpose(matrix))

def get_squared_bures(state_1, state_2):
    return 2 - 2 * sqrt(state_fidelity(state_1, state_2))

##Represents the posterior distribution; updates the distribution based on new data; auto-resamples when effective sample size is too small. 
class Posterior:
    
    possible_results = [0, 1]
    default_config = [pi/2, pi/2]
    config_bounds = [[0, pi], [0, pi]]
    dims = 2
    
    def __init__(self, n_particles = 100, resampling_tolerance = 0.1):
        self.n_particles = n_particles
        self.particle_states, self.particle_weights = self.initialize_particles()
        
        self.config_arr = []
        self.result_arr = []
        self.resampling_tolerance = resampling_tolerance
        
    def initialize_particles(self):
        return np.array([random_density_matrix(self.dims).data for i in range(self.n_particles)]), np.ones(self.n_particles) / self.n_particles
        
    def get_likelihood(self, configuration, result, particle_states):

        orient_unitary = orient_measure(configuration)
        rotated_states = orient_unitary @ particle_states @ get_adjoint(orient_unitary)
        
        return np.real(rotated_states[:, result, result])
    
    def update_weights(self, configuration, result):
        posterior = self.get_likelihood(configuration, result, self.particle_states) * self.particle_weights
        self.particle_weights = posterior / np.sum(posterior)

    def get_best_guess(self):
        return np.sum(self.particle_states * self.particle_weights[:, np.newaxis, np.newaxis], axis = 0)
 
    def get_info_gain(self, configuration):
        likelihoods = self.get_likelihood(configuration, self.possible_results, self.particle_states)
        
        avg_predictive_prob = np.sum(likelihoods * self.particle_weights[:, np.newaxis], axis = 0)
        uniformity = entropy(avg_predictive_prob)
        
        confidence = -np.sum(entropy(likelihoods) * self.particle_weights)
        return uniformity + confidence
    
    def get_best_config(self):
        optimized = minimize(lambda x: -self.get_info_gain(x), 
                         x0 = self.default_config,
                         bounds = self.config_bounds)
    
        return optimized.x
    

    ########################################
    ### Functions for MCMC Resampling  ###
    #########################################
        
    def get_effective_sample_size(self):
        return 1 / np.sum(self.particle_weights ** 2)
    
    def redraw_particles(self):
        indices = np.random.choice(np.arange(self.n_particles), 
                               p = self.particle_weights, 
                               size = self.n_particles)
        
        self.particle_states = self.particle_states[indices]
        self.particle_weights = np.ones(self.n_particles) / self.n_particles


    def get_purified_states(self, particle_states):
        return cholesky(particle_states).reshape(self.n_particles, self.dims ** 2)
    
    def get_traced_out(self, pure_states):
        pure_states_3d = pure_states.reshape(self.n_particles, self.dims, self.dims)

        return np.einsum('ijk,ilk->ijl', pure_states_3d, pure_states_3d.conj())
    
    def get_posterior_size(self):
        best_guess = self.get_best_guess()
        distances = np.array([get_squared_bures(state, best_guess) for state in self.particle_states])

        return sqrt(np.sum(distances * self.particle_weights))
        
    def get_full_log_likelihood(self, particle_states):
        log_likelihoods = np.zeros(self.n_particles)
        
        for configuration, result in zip(self.config_arr, self.result_arr):
            log_likelihoods += np.log(self.get_likelihood(configuration, result, particle_states))
            
        return log_likelihoods
    
    def draw_next_pos(self):
        random_vecs = np.random.normal(size = (self.n_particles, self.dims ** 2)) + np.random.normal(size = (self.n_particles, self.dims ** 2)) * 1j

        overlaps = np.einsum('ij,ij->i', np.conjugate(self.purified_states), random_vecs)

        step_vecs = random_vecs - self.purified_states * overlaps[:, np.newaxis]

        step_norms = (np.linalg.norm(step_vecs, axis = 1))

        step_vecs /= step_norms[:, np.newaxis]

        a = 1 - (np.random.normal(scale = self.posterior_size, size = self.n_particles) ** 2) / 2
        a[(a < 0)] = 0

        b = sqrt(1 - a ** 2)

        next_purified = a[:, np.newaxis] * self.purified_states + b[:, np.newaxis] * step_vecs

        next_purified /= np.linalg.norm(next_purified, axis = 1)[:, np.newaxis]

        return next_purified, self.get_traced_out(next_purified)
    
    
    def mc_step(self):

        next_pure, next_states = self.draw_next_pos()
        next_log_likelihoods = self.get_full_log_likelihood(next_states)
        
        acceptance_ratios = next_log_likelihoods - self.log_likelihoods
        accepted = (log(np.random.rand(self.n_particles)) < acceptance_ratios)
        
        self.particle_states[accepted] = next_states[accepted]
        self.purified_states[accepted] = next_pure[accepted]
        self.log_likelihoods[accepted] = next_log_likelihoods[accepted]

    def update_weight_sequence(self):
    
        for configuration, result in zip(self.config_arr, self.result_arr):
            self.update_weights(configuration, result)

    def resample(self, n_steps = 50):
    
        self.posterior_size = self.get_posterior_size()
        self.redraw_particles()

        self.purified_states = self.get_purified_states(self.particle_states)
        self.log_likelihoods = self.get_full_log_likelihood(self.particle_states)

        for i in range(n_steps): self.mc_step()

        self.update_weight_sequence()
    
    #Main interacting function
    def update(self, config, result):
        self.update_weights(config, result)
        self.config_arr.append(config)
        self.result_arr.append(result)

        if self.get_effective_sample_size() < (self.resampling_tolerance * self.n_particles):
            #print(f"Resampling with MC")
            self.resample()

   
class HiddenState:
    def __init__(self, hidden_state = None):
        if hidden_state is None:
            hidden_state = random_density_matrix(2).data
        self.hidden_state = hidden_state
    
    #Returns 0 for spin UP along axis, 1 for spin DOWN along axis!
    def measure_along_axis(self, configuration):
        orient_unitary = orient_measure(configuration)
        
        rotated_state = orient_unitary @ self.hidden_state @ get_adjoint(orient_unitary)
        
        return int(np.random.rand() > rotated_state[0][0])

        