import numpy as np
import gymnasium as gym
import os
import qiskit
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from qiskit.quantum_info import random_density_matrix, random_statevector, DensityMatrix
from adaptive_qst.plotting import PlotOneQubit
from adaptive_qst.max_info import Posterior, HiddenState
import matplotlib.pyplot as plt
from numpy import pi, sqrt
from qiskit.quantum_info import state_fidelity

class AQSTEnv(gym.Env):
    
    def __init__(self, n_particles = 30, n_measurements = 1000, hidden_state = None):
        super(AQSTEnv, self).__init__()
        
        self.n_particles = n_particles
        self.n_measurements = n_measurements
        self.posterior = Posterior(self.n_particles)
        self.hidden_state = HiddenState(hidden_state)
        self.step_num = 0

        self.observation_space = gym.spaces.Box(low = -1, high = 1, shape = (4 * self.n_particles,))  ##3 variables per density matrix, 1 variable for weight
        
        self.action_space = gym.spaces.Box(low= 0, high = pi, shape = (2,))  ##Orientation of measurement

    def step(self, action):
        
        result = self.hidden_state.measure_along_axis(action)
        self.posterior.update(action, result)
        fidelity = state_fidelity(self.hidden_state.hidden_state, self.posterior.get_best_guess())
        reward = -np.log(1 - fidelity)

        self.step_num += 1
        truncated = (self.step_num >= self.n_measurements)
        terminated = False

        return (self.get_observations(), 
                reward, 
                terminated, 
                truncated, 
                {})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.posterior = Posterior(self.n_particles)
        self.hidden_state = HiddenState()
        self.step_num = 0
        
        return self.get_observations(), {}
    
    #Package complex density matrix and weights into observation vector
    def get_observations(self):
        observations = np.zeros(4 * self.n_particles)
        observations[:self.n_particles] = np.real(self.posterior.particle_states[:, 0, 0])
        observations[self.n_particles : 2 * self.n_particles] = np.real(self.posterior.particle_states[:, 0, 1])
        observations[2 * self.n_particles : 3 * self.n_particles] = np.imag(self.posterior.particle_states[:, 0, 1])
        observations[3 * self.n_particles:] = self.posterior.particle_weights

        return observations.astype(np.float32)
        