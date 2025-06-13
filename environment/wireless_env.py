import numpy as np
import torch
from config.params import Config
from .channel import rayleigh_channel

class NOMAISACEnvironment:
    def __init__(self):
        self.num_scds = Config.NUM_SCDS
        self.num_targets = Config.NUM_TARGETS
        self.positions = self._initialize_positions()
        self.velocities = np.random.randn(self.num_scds, 2)*0.01
        self.sensing_matrices = self._calculate_sensing_matrices()
        
    def _initialize_positions(self) -> np.ndarray:
        """Initialize SCDs and targets on a 2D grid."""
        scd_pos = np.linspace(0, 1, self.num_scds)
        target_pos = np.linspace(0, 1, self.num_targets)
        return np.array([[x, y] for x in scd_pos for y in target_pos][:self.num_scds])
    
    def _calculate_sensing_matrices(self) -> np.ndarray:
        """Placeholder: Implement actual sensing matrix calculation."""
        return np.random.randn(self.num_scds, self.num_targets, Config.NT, Config.NT)
    
    def step(self, actions: list[int], t: int) -> float:
        """Execute one timestep, return global reward."""
        # 1. Convert actions to power allocations
        power_alloc = [Config.POWER_LEVELS[a // self.num_targets] 
                      for a in actions]
        targets = [a % self.num_targets for a in actions]
        
        # 2. Compute channels
        channels = [
            rayleigh_channel(
                np.linalg.norm(self.positions[i] - self.positions[targets[i]]),
                t,
                Config.DOPPLER_FREQ
            ) for i in range(self.num_scds)
        ]
        
        # 3. Compute metrics (simplified)
        throughput = sum([np.log2(1 + p*np.abs(h).mean()**2) 
                         for p, h in zip(power_alloc, channels)])
        sensing_rate = sum([np.random.rand()*p for p in power_alloc])  # Placeholder
        fairness = (sum(power_alloc)**2) / (self.num_scds * sum([p**2 for p in power_alloc]))
        
        # 4. Power penalty
        power_penalty = max(0, sum(power_alloc) - Config.POWER_BUDGET) * Config.POWER_PENALTY
        
        # 5. Composite reward
        reward = (
            Config.THROUGHPUT_WEIGHT * throughput +
            Config.SENSING_WEIGHT * sensing_rate +
            Config.FAIRNESS_WEIGHT * fairness -
            power_penalty
        )
        return reward
    
    def update_dynamics(self, t: int):
        """Update positions and channel states."""
        self.positions += self.velocities
        self.sensing_matrices = self._calculate_sensing_matrices()
