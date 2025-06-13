import numpy as np
from typing import List
from config.params import Config

class CollaborativeUCBAgent:
    def __init__(self, agent_id: int, num_actions: int):
        self.id = agent_id
        self.num_actions = num_actions
        self.alpha = Config.ALPHA
        
        # Bandit parameters
        self.global_counts = np.ones(num_actions)
        self.global_rewards = np.zeros(num_actions)
        self.local_counts = np.ones(num_actions)
        self.local_rewards = np.zeros(num_actions)
        
    def select_action(self, t: int) -> int:
        """UCB action selection with collaborative exploration."""
        total_counts = self.alpha*self.global_counts + (1-self.alpha)*self.local_counts
        mean_rewards = (self.alpha*self.global_rewards + (1-self.alpha)*self.local_rewards)
        mean_rewards /= (total_counts + 1e-6)
        
        ucb = mean_rewards + np.sqrt(2*np.log(t+1)/(total_counts + 1e-6))
        return np.argmax(ucb)
    
    def update(self, action: int, reward: float):
        """Update local and global statistics."""
        self.local_counts[action] += 1
        self.local_rewards[action] += reward
        self.global_counts[action] += 1  # Simplified: replace with message passing
