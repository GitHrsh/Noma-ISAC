import numpy as np
from agents.ucb_agents import CollaborativeUCBAgent
from environment.wireless_env import NOMAISACEnvironment
from config.params import Config

def run_simulation():
    env = NOMAISACEnvironment()
    agents = [
        CollaborativeUCBAgent(i, Config.NUM_TARGETS * len(Config.POWER_LEVELS))
        for i in range(Config.NUM_SCDS)
    ]
    
    rewards = []
    for t in range(1000):
        env.update_dynamics(t)
        actions = [agent.select_action(t) for agent in agents]
        reward = env.step(actions, t)
        for agent, action in zip(agents, actions):
            agent.update(action, reward)
        rewards.append(reward)
        
        if t % 100 == 0:
            print(f"Episode {t}: Reward = {np.mean(rewards[-100:]):.2f}")
    
    return rewards

if __name__ == "__main__":
    run_simulation()
