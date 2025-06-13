class Config:
    # System
    NUM_SCDS = 4
    NUM_TARGETS = 4
    NT = 8  # Transmit antennas
    POWER_LEVELS = [0.2, 0.5, 1.0]
    POWER_BUDGET = 3.0  # Total system power
    
    # Channel
    PATH_LOSS_EXPONENT = 3.0
    DOPPLER_FREQ = 0.1  # Normalized Doppler
    
    # Learning
    ALPHA = 0.1  # Collaboration weight
    GAMMA = 0.9  # Discount factor (for future extensions)
    
    # Reward Weights
    THROUGHPUT_WEIGHT = 0.5
    SENSING_WEIGHT = 0.3
    FAIRNESS_WEIGHT = 0.2
    POWER_PENALTY = 1.0
