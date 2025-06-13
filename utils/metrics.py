import numpy as np

def jains_fairness(rates: list[float]) -> float:
    """Compute Jain's fairness index."""
    return (sum(rates)**2) / (len(rates) * sum([r**2 for r in rates]))

def calculate_qos_violation(scd_power: list[float], 
                           min_throughput: list[float], 
                           actual_throughput: list[float]) -> float:
    """Calculate QoS violation penalty."""
    return sum([max(0, min_t - actual) 
               for min_t, actual in zip(min_throughput, actual_throughput)])
def calculate_power_violation(
    scd_power: list[float], 
    max_power: float) -> list[float]:
    """Calculate per-SCD power violation.
    Args:
        scd_power: List of power used by each SCD.
        max_power: Maximum allowed power for each SCD.
    Returns:
        List of violations (positive if power exceeds max_power, else 0).
    """
    return [max(0, p - max_power) for p in scd_power]

def calculate_total_power_violation(
    scd_power: list[float], 
    max_power: float
) -> float:
    """Calculate total power violation across all SCDs.
    Args:
        scd_power: List of power used by each SCD.
        max_power: Maximum allowed power for each SCD.
    Returns:
        Sum of all individual power violations.
    """
    return sum(calculate_power_violation(scd_power, max_power))
