import random

def dummy_user(rng, acceptance_rate=0.5):
    """
    Dummy user function that returns a random boolean based on acceptance_rate.
    
    Args:
        rng (numpy.random.Generator): Random number generator to use
        acceptance_rate (float): Probability of accepting (0.0 to 1.0)
    
    Returns:
        bool: True if accepted, False if rejected
    """
    return rng.random() <= acceptance_rate

if __name__ == "__main__":
    print(dummy_user())