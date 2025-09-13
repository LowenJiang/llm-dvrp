import random

def dummy_user(acceptance_rate=0.5):
    return random.random() <= acceptance_rate

if __name__ == "__main__":
    print(dummy_user())