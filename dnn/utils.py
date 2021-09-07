import numpy as np


def set_trainable(module, trainable=True):
    for param in module.parameters():
        param.requires_grad = trainable
        
def random_steps(max_steps):
    return max(1, max_steps - abs(int(np.random.normal(0, max_steps / 2, 1)[0])))
