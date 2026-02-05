import numpy as np
def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """

    # Write code here
    '''
    Approach:
    1. if current step is 0 return  base_rl
    2. if current step is total_steps return min_lr
    '''
    if current_step <= 0:
        return base_lr
    if current_step >= total_steps:
        return min_lr

    #formulae is cos_ann(curr_step) = min_lr + 0.5 *(base_lr - min_lr) *
    # (1 + np.cos(pi * curr_Step / total_steps))
    cosine_part = 1 + np.cos(np.pi * (current_step / total_steps))
    result = min_lr + 0.5 * (base_lr - min_lr) * cosine_part
    return result