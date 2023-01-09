import numpy as np
import random

def distribute_indices(length, alpha, c_num):

    ratios = np.round(np.random.dirichlet(np.repeat(alpha, 5))*length).astype(int)
    indices = list(range(length))
    random.shuffle(indices)

    ### Client num should be greater than 
    if sum(ratios) > length:
        ratios[4] -= (sum(ratios) - length)
    else:
        ratios[4] += (length - sum(ratios))

    indices0 = []
    indices1 = []
    indices2 = []
    indices3 = []
    indices4 = []

    for i in range(0,ratios[0]):
        indices0.append(indices[i])
    for i in range(ratios[0],ratios[0] + ratios[1]):
        indices1.append(indices[i])
    for i in range(ratios[0] + ratios[1],ratios[0] + ratios[1] + ratios[2]):
        indices2.append(indices[i])
    for i in range(ratios[0] + ratios[1] + ratios[2],ratios[0] + ratios[1] + ratios[2] + ratios[3]):
        indices3.append(indices[i])
    for i in range(ratios[0] + ratios[1] + ratios[2] + ratios[3], length):
        indices4.append(indices[i])

    indices = [indices0,indices1,indices2,indices3,indices4]

    return indices