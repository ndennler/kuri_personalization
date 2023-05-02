import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

Xs = np.arange(0,5, .1)
data = np.load('../data/behaviors.npy')
i=0
for traj in tqdm(data):
    # print(traj.shape)
    plt.clf()
    plt.plot(Xs, traj[:,0], label="pan")
    plt.plot(Xs, traj[:,1], label='tilt')
    plt.plot(Xs, traj[:,2], label='eyes')
    plt.legend()

    plt.savefig(f'../data/imgs/{i}.png', bbox_inches='tight')
    i += 1