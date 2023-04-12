from pytweening import linear, easeInOutSine, easeInOutElastic, easeInOutBack, easeInOutCirc
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import pandas as pd
import pickle

DURATION = 5 #seconds for each behavior
NEUTRAL = np.array([0.0, 0.0, 0.1]) # pan, tilt, eyes
MAX = np.array([0.78, 0.29, 0.41]) # pan, tilt, eyes
MIN = np.array([-0.78, -0.92, -0.16]) # pan, tilt, eyes

#prepend and append the neutral position for each DoF
def add_neutrals(sequence, type_index):
    sequence = [[0, NEUTRAL[type_index]]] + sequence + [[DURATION, NEUTRAL[type_index]]] 
    return sequence

def interp(sequence, method):
    res = []
    start_i = 0

    for t in np.linspace(0, DURATION, 10*DURATION):
        #increment the starting frame if we are ahead of it
        if t > sequence[start_i+1][0]:
            start_i += 1
            if start_i >= len(sequence):
                raise Exception('Time {t} is out of bounds for {sequence}')
        
        frac = (t - sequence[start_i][0]) / (sequence[start_i+1][0] - sequence[start_i][0])
        #interp according to the method
        frac = method(frac)
        res.append(sequence[start_i][1]*(1-frac) + sequence[start_i+1][1]*frac)
    
    return res


all_outputs = []

#TODO: read in from a file or something
pans = [[2,.2], [4,-.2]]
tilts = []
eyes = []

pans = add_neutrals(pans, 0)
tilts= add_neutrals(tilts, 1)
eyes = add_neutrals(eyes, 2)

for method in [linear, easeInOutSine, easeInOutElastic, easeInOutBack, easeInOutCirc]:
    pan_vector = np.clip(interp(pans,method),MIN[0], MAX[0])
    tilt_vector = np.clip(interp(tilts,method),MIN[1], MAX[1])
    eyes_vector = np.clip(interp(eyes,method),MIN[2], MAX[2])
    output = np.round(np.array([pan_vector, tilt_vector, eyes_vector]).T,2)

    all_outputs.append(output)

print(np.array(all_outputs).shape)
np.save('../data/behaviors',all_outputs)

