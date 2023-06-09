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

def read_animation_csv(file):
    all_animations = []
    names = []
    data = pd.read_csv(file)
    for _, row in data.iterrows():
        this_animation = []
        for i in range(1,5):
            if not np.isnan(row[i]):
                this_animation.append([i, row[i]])
        all_animations.append(this_animation)
        names.append(row[0])
    
    return all_animations, names

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

csv = []


pans, pnames = read_animation_csv('../data/pans.csv')
tilts, tnames = read_animation_csv('../data/tilts.csv')
eyes, enames = read_animation_csv('../data/eyes.csv')


index=0
for pan,pname in zip(pans,pnames):
    pan = add_neutrals(pan, 0)

    for tilt,tname in zip(tilts, tnames):
        tilt= add_neutrals(tilt, 1)

        for eye, ename in zip(eyes, enames):
            eye = add_neutrals(eye, 2)

            for method, mname in zip([linear, easeInOutSine, easeInOutElastic, easeInOutBack, easeInOutCirc], ['lin', 'sine','elastic','back','circ']):
                # print(f'{pan},\n{tilt},\n{eye}')
                pan_vector = np.clip(interp(pan,method),MIN[0], MAX[0])
                tilt_vector = np.clip(interp(tilt,method),MIN[1], MAX[1])
                eyes_vector = np.clip(interp(eye,method),MIN[2], MAX[2])
                output = np.round(np.array([pan_vector, tilt_vector, eyes_vector]).T,2)

                all_outputs.append(output)
                csv.append({
                    'id': index,
                    'name': f'{tname}_{pname}_{ename}_{mname}'
                })
                index+=1

print(np.array(all_outputs).shape)
np.save('../data/behaviors', all_outputs)
pd.DataFrame(csv).to_csv('../data/movement_names.csv')

