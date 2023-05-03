'''
The purpose of this script is to synchronize the data in all the places it needs to be. 
This involves copying over all of the sound/audio/kinesthetic files and the csvs that 
contain the indices for all of the stimuli.
'''

import pandas as pd
import os 
import numpy as np

# id, type, path, tags

data = []


#
#     VISUAL
#
df = pd.read_csv('visual/data/icons.csv')

for i, row in df.iterrows():
    data.append({
        'id':i,
        'type': 'Video',
        'file': f"{row['mp4_link'].split('/')[-1]}",
        'tags': row['Icon_tags']
    })


#
#     AUDITORY
#

df = pd.read_csv('auditory/sound-effect-library/files.csv')


for i, row in df.iterrows():
    data.append({
        'id':i,
        'type': 'Audio',
        'file': f"{row[1]}",
        'tags': f"{row[1].split('.')[0]}",
    })



#
#     Kinesthetic
#
df = pd.read_csv('kinesthetic/data/movement_names.csv')


for i, row in df.iterrows():
    data.append({
        'id':f"{row['id']}",
        'type': 'Movement',
        'file': f"{row['name']}",
        'tags': f"{row['name']}",
    })


print(data)
pd.DataFrame(data).to_csv('../web_interfaces/data/all_data.csv')