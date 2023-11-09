import numpy as np
import os
from enum import Enum

class Feature(Enum):
     PAN = 0
     TILT = 1
     EYES = 2

# load features from file
def load_features(file_name):
    behavior_array = np.load(file_name)
    return behavior_array

# get feature columns for each array
def get_feature_column(feature_array, feature_column):
    
    features = np.zeros([len(feature_array), len(feature_array[0])])
    for i in range(len(feature_array)):
        for j in range(len(feature_array[i])):
                features[i][j] = feature_array[i][j][feature_column]
    return features

# get the number of peaks in the value
def get_max_min(angles, output):
     max = angles[0]
     min = angles[0]
     for index, angle in enumerate(angles):
          if (index - 1) > 0 and (index + 1) < len(angles):
               if angle > max:
                    max = angle
               if angle < min:
                    min = angle
     output.append(max)
     output.append(max - min)
     return output

# get the number of peaks in the value
def get_num_peaks(angles, output):
     num_positive_peaks = 0
     num_negative_peaks = 0
     for index, angle in enumerate(angles):
          if (index - 1) > 0 and (index + 1) < len(angles):
               if angles[index - 1] < angle and angles[index + 1] < angle:
                    num_positive_peaks += 1
               if angles[index - 1] > angle and angles[index + 1] > angle:
                    num_negative_peaks += 1
     output.append(num_positive_peaks)
     output.append(num_negative_peaks)
     return output

def construct_features(angles, output, feature_functions):
     for feature_func in feature_functions:
          feature_func(angles, output)
     return np.array(output)


# global variables
functions = [get_max_min, get_num_peaks] 

# main function
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    behavior_array = load_features("../data/behaviors.npy")

    # load in all features
    feature_0 = get_feature_column(behavior_array, 0)
    feature_1 = get_feature_column(behavior_array, 1)
    feature_2 = get_feature_column(behavior_array, 2)
    len_features = np.sum([len(f(feature_0[0], [])) for f in functions])

    output_feature = np.zeros([len(feature_0), 3 * len_features])
    for example_num in range(len(feature_0)):
         feature_i = []
         for feature in range(3):
            output = []
            if feature == Feature.PAN.value:
                 feature_i.append(construct_features(feature_0[example_num], output, functions))
            elif feature == Feature.TILT.value:
                 feature_i.append(construct_features(feature_1[example_num], output, functions))
            elif feature == Feature.EYES.value:
                 feature_i.append(construct_features(feature_2[example_num], output, functions))
         temp = [value for i in feature_i for value in i]
         print(temp)
         output_feature[example_num] = np.array(temp)
    np.save("../features.npy", output_feature)

         