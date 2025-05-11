import cv2
import numpy as np
import os
import mediapipe as mp





no_sequences = 50

# Videos are going to be 30 frames in length
sequence_length = 30


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

holistic=mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


 
# Get the list of all files and directories
path = "MP_Data"
dir_list = os.listdir(path)
 


actions = np.array(dir_list)
DATA_PATH = os.path.join('MP_Data') 

print(actions)