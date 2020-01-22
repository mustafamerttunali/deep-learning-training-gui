# # # # # # # # # # # # # # # 
# Author: Mustafa Mert Tunalı
# ---------------------------
# ---------------------------
# Deep Learning Training GUI - Config Page 
# ---------------------------
# ---------------------------
# # # # # # # # # # # # # # # 

import numpy as np

# In Development For Object Detection

class Config(object):
    
    GPU_COUNT = 1
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 100
    NUM_CLASSES = 1 
    DETECTION_NMS_THRESHOLD = 0.3
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001


