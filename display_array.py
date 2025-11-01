import numpy as np

import os


ROOT_DIR = os.path.join(os.getcwd(),"spectrograms")
count=0
for emotion in os.listdir(ROOT_DIR):
    for spect in os.listdir(os.path.join(ROOT_DIR,emotion)):
        count +=1
print(count)