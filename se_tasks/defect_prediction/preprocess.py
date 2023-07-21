
import os
import numpy as np


for dict_map in os.listdir('./dataset/mapped data'):
    dict_map = np.load('./dataset/mapped dataset/' + dict_map)
    print()