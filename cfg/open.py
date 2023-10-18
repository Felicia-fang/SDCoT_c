import os
import numpy as np
import random
from collections import defaultdict
import pickle
from easydict import EasyDict
import random
# random.seed(10)
import csv
with open("0.05scannet_scene", 'rb') as f:
    class2scans = pickle.load(f)
    print(class2scans)