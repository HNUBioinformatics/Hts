import numpy as np
from scipy.stats import pearsonr
import sys
sys.path.append('')

from FIE1 import *

def att_sim(interaction_matrix):
    X = ATT_learner(2,np.array(interaction_matrix).shape[1],30,'cosine',6,0,'relu')
    X.forward = (interaction_matrix)
    result = X.forward
    #result = np.array(result1).reshape(87,87)
    return result
