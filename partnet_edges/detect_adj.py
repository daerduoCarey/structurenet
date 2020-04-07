import numpy as np
from scipy.spatial.distance import cdist
from utils import get_pc_scale

def compute_adj(pc1, pc2):
    pc1_scale = get_pc_scale(pc1)
    pc2_scale = get_pc_scale(pc2)
    min_d = np.min(cdist(pc1, pc2))
    min_d /= (pc1_scale + pc2_scale) / 2
    return min_d

