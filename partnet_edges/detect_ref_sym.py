import numpy as np
from utils import get_pc_center, get_chamfer_distance

''' Reflective Symmetry 
        Output: Ret: T/F, mid_pt: middle point xyz, direction: direction unit vector
        Usage: part_B_point = part_A_point + <mid_pt - part_A_point, direction> * 2 * direction
'''
def compute_ref_sym(pc1, pc2):
    pc1_center = get_pc_center(pc1)
    pc2_center = get_pc_center(pc2)
    mid_pt = (pc1_center + pc2_center) / 2
    trans = pc2_center - pc1_center
    direction = trans / np.linalg.norm(trans)
    new_pc1 = atob_ref_sym(pc1, mid_pt, direction)
    error = get_chamfer_distance(new_pc1, pc2)
    return error, mid_pt, direction

def atob_ref_sym(pc, mid_pt, direction):
    return np.tile(np.expand_dims(np.matmul((mid_pt - pc), direction) * 2, axis=-1), [1, 3]) * \
            np.tile(np.expand_dims(direction, axis=0), [pc.shape[0], 1]) + pc
    
