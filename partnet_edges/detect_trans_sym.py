import numpy as np
from utils import get_pc_center, get_chamfer_distance

''' translation symmetry
        Output: ret: T/F, trans: translation vector xyz
        Usage: part_B_point = part_A_point + (x, y, z)
'''
def compute_trans_sym(pc1, pc2):
    pc1_center = get_pc_center(pc1)
    pc2_center = get_pc_center(pc2)
    trans = pc2_center - pc1_center
    new_pc1 = atob_trans_sym(pc1, trans)
    error = get_chamfer_distance(new_pc1, pc2)
    return error, trans

def atob_trans_sym(pc, trans):
    return pc + trans

