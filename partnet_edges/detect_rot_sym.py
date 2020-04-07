import numpy as np
from utils import get_pc_center, get_pca_axes, get_chamfer_distance

''' rotational symmetry
        Output: ret: T/F, pt: pivot point xyz, nor: normal unit vector, angle: rotation radian angle
        Usage: part_B_point = Rot(part_A_point - pt, nor, angel) + pt
'''
def compute_params(pc1_center, pc2_center, pc1_v1, pc1_v2, pc2_v1, pc2_v2):
    mid_v1 = (pc1_v1 + pc2_v1) / 2
    nor_v1 = pc1_v1 - pc2_v1
    nor_v1_len = np.linalg.norm(nor_v1)
    if nor_v1_len < 1e-6:
        return np.zeros((3), dtype=np.float32), np.zeros((3), dtype=np.float32), 0.0
    nor_v1 /= nor_v1_len
    
    mid_v2 = (pc1_v2 + pc2_v2) / 2
    nor_v2 = pc1_v2 - pc2_v2
    nor_v2_len = np.linalg.norm(nor_v2)
    if nor_v2_len < 1e-6:
        return np.zeros((3), dtype=np.float32), np.zeros((3), dtype=np.float32), 0.0
    nor_v2 /= nor_v2_len

    # compute the axis direction
    nor = np.cross(nor_v1, nor_v2)
    nor_len = np.linalg.norm(nor)
    if nor_len < 1e-6:
        return np.zeros((3), dtype=np.float32), np.zeros((3), dtype=np.float32), 0.0
    nor /= nor_len

    # compute one pivot point (any point along the axis is good)
    A = np.array([[nor_v1[0], nor_v1[1], nor_v1[2]], \
                  [nor_v2[0], nor_v2[1], nor_v2[2]], \
                  [nor[0], nor[1], nor[2]]], dtype=np.float32)
    b = np.array([np.dot(nor_v1, mid_v1), np.dot(nor_v2, mid_v2), np.dot(nor, mid_v1)])
    pt = np.matmul(np.linalg.inv(A), b)

    # compute rotation angle
    tv1 = pc1_center - pt - nor * np.dot(pc1_center - pt, nor)
    tv2 = pc2_center - pt - nor * np.dot(pc2_center - pt, nor)
    c = np.dot(tv1, tv2) / (np.linalg.norm(tv1) * np.linalg.norm(tv2))
    c = np.clip(c, -1.0, 1.0)
    angle = np.arccos(c)

    return pt, nor, angle

def compute_rot_sym(pc1, pc2):
    pc1_center = get_pc_center(pc1)
    pc2_center = get_pc_center(pc2)
    pc1_axes = get_pca_axes(pc1)
    pc2_axes = get_pca_axes(pc2)

    min_error = 1e8; min_pt = None; min_nor = None; min_angle = None;
    for axe_id in range(3):
        pc1_axis1 = pc1_axes[axe_id]
        pc1_axis2 = pc1_axes[(axe_id+1)%3]
        pc2_axis1 = pc2_axes[axe_id]
        pc2_axis2 = pc2_axes[(axe_id+1)%3]

        pt, nor, angle = compute_params(pc1_center, pc2_center, pc1_center + pc1_axis1, pc1_center + pc1_axis2, pc2_center + pc2_axis1, pc2_center + pc2_axis2)
        new_pc1 = atob_rot_sym(pc1, pt, nor, angle)
        error = get_chamfer_distance(new_pc1, pc2)
        if error < min_error:
            min_error = error; min_pt = pt; min_nor = nor; min_angle = angle;

        pt, nor, angle = compute_params(pc1_center, pc2_center, pc1_center + pc1_axis1, pc1_center + pc1_axis2, pc2_center - pc2_axis1, pc2_center + pc2_axis2)
        new_pc1 = atob_rot_sym(pc1, pt, nor, angle)
        error = get_chamfer_distance(new_pc1, pc2)
        if error < min_error:
            min_error = error; min_pt = pt; min_nor = nor; min_angle = angle;

        pt, nor, angle = compute_params(pc1_center, pc2_center, pc1_center + pc1_axis1, pc1_center + pc1_axis2, pc2_center + pc2_axis1, pc2_center - pc2_axis2)
        new_pc1 = atob_rot_sym(pc1, pt, nor, angle)
        error = get_chamfer_distance(new_pc1, pc2)
        if error < min_error:
            min_error = error; min_pt = pt; min_nor = nor; min_angle = angle;

        pt, nor, angle = compute_params(pc1_center, pc2_center, pc1_center + pc1_axis1, pc1_center + pc1_axis2, pc2_center - pc2_axis1, pc2_center - pc2_axis2)
        new_pc1 = atob_rot_sym(pc1, pt, nor, angle)
        error = get_chamfer_distance(new_pc1, pc2)
        if error < min_error:
            min_error = error; min_pt = pt; min_nor = nor; min_angle = angle;

    return min_error, min_pt, min_nor, min_angle

def atob_rot_sym(pc, pt, nor, angle):
    s = np.sin(angle); c = np.cos(angle); nx = nor[0]; ny = nor[1]; nz = nor[2];
    rotmat = np.array([[c + (1 - c) * nx * nx, (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny], \
                       [(1 - c) * nx * ny + s * nz, c + (1 - c) * ny * ny, (1 - c) * ny * nz - s * nx], \
                       [(1 - c) * nx * nz - s * ny, (1 - c) * ny * nz + s * nx, c + (1 - c) * nz * nz]], dtype=np.float32)
    return np.matmul(rotmat, (pc - pt).T).T + pt
 
