"""
    This file performs online symmetry computation for edges while training the networks.
"""

import numpy as np
import torch
from pyquaternion import Quaternion
from utils import load_pts, export_ply_with_label, transform_pc
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import scipy.spatial

''' translation symmetry
        Output: ret: T/F, trans: translation vector xyz
        Usage: part_B_point = part_A_point + (x, y, z)
'''
unit_cube = torch.from_numpy(load_pts('cube.pts'))
def compute_trans_sym(obb1, obb2):
    mat1to2 = np.eye(4)[:3, :]
    mat1to2[:, 3] = obb2[:3] - obb1[:3]
    mat2to1 = np.eye(4)[:3, :]
    mat2to1[:, 3] = obb1[:3] - obb2[:3]
    return mat1to2, mat2to1

''' Reflective Symmetry 
        Output: Ret: T/F, mid_pt: middle point xyz, direction: direction unit vector
        Usage: part_B_point = part_A_point + <mid_pt - part_A_point, direction> * 2 * direction
'''
def compute_ref_sym(obb1, obb2):
    m = (obb1[:3] + obb2[:3]) / 2
    t = obb2[:3] - obb1[:3]
    d = t / (np.linalg.norm(t) + 1e-10)
    mat = np.array([\
            [1-2*d[0]*d[0], -2*d[0]*d[1], -2*d[0]*d[2], 2*d[0]*d[0]*m[0]+2*d[0]*d[1]*m[1]+2*d[0]*d[2]*m[2]], \
            [-2*d[1]*d[0], 1-2*d[1]*d[1], -2*d[1]*d[2], 2*d[1]*d[0]*m[0]+2*d[1]*d[1]*m[1]+2*d[1]*d[2]*m[2]], \
            [-2*d[2]*d[0], -2*d[2]*d[1], 1-2*d[2]*d[2], 2*d[2]*d[0]*m[0]+2*d[2]*d[1]*m[1]+2*d[2]*d[2]*m[2]], \
        ], dtype=np.float32)
    return mat, mat

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

def get_pc_scale(pc):
    return np.sqrt(np.max(np.sum((pc - np.mean(pc, axis=0))**2, axis=1)))

def get_chamfer_distance(pc1, pc2):
    dist = cdist(pc1, pc2)
    error = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
    scale = get_pc_scale(pc1) + get_pc_scale(pc2)
    return error / scale

def atob_rot_sym(pc, pt, nor, angle):
    s = np.sin(angle); c = np.cos(angle); nx = nor[0]; ny = nor[1]; nz = nor[2];
    rotmat = np.array([[c + (1 - c) * nx * nx, (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny], \
                       [(1 - c) * nx * ny + s * nz, c + (1 - c) * ny * ny, (1 - c) * ny * nz - s * nx], \
                       [(1 - c) * nx * nz - s * ny, (1 - c) * ny * nz + s * nx, c + (1 - c) * nz * nz]], dtype=np.float32)
    return np.matmul(rotmat, (pc - pt).T).T + pt
 
def compute_rot_sym(obb1, obb2):
    pc1_center = obb1[:3]
    pc2_center = obb2[:3]
    rotmat1 = Quaternion(obb1[6], obb1[7], obb1[8], obb1[9]).rotation_matrix
    rotmat2 = Quaternion(obb2[6], obb2[7], obb2[8], obb2[9]).rotation_matrix
    pc1 = transform_pc(unit_cube, torch.tensor(obb1, dtype=torch.float32)).numpy()
    pc2 = transform_pc(unit_cube, torch.tensor(obb2, dtype=torch.float32)).numpy()
        
    min_error = 1e8; min_pt = None; min_nor = None; min_angle = None;
    for axe_id in range(3):
        pc1_axis1 = rotmat1[axe_id]
        pc1_axis2 = rotmat1[(axe_id+1)%3]
        pc2_axis1 = rotmat2[axe_id]
        pc2_axis2 = rotmat2[(axe_id+1)%3]

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

    s = np.sin(min_angle); c = np.cos(min_angle); nx = min_nor[0]; ny = min_nor[1]; nz = min_nor[2];
    rotmat = np.array([[c + (1 - c) * nx * nx, (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny], \
                       [(1 - c) * nx * ny + s * nz, c + (1 - c) * ny * ny, (1 - c) * ny * nz - s * nx], \
                       [(1 - c) * nx * nz - s * ny, (1 - c) * ny * nz + s * nx, c + (1 - c) * nz * nz]], dtype=np.float32)
    mat1to2 = np.zeros((3, 4), dtype=np.float32)
    mat1to2[:3, :3] = rotmat
    mat1to2[:, 3] = np.dot(np.eye(3)-rotmat, min_pt)
    
    s = -s;
    rotmat = np.array([[c + (1 - c) * nx * nx, (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny], \
                       [(1 - c) * nx * ny + s * nz, c + (1 - c) * ny * ny, (1 - c) * ny * nz - s * nx], \
                       [(1 - c) * nx * nz - s * ny, (1 - c) * ny * nz + s * nx, c + (1 - c) * nz * nz]], dtype=np.float32)
    mat2to1 = np.zeros((3, 4), dtype=np.float32)
    mat2to1[:3, :3] = rotmat
    mat2to1[:, 3] = np.dot(np.eye(3)-rotmat, min_pt)

    return mat1to2, mat2to1

def compute_obb(points):
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    center = center[0, :]

    pca = PCA()
    pca.fit(points)
    pcomps = pca.components_

    points_local = np.matmul(pcomps, points.transpose()).transpose()

    size = points_local.max(axis=0) - points_local.min(axis=0)

    xdir = pcomps[0, :]
    xdir /= np.linalg.norm(xdir)
    ydir = pcomps[1, :]
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)
            
    rotmat = np.vstack([xdir, ydir, zdir]).T
    q = Quaternion(matrix=rotmat)
    quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

    return np.hstack([center, size, quat]).astype(np.float32)
