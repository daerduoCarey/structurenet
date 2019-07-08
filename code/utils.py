"""
    This file contains all helper utility functions.
"""

import os
import sys
import math
import importlib
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import trimesh

def save_checkpoint(models, model_names, dirname, epoch=None, prepend_epoch=False, optimizers=None, optimizer_names=None):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        torch.save(model.state_dict(), os.path.join(dirname, filename))

    if optimizers is not None:
        filename = 'checkpt.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        checkpt = {'epoch': epoch}
        for opt, optimizer_name in zip(optimizers, optimizer_names):
            checkpt[f'opt_{optimizer_name}'] = opt.state_dict()
        torch.save(checkpt, os.path.join(dirname, filename))

def load_checkpoint(models, model_names, dirname, epoch=None, optimizers=None, optimizer_names=None, strict=True):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if epoch is not None:
            filename = f'{epoch}_' + filename
        model.load_state_dict(torch.load(os.path.join(dirname, filename)), strict=strict)

    start_epoch = 0
    if optimizers is not None:
        filename = os.path.join(dirname, 'checkpt.pth')
        if epoch is not None:
            filename = f'{epoch}_' + filename
        if os.path.exists(filename):
            checkpt = torch.load(filename)
            start_epoch = checkpt['epoch']
            for opt, optimizer_name in zip(optimizers, optimizer_names):
                opt.load_state_dict(checkpt[f'opt_{optimizer_name}'])
            print(f'resuming from checkpoint {filename}')
        else:
            response = input(f'Checkpoint {filename} not found for resuming, refine saved models instead? (y/n) ')
            if response != 'y':
                sys.exit()

    return start_epoch

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = rotvector.new_tensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]])
    return m

def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module(model_version)

# row_counts, col_counts: row and column counts of each distance matrix (assumed to be full if given)
def linear_assignment(distance_mat, row_counts=None, col_counts=None):
    batch_ind = []
    row_ind = []
    col_ind = []
    for i in range(distance_mat.shape[0]):
        # print(f'{i} / {distance_mat.shape[0]}')

        dmat = distance_mat[i, :, :]
        if row_counts is not None:
            dmat = dmat[:row_counts[i], :]
        if col_counts is not None:
            dmat = dmat[:, :col_counts[i]]

        rind, cind = linear_sum_assignment(dmat.to('cpu').numpy())
        rind = list(rind)
        cind = list(cind)

        if len(rind) > 0:
            rind, cind = zip(*sorted(zip(rind, cind)))
            rind = list(rind)
            cind = list(cind)

        # complete the assignemnt for any remaining non-active elements (in case row_count or col_count was given),
        # by assigning them randomly
        #if len(rind) < distance_mat.shape[1]:
        #    rind.extend(set(range(distance_mat.shape[1])).difference(rind))
        #    cind.extend(set(range(distance_mat.shape[1])).difference(cind))

        batch_ind += [i]*len(rind)
        row_ind += rind
        col_ind += cind    

    return batch_ind, row_ind, col_ind

def object_batch_boxes(objects, max_box_num):
    box_num = []
    boxes = torch.zeros(len(objects), 12, max_box_num)
    for oi, obj in enumerate(objects):
        obj_boxes = obj.boxes()
        box_num.append(len(obj_boxes))
        if box_num[-1] > max_box_num:
            print(f'WARNING: too many boxes in object, please use a dataset that does not have objects with too many boxes, clipping the object for now.')
            box_num[-1] = max_box_num
            obj_boxes = obj_boxes[:box_num[-1]]
        obj_boxes = [o.view(-1, 1) for o in obj_boxes]
        boxes[oi, :, :box_num[-1]] = torch.cat(obj_boxes, dim=1)

    return boxes, box_num

# out shape: (label_count, in shape)
def one_hot(inp, label_count):
    out = torch.zeros(label_count, inp.numel(), dtype=torch.uint8, device=inp.device)
    out[inp.view(-1), torch.arange(out.shape[1])] = 1
    out = out.view((label_count,) + inp.shape)
    return out

def collate_feats(b):
    return list(zip(*b))

def export_ply_with_label(out, v, l):
    num_colors = len(colors)
    with open(out, 'w') as fout:
        fout.write('ply\n');
        fout.write('format ascii 1.0\n');
        fout.write('element vertex '+str(v.shape[0])+'\n');
        fout.write('property float x\n');
        fout.write('property float y\n');
        fout.write('property float z\n');
        fout.write('property uchar red\n');
        fout.write('property uchar green\n');
        fout.write('property uchar blue\n');
        fout.write('end_header\n');

        for i in range(v.shape[0]):
            cur_color = colors[l[i]%num_colors]
            fout.write('%f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], \
                    int(cur_color[0]*255), int(cur_color[1]*255), int(cur_color[2]*255)))

def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        return pts

def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f

def export_obj(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

# pc is N x 3, feat is 10-dim
def transform_pc(pc, feat):
    num_point = pc.size(0)
    center = feat[:3]
    shape = feat[3:6]
    quat = feat[6:]
    pc = pc * shape.repeat(num_point, 1)
    pc = qrot(quat.repeat(num_point, 1), pc)
    pc = pc + center.repeat(num_point, 1)
    return pc

# pc is N x 3, feat is B x 10-dim
def transform_pc_batch(pc, feat, anchor=False):
    batch_size = feat.size(0)
    num_point = pc.size(0)
    pc = pc.repeat(batch_size, 1, 1)
    center = feat[:, :3].unsqueeze(dim=1).repeat(1, num_point, 1)
    shape = feat[:, 3:6].unsqueeze(dim=1).repeat(1, num_point, 1)
    quat = feat[:, 6:].unsqueeze(dim=1).repeat(1, num_point, 1)
    if not anchor:
        pc = pc * shape
    pc = qrot(quat.view(-1, 4), pc.view(-1, 3)).view(batch_size, num_point, 3)
    if not anchor:
        pc = pc + center
    return pc

def get_surface_reweighting(xyz, cube_num_point):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
    np = cube_num_point // 6
    out = torch.cat([(x*y).repeat(np*2), (y*z).repeat(np*2), (x*z).repeat(np*2)])
    out = out / (out.sum() + 1e-12)
    return out

def get_surface_reweighting_batch(xyz, cube_num_point):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
    np = cube_num_point // 6
    out = torch.cat([(x*y).unsqueeze(dim=1).repeat(1, np*2), \
                     (y*z).unsqueeze(dim=1).repeat(1, np*2), \
                     (x*z).unsqueeze(dim=1).repeat(1, np*2)], dim=1)
    out = out / (out.sum(dim=1).unsqueeze(dim=1) + 1e-12)
    return out

def gen_obb_mesh(obbs):
    # load cube
    cube_v, cube_f = load_obj('cube.obj')

    all_v = []; all_f = []; vid = 0;
    for pid in range(obbs.shape[0]):
        p = obbs[pid, :]
        center = p[0: 3]
        lengths = p[3: 6]
        dir_1 = p[6: 9]
        dir_2 = p[9: ]

        dir_1 = dir_1/np.linalg.norm(dir_1)
        dir_2 = dir_2/np.linalg.norm(dir_2)
        dir_3 = np.cross(dir_1, dir_2)
        dir_3 = dir_3/np.linalg.norm(dir_3)

        v = np.array(cube_v, dtype=np.float32)
        f = np.array(cube_f, dtype=np.int32)
        rot = np.vstack([dir_1, dir_2, dir_3])
        v *= lengths
        v = np.matmul(v, rot)
        v += center

        all_v.append(v)
        all_f.append(f+vid)
        vid += v.shape[0]

    all_v = np.vstack(all_v)
    all_f = np.vstack(all_f)
    return all_v, all_f

def sample_pc(v, f, n_points=2048):
    mesh = trimesh.Trimesh(vertices=v, faces=f-1)
    points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
    return points

