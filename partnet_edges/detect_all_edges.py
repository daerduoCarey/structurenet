import os
import sys
import time
import random
import json
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from geometry_utils import load_obj, sample_points
from commons import check_mkdir, force_mkdir
from progressbar import ProgressBar
from subprocess import call
from PIL import Image
import scipy.misc as misc

from detect_adj import compute_adj
from detect_ref_sym import compute_ref_sym, atob_ref_sym
from detect_trans_sym import compute_trans_sym, atob_trans_sym
from detect_rot_sym import compute_rot_sym, atob_rot_sym

anno_id = sys.argv[1]
in_dir = os.path.join('../../data', anno_id)
render_dir = os.path.join(in_dir, 'parts_render_after_merging')

out_dir = os.path.join('results', anno_id)
check_mkdir(out_dir)
visu_dir = os.path.join(out_dir, 'visu')
force_mkdir(visu_dir)

parent_dir = os.path.join(visu_dir, 'parent')
os.mkdir(parent_dir)
info_dir = os.path.join(visu_dir, 'info')
os.mkdir(info_dir)
child_dir = os.path.join(visu_dir, 'child')
os.mkdir(child_dir)

json_fn = os.path.join(in_dir, 'result_after_merging.json')
with open(json_fn, 'r') as fin:
    data = json.load(fin)[0]

found_edges = dict()

def get_mesh(objs):
    v = []; f = []; vid = 0;
    for item in objs:
        mesh = load_obj(os.path.join(in_dir, 'objs', item+'.obj'))
        v.append(mesh['vertices'])
        f.append(mesh['faces']+vid)
        vid += mesh['vertices'].shape[0]
    v = np.vstack(v)
    f = np.vstack(f)
    return v, f

def get_pc(v, f, num_points=1000):
    pc, _ = sample_points(v, f, num_points=num_points)
    return pc

def export_obj(out, v, f, color):
    mtl_out = out.replace('.obj', '.mtl')

    with open(out, 'w') as fout:
        fout.write('mtllib %s\n' % mtl_out)
        fout.write('usemtl m1\n')
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    with open(mtl_out, 'w') as fout:
        fout.write('newmtl m1\n')
        fout.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout.write('Ka 0 0 0\n')

    return mtl_out

def render_mesh(vv, f, color=[0.216, 0.494, 0.722]):
    v = vv * 0.55

    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    tmp_obj = os.path.join(tmp_dir, str(time.time()).replace('.', '_')+'_'+str(random.random()).replace('.', '_')+'.obj')
    tmp_png = tmp_obj.replace('.obj', '.png')

    tmp_mtl = export_obj(tmp_obj, v, f, color=color)

    cmd = 'bash render.sh %s %s' % (tmp_obj, tmp_png)
    call(cmd, shell=True)

    img = misc.imread(tmp_png)
    img = img[1:401, 350: 750, :]
    img = img.astype(np.float32)

    all_white = np.ones((img.shape), dtype=np.float32) * 255

    img_alpha = img[:, :, 3] * 1.0 / 256
    all_white_alpha = 1.0 - img_alpha

    all_white[:, :, 0] *= all_white_alpha
    all_white[:, :, 1] *= all_white_alpha
    all_white[:, :, 2] *= all_white_alpha

    img[:, :, 0] *= img_alpha
    img[:, :, 1] *= img_alpha
    img[:, :, 2] *= img_alpha

    out = img[:, :, :3] + all_white[:, :, :3]

    cmd = 'rm -rf %s %s %s' % (tmp_obj, tmp_png, tmp_mtl)
    call(cmd, shell=True)

    return out

root_v, root_f = get_mesh(data['objs'])
root_render = render_mesh(root_v, root_f)


def detect_edges(record):

    if 'children' in record.keys():
        pid = record['id']

        # visu
        cmd = 'cp %s %s' % (os.path.join(render_dir, '%d.png' % pid), os.path.join(parent_dir, 'part-%03d.png' % pid))
        call(cmd, shell=True)
        out_fn = os.path.join(info_dir, 'part-%03d.txt' % pid)
        with open(out_fn, 'w') as fout:
            fout.write('parent_id: %d\n' % pid)
 
        # main
        num_children = len(record['children'])
        found_edges[pid] = []

        # get all part meshes and pc
        children_v = dict(); children_f = dict(); children_pc = dict();
        for i in range(num_children):
            id_i = record['children'][i]['id']
            v, f = get_mesh(record['children'][i]['objs'])
            children_v[id_i] = v; children_f[id_i] = f;
            children_pc[id_i] = get_pc(v, f)

        # detect pairwise adjancency
        for i in range(1, num_children):
            for j in range(i):
                id_i = record['children'][i]['id']
                id_j = record['children'][j]['id']
                min_dist = compute_adj(children_pc[id_i], children_pc[id_j])
                if min_dist < 0.05:
                    found_edges[pid].append({'type': 'adj', 'part_a': id_i, 'part_b': id_j, 'min_dist': min_dist})

        # detect symmetry within each semantics
        children_per_sem = dict()
        for item in record['children']:
            sem_name = item['name']
            if sem_name not in children_per_sem:
                children_per_sem[sem_name] = []
            children_per_sem[sem_name].append({'id': item['id'], 'objs': item['objs']})

        for sem_name in children_per_sem.keys():
            cur_children_subset = children_per_sem[sem_name]
            num_children_subset = len(cur_children_subset)
            for i in range(1, num_children_subset):
                for j in range(i):
                    id_i = cur_children_subset[i]['id']
                    id_j = cur_children_subset[j]['id']

                    error, mid_pt, direction = compute_ref_sym(children_pc[id_i], children_pc[id_j])
                    if error < 0.05:
                        found_edges[pid].append({'type': 'ref_sym', 'part_a': id_i, 'part_b': id_j, 'mid_pt': mid_pt.tolist(), 'dir': direction.tolist(), 'error': error})

                    error, trans = compute_trans_sym(children_pc[id_i], children_pc[id_j])
                    if error < 0.05:
                        found_edges[pid].append({'type': 'trans_sym', 'part_a': id_i, 'part_b': id_j, 'trans': trans.tolist(), 'error': error})
            
                    error, pt, nor, angle = compute_rot_sym(children_pc[id_i], children_pc[id_j])
                    if error < 0.05:
                        found_edges[pid].append({'type': 'rot_sym', 'part_a': id_i, 'part_b': id_j, 'pt': pt.tolist(), 'nor': nor.tolist(), 'angle': float(angle), 'error': error})

        # visu
        cur_child_dir = os.path.join(child_dir, 'part-%03d' % pid)
        os.mkdir(cur_child_dir)
        cur_part_a_dir = os.path.join(cur_child_dir, 'part_a')
        os.mkdir(cur_part_a_dir)
        cur_part_b_dir = os.path.join(cur_child_dir, 'part_b')
        os.mkdir(cur_part_b_dir)
        cur_atob_dir = os.path.join(cur_child_dir, 'atob')
        os.mkdir(cur_atob_dir)
        cur_info_dir = os.path.join(cur_child_dir, 'info')
        os.mkdir(cur_info_dir)

        for rid in range(len(found_edges[pid])):
            cur_rel = found_edges[pid][rid]
            cmd = 'cp %s %s' % (os.path.join(render_dir, '%d.png' % cur_rel['part_a']), os.path.join(cur_part_a_dir, 'rel-%03d.png' % rid))
            call(cmd, shell=True)
            cmd = 'cp %s %s' % (os.path.join(render_dir, '%d.png' % cur_rel['part_b']), os.path.join(cur_part_b_dir, 'rel-%03d.png' % rid))
            call(cmd, shell=True)
            fout = open(os.path.join(cur_info_dir, 'rel-%03d.txt' % rid), 'w')
            fout.write('type: %s\n' % cur_rel['type'])
            fout.write('part_a: %d\n' % cur_rel['part_a'])
            fout.write('part_b: %d\n' % cur_rel['part_b'])
            transformed_part_a_v = None
            if cur_rel['type'] == 'adj':
                fout.write('min_dist: %f\n' % cur_rel['min_dist'])
            elif cur_rel['type'] == 'ref_sym':
                transformed_part_a_v = atob_ref_sym(children_v[cur_rel['part_a']], cur_rel['mid_pt'], cur_rel['dir'])
                fout.write('mid_pt: %s\n' % str(cur_rel['mid_pt']))
                fout.write('dir: %s\n' % str(cur_rel['dir']))
                fout.write('error: %f\n' % cur_rel['error'])
            elif cur_rel['type'] == 'trans_sym':
                transformed_part_a_v = atob_trans_sym(children_v[cur_rel['part_a']], cur_rel['trans'])
                fout.write('trans: %s\n' % str(cur_rel['trans']))
                fout.write('error: %f\n' % cur_rel['error'])
            elif cur_rel['type'] == 'rot_sym':
                transformed_part_a_v = atob_rot_sym(children_v[cur_rel['part_a']], cur_rel['pt'], cur_rel['nor'], cur_rel['angle'])
                fout.write('pt: %s\n' % str(cur_rel['pt']))
                fout.write('nor: %s\n' % str(cur_rel['nor']))
                fout.write('angle: %f\n' % cur_rel['angle'])
                fout.write('error: %f\n' % cur_rel['error'])
            if transformed_part_a_v is not None:
                part_render = render_mesh(transformed_part_a_v, children_f[cur_rel['part_a']], color=[0.93, 0.0, 0.0])
                part_render = 0.3 * root_render + 0.7 * part_render
                Image.fromarray(part_render.astype(np.uint8)).save(os.path.join(cur_atob_dir, 'rel-%03d.png' % rid))
            fout.close()

        # run for all children
        for item in record['children']:
            detect_edges(item)

# main
detect_edges(data)

# export json
with open(os.path.join(out_dir, 'edges.json'), 'w') as fout:
    json.dump(found_edges, fout)

# visu
cmd = 'cd %s && python2.7 %s . 1 htmls parent,info:part_a,part_b,atob,info parent,info:part_a,part_b,atob,info >> /dev/null' % (visu_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierarchy_local.py'))
call(cmd, shell=True)

