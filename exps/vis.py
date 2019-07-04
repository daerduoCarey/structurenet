"""
    This file contains utility functions for jupyter notebook visualization.
    Please use jupyter notebook to open vis.ipynb for results visualization.
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rand_cmap import rand_cmap
cmap = rand_cmap(300, type='bright', first_color_black=True, last_color_black=False, verbose=False)

def draw_box(ax, p, color, rot=None):
    center = p[0: 3]
    lengths = p[3: 6]
    dir_1 = p[6: 9]
    dir_2 = p[9:]

    if rot is not None:
        center = (rot * center.reshape(-1, 1)).reshape(-1)
        dir_1 = (rot * dir_1.reshape(-1, 1)).reshape(-1)
        dir_2 = (rot * dir_2.reshape(-1, 1)).reshape(-1)

    dir_1 = dir_1/np.linalg.norm(dir_1)
    dir_2 = dir_2/np.linalg.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3/np.linalg.norm(dir_3)
    cornerpoints = np.zeros([8, 3])

    d1 = 0.5*lengths[0]*dir_1
    d2 = 0.5*lengths[1]*dir_2
    d3 = 0.5*lengths[2]*dir_3

    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3

    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]],
            [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]],
            [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]],
            [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]],
            [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]],
            [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]],
            [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]],
            [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]],
            [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]],
            [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]],
            [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]],
            [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]],
            [cornerpoints[3][2], cornerpoints[7][2]], c=color)

def draw_edge(ax, e, box1, box2, rot=None):
    center1 = box1[:3]
    center2 = box2[:3]

    if rot is not None:
        center1 = (rot * center1.reshape(-1, 1)).reshape(-1)[0]
        center2 = (rot * center2.reshape(-1, 1)).reshape(-1)[0]

    edge_type_colors = {
        'ADJ': (1, 0, 0),
        'ROT_SYM': (1, 1, 0),
        'TRANS_SYM': (1, 0, 1),
        'REF_SYM': (0, 0, 0)}

    edge_type_linewidth = {
        'ADJ': 8,
        'ROT_SYM': 6,
        'TRANS_SYM': 4,
        'REF_SYM': 2}

    ax.plot(
        [center1[0, 0], center2[0, 0]], [center1[0, 1], center2[0, 1]], [center1[0, 2], center2[0, 2]],
        c=edge_type_colors[e['type']],
        linestyle=':',
        linewidth=edge_type_linewidth[e['type']])

def draw_partnet_objects(objects, object_names=None, filename=None, figsize=None, leafs_only=False, use_id_as_color=False, visu_edges=True):
    if figsize is not None:
        fig = plt.figure(0, figsize=figsize)
    else:
        fig = plt.figure(0)
    extent = 0.7
    for i, obj in enumerate(objects):

        boxes, edges, box_ids = obj.graph(leafs_only=leafs_only)
        # import pdb; pdb.set_trace()

        ax = fig.add_subplot(1, len(objects), i+1, projection='3d')
        ax.set_xlim(-extent, extent)
        ax.set_ylim(extent, -extent)
        ax.set_zlim(-extent, extent)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_aspect('equal')
        ax.set_proj_type('persp')

        if object_names is not None:
            ax.set_title(object_names[i])

        # transform coordinates so z is up (from y up)
        coord_rot = np.matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        edge_type_order = {
            'ADJ': 1,
            'ROT_SYM': 2,
            'TRANS_SYM': 3,
            'REF_SYM': 4}

        # sort edges by type, so first ones get drawn first
        edges = sorted(edges, key=lambda edge: edge_type_order[edge['type']])

        for jj in range(len(boxes)):
            if boxes[jj] is not None:
                color_id = box_ids[jj]
                if use_id_as_color:
                    color_id = jj
                draw_box(
                    ax=ax, p=boxes[jj].cpu().numpy().reshape(-1),
                    color=cmap(color_id), rot=coord_rot)
        
        if visu_edges:
            for jj in range(len(edges)):
                draw_edge(
                    ax=ax, e=edges[jj],
                    box1=boxes[edges[jj]['part_a']].cpu().numpy().reshape(-1),
                    box2=boxes[edges[jj]['part_b']].cpu().numpy().reshape(-1),
                    rot=coord_rot)

    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

