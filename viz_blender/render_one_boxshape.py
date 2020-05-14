import os
import sys
import numpy as np
import bpy
sys.path.append(os.getcwd()) # for some reason the working directory is not in path
from vis_utils import load_object, hex2rgb, read_textfile_list, load_semantic_colors

colors_filename = 'semantic_colors.txt'
leafs_only = True

def create_box(box_params):
    center = box_params[0: 3]
    lengths = box_params[3: 6]
    dir_1 = box_params[6: 9]
    dir_2 = box_params[9:]
    #
    dir_1 = dir_1/np.linalg.norm(dir_1)
    dir_2 = dir_2/np.linalg.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3/np.linalg.norm(dir_3)
    #
    d1 = 0.5*lengths[0]*dir_1
    d2 = 0.5*lengths[1]*dir_2
    d3 = 0.5*lengths[2]*dir_3
    #
    verts = np.zeros([8, 3])
    verts[0, :] = center - d1 - d2 - d3
    verts[1, :] = center - d1 + d2 - d3
    verts[2, :] = center + d1 - d2 - d3
    verts[3, :] = center + d1 + d2 - d3
    verts[4, :] = center - d1 - d2 + d3
    verts[5, :] = center - d1 + d2 + d3
    verts[6, :] = center + d1 - d2 + d3
    verts[7, :] = center + d1 + d2 + d3
    #
    faces = np.zeros([6, 4], dtype='int64')
    faces[0, :] = [3, 2, 0, 1]
    faces[1, :] = [4, 6, 7, 5]
    faces[2, :] = [0, 2, 6, 4]
    faces[3, :] = [3, 1, 5, 7]
    faces[4, :] = [2, 3, 7, 6]
    faces[5, :] = [1, 0, 4, 5]
    #
    return verts, faces

colors = load_semantic_colors(filename=colors_filename)

obj_filename = sys.argv[6]
output_filename = sys.argv[7]

if 'object' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['object'], do_unlink=True)

if 'wireframe' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['wireframe'], do_unlink=True)

if 'object' in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes['object'], do_unlink=True)

if 'wireframe' in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes['wireframe'], do_unlink=True)

obj = load_object(obj_filename)

boxes = obj.boxes(leafs_only=leafs_only)

verts = []
faces = []
vert_colors = []
coord_rot = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
for node in obj.depth_first_traversal():
    if node.is_leaf:
        box = node.box
        box[:3] = np.matmul(coord_rot, box[:3])
        box[6:9] = np.matmul(coord_rot, box[6:9])
        box[9:] = np.matmul(coord_rot, box[9:])
        box_verts, box_faces = create_box(box_params=box)
        faces.append(box_faces+len(verts)*box_verts.shape[0])
        verts.append(box_verts)
        box_color = colors[node.full_label]
        vert_colors.append(np.tile(np.array(box_color).astype('float32')/255.0, [box_verts.shape[0], 1]))

verts = np.concatenate(verts, axis=0)
faces = np.concatenate(faces, axis=0)
vert_colors = np.concatenate(vert_colors, axis=0)
vert_colors = vert_colors[faces.reshape(-1), :]

verts[:, 2] -= verts.min(axis=0)[2]

verts = verts.tolist()
faces = faces.tolist()
vert_colors = vert_colors.tolist()

scene = bpy.context.scene
mesh = bpy.data.meshes.new('object')
mesh.from_pydata(verts, [], faces)
mesh.validate()

mesh.vertex_colors.new(name='Col') # named 'Col' by default
mesh_vert_colors = mesh.vertex_colors['Col']

wireframe_mesh = mesh.copy()
wireframe_mesh.name = 'wireframe'

for i, c in enumerate(mesh.vertex_colors['Col'].data):
    c.color = vert_colors[i]

obj = bpy.data.objects.new('object', mesh)
obj.data.materials.append(bpy.data.materials['sphere_material'])
scene.objects.link(obj)

obj_wireframe = bpy.data.objects.new('wireframe', wireframe_mesh)
obj_wireframe.data.materials.append(bpy.data.materials['wireframe_material'])
scene.objects.link(obj_wireframe)

bpy.context.scene.objects.active = obj_wireframe

mod = obj_wireframe.modifiers.new(name='wireframe', type='WIREFRAME')
mod.thickness = 0.015

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = output_filename
bpy.ops.render.render(write_still=True)
