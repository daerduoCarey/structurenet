import os
import sys
import numpy as np
import bpy
import bmesh
sys.path.append(os.getcwd()) # for some reason the working directory is not in path
from vis_utils import Tree, hex2rgb, read_textfile_list, load_pts, load_semantic_colors, load_semantics_list, load_colors

colors_filename = 'semantic_colors.txt'
color_offsets_filename = 'instance_color_offsets.txt'
semantics_filename = 'semantics.txt'
point_radius = 0.01
result_type = 'orig'
ins_color_offset_strength = 0.3

# colors = np.array([[float(c)/255.0 for c in hex2rgb(color)] for color in colors])
semantic_colors = load_semantic_colors(filename=colors_filename)

semantics = load_semantics_list(filename=semantics_filename)

colors = np.zeros([len(semantics), 3])
for si, semantic in enumerate(semantics):
    colors[si, :] = semantic_colors[semantic]
colors = colors.astype('float32') / 255.0

color_offsets = np.array(load_colors(filename=color_offsets_filename)).astype('float32') / 255.0 - 0.5

pts_filename = sys.argv[6]
ins_lbls_filename = pts_filename.replace('.pts', '-ins.label')
sem_lbls_filename = pts_filename.replace('.pts', '-sem.label')
output_filename = sys.argv[7]

pts = load_pts(pts_filename)
# rotate for blender
coord_rot = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
pts = np.matmul(pts, coord_rot.transpose())

pt_ins_lbls = []
with open(ins_lbls_filename, 'r') as f:
    pt_ins_lbls = f.readlines()
pt_ins_lbls = [int(x.strip()) for x in pt_ins_lbls]
pt_ins_lbls = np.array(list(filter(lambda x: x is not None, pt_ins_lbls))).reshape(-1, 1)

pt_sem_lbls = []
with open(sem_lbls_filename, 'r') as f:
    pt_sem_lbls = f.readlines()
pt_sem_lbls = [int(x.strip()) for x in pt_sem_lbls]
pt_sem_lbls = np.array(list(filter(lambda x: x is not None, pt_sem_lbls))).reshape(-1, 1) - 1 # base 1 to base 0

# get instance indices that are unique only inside each semantic
pt_sem_ins_lbls = np.zeros(pt_ins_lbls.shape, dtype='int32')
used_semantics = np.unique(pt_sem_lbls)
for i in range(used_semantics.shape[0]):
    cur_sem_ins_lbls = np.unique(pt_ins_lbls[pt_sem_lbls == used_semantics[i]])
    for si, cur_sem_ins_lbl in enumerate(cur_sem_ins_lbls):
        pt_sem_ins_lbls[pt_ins_lbls == cur_sem_ins_lbl] = si

pt_colors = (colors[pt_sem_lbls[:, 0], :] + color_offsets[pt_sem_ins_lbls[:, 0], :] * ins_color_offset_strength).clip(min=0.0, max=1.0)

if 'object' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['object'], do_unlink=True)

if 'sphere' in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes['sphere'], do_unlink=True)

if 'object' in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes['object'], do_unlink=True)

sphere_mesh = bpy.data.meshes.new('sphere')
sphere_bmesh = bmesh.new()
bmesh.ops.create_icosphere(sphere_bmesh, subdivisions=2, diameter=point_radius*2)
sphere_bmesh.to_mesh(sphere_mesh)
sphere_bmesh.free()

sphere_verts = np.array([[v.co.x, v.co.y, v.co.z] for v in sphere_mesh.vertices])
sphere_faces = np.array([[p.vertices[0], p.vertices[1], p.vertices[2]] for p in sphere_mesh.polygons])

verts = (np.expand_dims(sphere_verts, axis=0) + np.expand_dims(pts, axis=1)).reshape(-1, 3)
faces = (np.expand_dims(sphere_faces, axis=0) + (np.arange(pts.shape[0]) * sphere_verts.shape[0]).reshape(-1, 1, 1)).reshape(-1, 3)
# vert_normals = np.tile(sphere.vertex_normals, [pts.shape[0], 1])
vert_colors = np.repeat(pt_colors, sphere_verts.shape[0], axis=0).astype(dtype='float64')
vert_colors = vert_colors[faces.reshape(-1), :]

verts[:, 2] -= verts.min(axis=0)[2]

print(verts.shape, faces.shape, vert_colors.shape)
verts = verts.tolist()
faces = faces.tolist()
vert_colors = vert_colors.tolist()

scene = bpy.context.scene
mesh = bpy.data.meshes.new('object')
mesh.from_pydata(verts, [], faces)
mesh.validate()

mesh.vertex_colors.new(name='Col') # named 'Col' by default
mesh_vert_colors = mesh.vertex_colors['Col']

for i, c in enumerate(mesh.vertex_colors['Col'].data):
    c.color = vert_colors[i]

obj = bpy.data.objects.new('object', mesh)
obj.data.materials.append(bpy.data.materials['sphere_material'])
scene.objects.link(obj)

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = output_filename
bpy.ops.render.render(write_still=True)
