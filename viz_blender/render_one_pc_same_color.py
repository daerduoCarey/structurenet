import os
import sys
import numpy as np
import bpy
import bmesh
sys.path.append(os.getcwd()) # for some reason the working directory is not in path
from vis_utils import hex2rgb, read_textfile_list, load_pts

point_radius = 0.01
result_type = 'orig'

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

colors = np.array([[float(c)/255.0 for c in hex2rgb(color)] for color in colors])

pts_filename = sys.argv[6]
output_filename = sys.argv[7]

pts = load_pts(pts_filename)
# rotate for blender
coord_rot = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
pts = np.matmul(pts, coord_rot.transpose())

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

verts[:, 2] -= verts.min(axis=0)[2]

verts = verts.tolist()
faces = faces.tolist()

scene = bpy.context.scene
mesh = bpy.data.meshes.new('object')
mesh.from_pydata(verts, [], faces)
mesh.validate()

mesh.vertex_colors.new(name='Col') # named 'Col' by default

for i, c in enumerate(mesh.vertex_colors['Col'].data):
    c.color = colors[0]

obj = bpy.data.objects.new('object', mesh)
obj.data.materials.append(bpy.data.materials['sphere_material'])
scene.objects.link(obj)

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = output_filename
bpy.ops.render.render(write_still=True)
