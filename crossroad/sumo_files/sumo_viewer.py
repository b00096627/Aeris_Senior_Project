import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("crossroad_road/meshes/road.obj")
mesh.compute_vertex_normals()

# Parse OBJ manually to assign per-triangle colors
verts_np = np.asarray(mesh.vertices)
tris_np  = np.asarray(mesh.triangles)
colors   = np.full((len(verts_np), 3), [0.15, 0.15, 0.15])  # default asphalt

mat_color = {"asphalt": [0.15,0.15,0.15], "white_mark": [1,1,1], "yellow_mark": [1,0.85,0]}
current_color = mat_color["asphalt"]
vert_colors = colors.copy()

with open("crossroad_road/meshes/road.obj") as f:
    vi = 0
    tri_i = 0
    face_mat = {}
    cur_mat = "asphalt"
    face_idx = 0
    for line in f:
        if line.startswith("usemtl"):
            cur_mat = line.split()[1]
        elif line.startswith("f "):
            face_mat[face_idx] = cur_mat
            face_idx += 1

tri_colors = np.zeros((len(tris_np), 3))
for i, tri in enumerate(tris_np):
    c = mat_color.get(face_mat.get(i, "asphalt"), [0.15,0.15,0.15])
    tri_colors[i] = c

# Paint vertices by their first-encountered triangle color
vc = np.full((len(verts_np), 3), [0.15,0.15,0.15])
for i, tri in enumerate(tris_np):
    for v in tri:
        vc[v] = tri_colors[i]

mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
o3d.visualization.draw_geometries([mesh])