#!/usr/bin/env python3
"""
sumo_net_to_gazebo.py
─────────────────────────────────────────────────────────────────────────────
Converts a SUMO net.xml directly into a Gazebo Harmonic model with:
  • Asphalt road surface (per lane, including internal junction lanes)
  • Solid white  → outer road edges + rightmost boundary
  • Dashed white → inner lane dividers (same direction traffic)
  • Solid yellow → leftmost edge of each edge (centre-of-road side)

Output layout (drop the whole folder into your GZ_SIM_RESOURCE_PATH):
  <output_dir>/
    model.config
    model.sdf
    meshes/
      road.obj
      road.mtl

Usage:
  pip install sumolib          # if not already in your SUMO install
  python3 sumo_net_to_gazebo.py crossroad.net.xml [output_dir]

  output_dir defaults to  ./crossroad_gazebo_model
─────────────────────────────────────────────────────────────────────────────
"""

import os, sys, math
import sumolib

# ── tunables ──────────────────────────────────────────────────────────────────
ROAD_Z    = 0.000   # asphalt surface height [m]
MARK_Z    = 0.003   # marking height above road [m]
MARK_W    = 0.12    # lane-marking strip width [m]
DASH_LEN  = 3.0     # dashed-line painted segment length [m]
DASH_GAP  = 5.0     # dashed-line gap length [m]

# Edge types to skip entirely
SKIP_TYPES = {
    "rail", "railway", "tram", "subway",
    "bus_stop", "pedestrian", "footway", "cycleway", "path",
}

# Materials: index → (name, R, G, B)
MATS = {
    0: ("asphalt",      0.15, 0.15, 0.15),
    1: ("white_mark",   0.95, 0.95, 0.95),
    2: ("yellow_mark",  0.95, 0.80, 0.00),
}

# ── geometry helpers ──────────────────────────────────────────────────────────

def _normalise(dx, dy):
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)


def _left_normal(pts, i):
    """Unit left-perpendicular at index i along a polyline."""
    n = len(pts)
    if n < 2:
        return (0.0, 1.0)
    if i == 0:
        dx, dy = pts[1][0] - pts[0][0],  pts[1][1] - pts[0][1]
    elif i == n - 1:
        dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
    else:
        dx, dy = pts[i+1][0] - pts[i-1][0], pts[i+1][1] - pts[i-1][1]
    fx, fy = _normalise(dx, dy)
    return (-fy, fx)   # rotate 90° CCW → left normal


def offset_polyline(pts, dist):
    """Shift every point of a polyline `dist` metres to the left."""
    result = []
    for i, (px, py) in enumerate(pts):
        nx, ny = _left_normal(pts, i)
        result.append((px + nx * dist, py + ny * dist))
    return result


def quad_strip(centerline, width, z):
    """
    Flat ribbon of `width` centred on `centerline` at height `z`.
    Returns (verts, tris) – verts are (x,y,z), tris are 0-based local indices.
    """
    pts = [(p[0], p[1]) for p in centerline]
    n = len(pts)
    if n < 2:
        return [], []

    hw = width * 0.5
    left, right = [], []
    for i, (px, py) in enumerate(pts):
        nx, ny = _left_normal(pts, i)
        left.append( (px + nx * hw, py + ny * hw, z))
        right.append((px - nx * hw, py - ny * hw, z))

    verts = left + right          # indices 0..n-1 left, n..2n-1 right
    tris  = []
    for i in range(n - 1):
        l0, l1 = i,     i + 1
        r0, r1 = n + i, n + i + 1
        tris += [(l0, r0, l1), (l1, r0, r1)]
    return verts, tris


def dashed_quad_strips(centerline, width, z, dash_len, gap_len):
    """
    Dashed ribbon along `centerline`.
    Returns a list of (verts, tris) pairs, one per visible dash.
    """
    pts = [(p[0], p[1]) for p in centerline]
    if len(pts) < 2:
        return []

    # Arc-length table
    arcs = [0.0]
    for i in range(1, len(pts)):
        arcs.append(arcs[-1] + math.hypot(pts[i][0] - pts[i-1][0],
                                           pts[i][1] - pts[i-1][1]))
    total = arcs[-1]
    if total < 1e-6:
        return []

    def _at(s):
        s = max(0.0, min(s, total))
        for i in range(1, len(arcs)):
            if arcs[i] >= s - 1e-9:
                seg = arcs[i] - arcs[i-1]
                t   = (s - arcs[i-1]) / max(seg, 1e-9)
                x   = pts[i-1][0] + t * (pts[i][0] - pts[i-1][0])
                y   = pts[i-1][1] + t * (pts[i][1] - pts[i-1][1])
                # Normal at this segment
                dx, dy = pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]
                fx, fy = _normalise(dx, dy)
                return x, y, -fy, fx
        dx, dy = pts[-1][0]-pts[-2][0], pts[-1][1]-pts[-2][1]
        fx, fy = _normalise(dx, dy)
        return pts[-1][0], pts[-1][1], -fy, fx

    hw      = width * 0.5
    results = []
    s       = 0.0
    while s < total:
        s_end = min(s + dash_len, total)
        if s_end - s >= 0.3:
            x0, y0, nx0, ny0 = _at(s)
            x1, y1, nx1, ny1 = _at(s_end)
            verts = [
                (x0 + nx0*hw, y0 + ny0*hw, z),   # 0 start-left
                (x0 - nx0*hw, y0 - ny0*hw, z),   # 1 start-right
                (x1 + nx1*hw, y1 + ny1*hw, z),   # 2 end-left
                (x1 - nx1*hw, y1 - ny1*hw, z),   # 3 end-right
            ]
            results.append((verts, [(0, 1, 2), (2, 1, 3)]))
        s += dash_len + gap_len
    return results


# ── OBJ builder ───────────────────────────────────────────────────────────────

class ObjBuilder:
    def __init__(self):
        self.verts  = []          # (x,y,z)
        self.groups = {}          # mat_idx → [(i0,i1,i2), ...]

    def add(self, verts, tris, mat_idx):
        if not verts or not tris:
            return
        base   = len(self.verts)
        self.verts.extend(verts)
        bucket = self.groups.setdefault(mat_idx, [])
        for t in tris:
            bucket.append((base + t[0], base + t[1], base + t[2]))

    def centre(self):
        """Translate all vertices so the XY centroid is at the origin."""
        if not self.verts:
            return
        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5
        self.verts = [(x - cx, y - cy, z) for x, y, z in self.verts]
        print(f"  Centred mesh: offset ({cx:.2f}, {cy:.2f})")

    def write_obj(self, path, mtl_filename):
        lines = [f"mtllib {mtl_filename}\n"]
        for x, y, z in self.verts:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        # One normal per z-level: +Z for road surface, -Z for anything flipped.
        # All our geometry is flat horizontal so a single up-normal covers everything.
        lines.append("vn 0.0 0.0 1.0\n")
        for mat_idx, tris in sorted(self.groups.items()):
            lines.append(f"g mat_{mat_idx}\n")
            lines.append(f"usemtl {MATS[mat_idx][0]}\n")
            for t in tris:
                # f v//vn  — same normal index (1) for every vertex
                lines.append(f"f {t[0]+1}//1 {t[1]+1}//1 {t[2]+1}//1\n")
        with open(path, "w") as f:
            f.writelines(lines)

    def write_mtl(self, path):
        lines = []
        for _, (name, r, g, b) in sorted(MATS.items()):
            lines += [
                f"newmtl {name}\n",
                f"Kd {r:.3f} {g:.3f} {b:.3f}\n",
                f"Ka 0.05 0.05 0.05\n",
                f"Ks 0.02 0.02 0.02\n",
                f"Ns 10.0\n",
                f"d 1.0\n\n",
            ]
        with open(path, "w") as f:
            f.writelines(lines)


# ── SDF / config writers ──────────────────────────────────────────────────────

def write_sdf(out_dir, model_name):
    content = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="{model_name}">
    <static>true</static>

    <link name="road_link">

      <!-- Road surface + lane markings -->
      <visual name="road_visual">
        <geometry>
          <mesh>
            <uri>meshes/road.obj</uri>
          </mesh>
        </geometry>
      </visual>

      <!-- Collision uses same mesh; disable if performance matters -->
      <collision name="road_collision">
        <surface>
          <friction>
            <ode>
              <mu>0.8</mu>
              <mu2>0.8</mu2>
            </ode>
          </friction>
        </surface>
        <geometry>
          <mesh>
            <uri>meshes/road.obj</uri>
          </mesh>
        </geometry>
      </collision>

    </link>
  </model>
</sdf>
"""
    with open(os.path.join(out_dir, "model.sdf"), "w") as f:
        f.write(content)


def write_config(out_dir, model_name):
    content = f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author>
    <name>sumo_net_to_gazebo.py</name>
  </author>
  <description>
    Road network generated from SUMO net.xml.
    Asphalt surface with lane marking geometry.
  </description>
</model>
"""
    with open(os.path.join(out_dir, "model.config"), "w") as f:
        f.write(content)


# ── main conversion ───────────────────────────────────────────────────────────

def should_skip(edge):
    etype = (edge.getType() or "").lower()
    if etype.startswith(":"):       # internal junction edges → keep
        return False
    return any(s in etype for s in SKIP_TYPES)


def convert(net_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    print(f"Loading {net_path} …")
    net = sumolib.net.readNet(net_path, withInternal=True)

    obj    = ObjBuilder()
    edges  = net.getEdges()
    total  = len(edges)
    kept   = 0

    for ei, edge in enumerate(edges):
        if should_skip(edge):
            continue
        kept += 1

        lanes   = edge.getLanes()
        n_lanes = len(lanes)

        for lane_idx, lane in enumerate(lanes):
            raw_shape = lane.getShape()
            if len(raw_shape) < 2:
                continue
            pts   = [(p[0], p[1]) for p in raw_shape]
            width = lane.getWidth()

            # ── road surface ───────────────────────────────────────────
            sv, st = quad_strip(pts, width, ROAD_Z)
            obj.add(sv, st, 0)

            # ── lane markings ──────────────────────────────────────────
            # RIGHT boundary (outer / kerb side) → always solid white
            r_line = offset_polyline(pts, -width / 2)
            rv, rt = quad_strip(r_line, MARK_W, MARK_Z)
            obj.add(rv, rt, 1)

            # LEFT boundary
            l_line = offset_polyline(pts, +width / 2)
            is_leftmost = (lane_idx == n_lanes - 1)

            if is_leftmost:
                # Centre-of-road side → solid yellow
                lv, lt = quad_strip(l_line, MARK_W, MARK_Z)
                obj.add(lv, lt, 2)
            else:
                # Lane divider (same direction traffic) → dashed white
                for dv, dt in dashed_quad_strips(l_line, MARK_W, MARK_Z,
                                                 DASH_LEN, DASH_GAP):
                    obj.add(dv, dt, 1)

    print(f"  Processed {kept}/{total} edges "
          f"({total - kept} skipped as non-road)")
    print(f"  Total vertices: {len(obj.verts):,}")

    obj.centre()

    obj_path = os.path.join(mesh_dir, "road.obj")
    mtl_path = os.path.join(mesh_dir, "road.mtl")
    obj.write_obj(obj_path, "road.mtl")
    obj.write_mtl(mtl_path)
    print(f"  Mesh  → {obj_path}")

    model_name = os.path.basename(os.path.abspath(out_dir))
    write_sdf(out_dir, model_name)
    write_config(out_dir, model_name)
    print(f"  SDF   → {os.path.join(out_dir, 'model.sdf')}")
    print(f"  Done.")
    print()
    print("── Next steps ────────────────────────────────────────────────")
    print(f"  1. Copy {out_dir}/  into your GZ_SIM_RESOURCE_PATH, e.g.:")
    print(f"       cp -r {out_dir} ~/.gz/sim/models/")
    print(f"  2. In your .world / launch file, include:")
    print(f"       <include>")
    print(f"         <uri>model://{model_name}</uri>")
    print(f"       </include>")
    print("──────────────────────────────────────────────────────────────")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sumo_net_to_gazebo.py <net.xml> [output_dir]")
        sys.exit(1)

    net_path = sys.argv[1]
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "crossroad_gazebo_model"

    if not os.path.isfile(net_path):
        print(f"Error: file not found: {net_path}")
        sys.exit(1)

    convert(net_path, out_dir)