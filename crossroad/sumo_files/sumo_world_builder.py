#!/usr/bin/env python3
"""
sumo_world_builder.py
─────────────────────────────────────────────────────────────────────────────
Reads a SUMO net.xml (+ optional poly.xml) and generates a complete
Gazebo Harmonic world SDF containing:

  • Road model  (references the crossroad_road model you already built)
  • Traffic light poles at every signalised junction
  • Overhead gantry signs on major roads (≥2 lanes)
  • Trees  — from poly.xml park/grass polygons if supplied,
             otherwise procedurally scattered away from roads
  • Buildings — from poly.xml building footprints if supplied,
                otherwise procedural box buildings along road edges

Usage:
  python3 sumo_world_builder.py crossroad.net.xml \\
          [--poly crossroad.poly.xml]             \\
          [--road-model crossroad_road]            \\
          [--out crossroad_world.sdf]

  Drop the .sdf next to your launch file and run:
      gz sim crossroad_world.sdf
─────────────────────────────────────────────────────────────────────────────
"""

import os, sys, math, random, argparse
import xml.etree.ElementTree as ET
import sumolib

random.seed(42)   # reproducible placement

# ── tunables ──────────────────────────────────────────────────────────────────
TL_OFFSET          = 3.5    # traffic-light pole distance from junction centre [m]
GANTRY_MIN_LANES   = 2      # min lanes for a gantry to be placed
GANTRY_EXTRA_SPAN  = 2.5    # extra span beyond road width for each side [m]
GANTRY_HEIGHT      = 6.0    # top of gantry beam [m]
GANTRY_SPACING     = 60.0   # minimum spacing between gantries on same edge [m]
TREE_SPACING       = 6.0    # tree-to-tree spacing in park areas [m]
TREE_ROAD_CLEAR    = 5.0    # min distance from road centre-line for procedural trees [m]
TREE_PROC_GRID     = 12.0   # procedural tree grid spacing [m]
BLDG_OFFSET_MIN    = 6.0    # min building setback from road edge [m]
BLDG_OFFSET_MAX    = 12.0   # max building setback from road edge [m]
BLDG_PROC_SPACING  = 20.0   # spacing between procedural buildings along each edge [m]

SKIP_EDGE_TYPES = {
    "rail","railway","tram","subway",
    "bus_stop","pedestrian","footway","cycleway","path",
}

# ── geometry helpers ──────────────────────────────────────────────────────────

def net_centre(net):
    b = net.getBoundary()
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def tx(x, cx): return x - cx
def ty(y, cy): return y - cy


def _norm(dx, dy):
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)


def edge_midpoint_and_heading(edge):
    """Return (mx, my, heading_rad) at the midpoint of the first lane."""
    shape = edge.getLanes()[0].getShape()
    n = len(shape)
    if n < 2:
        return shape[0][0], shape[0][1], 0.0
    mi = n // 2
    x0, y0 = shape[mi - 1]
    x1, y1 = shape[mi]
    return (x0 + x1) / 2, (y0 + y1) / 2, math.atan2(y1 - y0, x1 - x0)


def edge_total_width(edge):
    return sum(l.getWidth() for l in edge.getLanes())


def left_normal_at(pts, i):
    n = len(pts)
    if n < 2: return 0.0, 1.0
    if i == 0:       dx, dy = pts[1][0]-pts[0][0],   pts[1][1]-pts[0][1]
    elif i == n-1:   dx, dy = pts[-1][0]-pts[-2][0],  pts[-1][1]-pts[-2][1]
    else:            dx, dy = pts[i+1][0]-pts[i-1][0],pts[i+1][1]-pts[i-1][1]
    fx, fy = _norm(dx, dy)
    return -fy, fx


def poly_centroid(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return sum(xs)/len(xs), sum(ys)/len(ys)


def poly_bbox(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_poly(px, py, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj-xi)*(py-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside


def scatter_in_poly(pts, spacing):
    """Return a grid of points inside polygon `pts` at `spacing` metres."""
    xmin, ymin, xmax, ymax = poly_bbox(pts)
    result = []
    x = xmin + spacing / 2
    while x < xmax:
        y = ymin + spacing / 2
        while y < ymax:
            jx = random.uniform(-spacing * 0.3, spacing * 0.3)
            jy = random.uniform(-spacing * 0.3, spacing * 0.3)
            if point_in_poly(x + jx, y + jy, pts):
                result.append((x + jx, y + jy))
            y += spacing
        x += spacing
    return result


def all_lane_centerpoints(net):
    """Flat list of all (x,y) lane-shape points, for road-proximity checks."""
    pts = []
    for edge in net.getEdges():
        for lane in edge.getLanes():
            pts.extend((p[0], p[1]) for p in lane.getShape())
    return pts


def min_road_dist(x, y, road_pts, sample=50):
    """Approximate minimum distance from (x,y) to any road point (sampled)."""
    step = max(1, len(road_pts) // sample)
    return min(math.hypot(x - rx, y - ry)
               for rx, ry in road_pts[::step])


# ── SDF element generators ────────────────────────────────────────────────────

def _mat(r, g, b, emit=None):
    em = f"<emissive>{emit[0]} {emit[1]} {emit[2]} 1</emissive>" if emit else ""
    return (f"<material>"
            f"<ambient>{r:.2f} {g:.2f} {b:.2f} 1</ambient>"
            f"<diffuse>{r:.2f} {g:.2f} {b:.2f} 1</diffuse>"
            f"{em}</material>")


def traffic_light_sdf(idx, x, y, yaw):
    pole_h = 5.5
    return f"""
    <model name="tl_{idx}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.3f}</pose>
      <link name="link">
        <visual name="pole">
          <pose>0 0 {pole_h/2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>0.06</radius>
            <length>{pole_h:.2f}</length></cylinder></geometry>
          {_mat(0.25,0.25,0.25)}
        </visual>
        <visual name="housing">
          <pose>0 0 {pole_h+0.45:.3f} 0 0 0</pose>
          <geometry><box><size>0.33 0.33 0.9</size></box></geometry>
          {_mat(0.08,0.08,0.08)}
        </visual>
        <visual name="red">
          <pose>0 0 {pole_h+0.75:.3f} 0 0 0</pose>
          <geometry><sphere><radius>0.09</radius></sphere></geometry>
          {_mat(0.9,0.05,0.05, emit=(0.6,0,0))}
        </visual>
        <visual name="amber">
          <pose>0 0 {pole_h+0.45:.3f} 0 0 0</pose>
          <geometry><sphere><radius>0.09</radius></sphere></geometry>
          {_mat(0.9,0.55,0.0)}
        </visual>
        <visual name="green">
          <pose>0 0 {pole_h+0.15:.3f} 0 0 0</pose>
          <geometry><sphere><radius>0.09</radius></sphere></geometry>
          {_mat(0.0,0.78,0.0)}
        </visual>
        <collision name="col">
          <pose>0 0 {pole_h/2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>0.06</radius>
            <length>{pole_h:.2f}</length></cylinder></geometry>
        </collision>
      </link>
    </model>"""


def gantry_sdf(idx, x, y, yaw, span):
    """
    Overhead gantry with two posts + horizontal beam + green sign panel.
    yaw should be (road_heading + pi/2) so beam is perpendicular to traffic.
    """
    half = span / 2
    post_h = GANTRY_HEIGHT - 0.15
    sign_w = span * 0.72
    return f"""
    <model name="gantry_{idx}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.3f}</pose>
      <link name="link">
        <visual name="post_l">
          <pose>{-half:.3f} 0 {post_h/2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>0.1</radius>
            <length>{post_h:.2f}</length></cylinder></geometry>
          {_mat(0.30,0.30,0.35)}
        </visual>
        <visual name="post_r">
          <pose>{half:.3f} 0 {post_h/2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>0.1</radius>
            <length>{post_h:.2f}</length></cylinder></geometry>
          {_mat(0.30,0.30,0.35)}
        </visual>
        <visual name="beam">
          <pose>0 0 {GANTRY_HEIGHT:.3f} 0 0 0</pose>
          <geometry><box>
            <size>{span:.3f} 0.22 0.22</size></box></geometry>
          {_mat(0.30,0.30,0.35)}
        </visual>
        <visual name="sign_border">
          <pose>0 -0.14 {GANTRY_HEIGHT-0.55:.3f} 0 0 0</pose>
          <geometry><box>
            <size>{sign_w+0.08:.3f} 0.04 0.92</size></box></geometry>
          {_mat(0.9,0.85,0.1)}
        </visual>
        <visual name="sign_panel">
          <pose>0 -0.15 {GANTRY_HEIGHT-0.55:.3f} 0 0 0</pose>
          <geometry><box>
            <size>{sign_w:.3f} 0.05 0.86</size></box></geometry>
          {_mat(0.05,0.38,0.1)}
        </visual>
        <collision name="col_post_l">
          <pose>{-half:.3f} 0 {post_h/2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>0.1</radius>
            <length>{post_h:.2f}</length></cylinder></geometry>
        </collision>
        <collision name="col_post_r">
          <pose>{half:.3f} 0 {post_h/2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>0.1</radius>
            <length>{post_h:.2f}</length></cylinder></geometry>
        </collision>
        <collision name="col_beam">
          <pose>0 0 {GANTRY_HEIGHT:.3f} 0 0 0</pose>
          <geometry><box>
            <size>{span:.3f} 0.22 0.22</size></box></geometry>
        </collision>
      </link>
    </model>"""


def tree_sdf(idx, x, y, scale=1.0):
    h_trunk  = 3.2 * scale
    r_trunk  = 0.14 * scale
    r_canopy = 1.9 * scale
    z_canopy = h_trunk + r_canopy * 0.65
    return f"""
    <model name="tree_{idx}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 0</pose>
      <link name="link">
        <visual name="trunk">
          <pose>0 0 {h_trunk/2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>{r_trunk:.3f}</radius>
            <length>{h_trunk:.3f}</length></cylinder></geometry>
          {_mat(0.38,0.22,0.08)}
        </visual>
        <visual name="canopy">
          <pose>0 0 {z_canopy:.3f} 0 0 0</pose>
          <geometry><sphere><radius>{r_canopy:.3f}</radius></sphere></geometry>
          {_mat(0.08,0.44,0.10)}
        </visual>
        <collision name="col">
          <pose>0 0 {h_trunk/2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>{r_trunk:.3f}</radius>
            <length>{h_trunk:.3f}</length></cylinder></geometry>
        </collision>
      </link>
    </model>"""


def building_sdf(idx, x, y, w, d, h, yaw, r=0.62, g=0.60, b=0.58):
    return f"""
    <model name="building_{idx}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {h/2:.3f} 0 0 {yaw:.3f}</pose>
      <link name="link">
        <visual name="body">
          <geometry><box><size>{w:.2f} {d:.2f} {h:.2f}</size></box></geometry>
          {_mat(r, g, b)}
        </visual>
        <collision name="col">
          <geometry><box><size>{w:.2f} {d:.2f} {h:.2f}</size></box></geometry>
        </collision>
      </link>
    </model>"""


# ── object placement logic ────────────────────────────────────────────────────

def gen_traffic_lights(net, cx, cy):
    models = []
    idx = 0
    for node in net.getNodes():
        if node.getType() not in ("traffic_light", "traffic_light_unregulated",
                                   "traffic_light_right_on_red"):
            continue
        nx, ny = node.getCoord()
        wx, wy = tx(nx, cx), ty(ny, cy)

        # Place one pole per incoming edge direction, offset from centre
        incoming = node.getIncoming()
        placed_angles = set()
        for edge in incoming:
            shape = edge.getLanes()[0].getShape()
            if len(shape) < 2:
                continue
            # Heading of edge arriving at junction (last segment, reversed)
            ex, ey = shape[-1][0]-shape[-2][0], shape[-1][1]-shape[-2][1]
            arrive_angle = math.atan2(ey, ex)
            # Pole is to the right of arriving traffic → arrive_angle - pi/2
            pole_angle = arrive_angle - math.pi / 2
            # Snap to nearest 45° to avoid duplicates
            snapped = round(math.degrees(pole_angle) / 45) * 45
            if snapped in placed_angles:
                continue
            placed_angles.add(snapped)
            rad = math.radians(snapped)
            px = wx + TL_OFFSET * math.cos(rad)
            py = wy + TL_OFFSET * math.sin(rad)
            # Face traffic: signal faces direction arrive_angle + pi
            yaw = arrive_angle + math.pi
            models.append(traffic_light_sdf(idx, px, py, yaw))
            idx += 1

    print(f"  Traffic lights: {idx}")
    return models


def gen_gantries(net, cx, cy):
    models = []
    idx = 0
    for edge in net.getEdges():
        etype = (edge.getType() or "").lower()
        if any(s in etype for s in SKIP_EDGE_TYPES):
            continue
        if etype.startswith(":"):  # internal
            continue
        lanes = edge.getLanes()
        if len(lanes) < GANTRY_MIN_LANES:
            continue

        total_w = edge_total_width(edge)
        span    = total_w + GANTRY_EXTRA_SPAN * 2

        # Walk along the edge, placing gantries at GANTRY_SPACING intervals
        mid_lane_shape = lanes[len(lanes) // 2].getShape()
        pts = [(p[0], p[1]) for p in mid_lane_shape]

        # Arc-length table
        arcs = [0.0]
        for i in range(1, len(pts)):
            arcs.append(arcs[-1] + math.hypot(pts[i][0]-pts[i-1][0],
                                               pts[i][1]-pts[i-1][1]))
        total_len = arcs[-1]
        if total_len < GANTRY_SPACING * 0.5:
            continue

        # Place first gantry at midpoint, then space from there
        positions = []
        s = total_len / 2
        while s <= total_len:
            positions.append(s)
            s += GANTRY_SPACING
        s = total_len / 2 - GANTRY_SPACING
        while s >= 0:
            positions.append(s)
            s -= GANTRY_SPACING

        for s in positions:
            if s < 5 or s > total_len - 5:
                continue
            # Interpolate point and heading
            for i in range(1, len(arcs)):
                if arcs[i] >= s - 1e-9:
                    seg = max(arcs[i] - arcs[i-1], 1e-9)
                    t   = (s - arcs[i-1]) / seg
                    gx  = pts[i-1][0] + t * (pts[i][0] - pts[i-1][0])
                    gy  = pts[i-1][1] + t * (pts[i][1] - pts[i-1][1])
                    heading = math.atan2(pts[i][1]-pts[i-1][1],
                                         pts[i][0]-pts[i-1][0])
                    break

            # Road centre: offset from mid-lane left by half remaining width
            left_lanes = lanes[len(lanes)//2 + 1:]
            left_offset = sum(l.getWidth() for l in left_lanes) + lanes[len(lanes)//2].getWidth()/2
            nx_v, ny_v = -math.sin(heading), math.cos(heading)  # left normal
            gcx = gx + nx_v * left_offset - nx_v * total_w / 2
            gcy = gy + ny_v * left_offset - ny_v * total_w / 2

            # Gantry beam perpendicular to road → yaw = heading + pi/2
            gantry_yaw = heading + math.pi / 2
            models.append(gantry_sdf(idx,
                                     tx(gcx, cx), ty(gcy, cy),
                                     gantry_yaw, span))
            idx += 1

    print(f"  Gantry signs:   {idx}")
    return models


MAX_TREES_PER_POLY = 25
MAX_TREES_TOTAL    = 400

def gen_trees_from_poly(polys, cx, cy):
    """Scatter trees inside each park/grass polygon."""
    models = []
    idx = 0
    for poly_pts in polys:
        if idx >= MAX_TREES_TOTAL:
            break
        candidates = scatter_in_poly(poly_pts, TREE_SPACING)
        if len(candidates) > MAX_TREES_PER_POLY:
            import random as _r
            candidates = _r.sample(candidates, MAX_TREES_PER_POLY)
        for (px, py) in candidates:
            if idx >= MAX_TREES_TOTAL:
                break
            scale = random.uniform(0.75, 1.3)
            models.append(tree_sdf(idx, tx(px, cx), ty(py, cy), scale))
            idx += 1
    print(f"  Trees (poly):   {idx}")
    return models


def gen_trees_procedural(net, cx, cy, max_trees=300):
    """Place trees on a grid, keeping them away from roads."""
    b = net.getBoundary()
    road_pts = all_lane_centerpoints(net)

    models = []
    idx = 0
    gx = b[0]
    while gx < b[2] and idx < max_trees:
        gy = b[1]
        while gy < b[3] and idx < max_trees:
            jx = random.uniform(-TREE_PROC_GRID*0.35, TREE_PROC_GRID*0.35)
            jy = random.uniform(-TREE_PROC_GRID*0.35, TREE_PROC_GRID*0.35)
            px, py = gx + jx, gy + jy
            if min_road_dist(px, py, road_pts) > TREE_ROAD_CLEAR:
                scale = random.uniform(0.75, 1.4)
                models.append(tree_sdf(idx, tx(px, cx), ty(py, cy), scale))
                idx += 1
            gy += TREE_PROC_GRID
        gx += TREE_PROC_GRID

    print(f"  Trees (proc):   {idx}")
    return models


def gen_buildings_from_poly(polys, cx, cy):
    """Extrude each building footprint into a box."""
    # Palette of realistic building colours
    colours = [
        (0.65,0.60,0.55), (0.70,0.65,0.58), (0.60,0.58,0.55),
        (0.72,0.68,0.62), (0.55,0.53,0.50), (0.78,0.74,0.68),
        (0.62,0.58,0.52), (0.68,0.64,0.60),
    ]
    models = []
    for idx, poly_pts in enumerate(polys):
        xmin, ymin, xmax, ymax = poly_bbox(poly_pts)
        w = xmax - xmin
        d = ymax - ymin
        if w < 2 or d < 2:
            continue
        pcx, pcy = (xmin+xmax)/2, (ymin+ymax)/2
        h = random.uniform(5, 28)
        yaw = random.uniform(-0.15, 0.15)
        r, g, b = colours[idx % len(colours)]
        r += random.uniform(-0.05, 0.05)
        g += random.uniform(-0.05, 0.05)
        b += random.uniform(-0.05, 0.05)
        models.append(building_sdf(idx,
                                   tx(pcx, cx), ty(pcy, cy),
                                   min(w, 60), min(d, 60), h, yaw,
                                   r, g, b))
    print(f"  Buildings (poly): {len(models)}")
    return models


def gen_buildings_procedural(net, cx, cy):
    """Place box buildings along both sides of each edge."""
    colours = [
        (0.65,0.60,0.55),(0.70,0.65,0.58),(0.60,0.58,0.55),
        (0.72,0.68,0.62),(0.55,0.53,0.50),(0.78,0.74,0.68),
    ]
    models = []
    idx = 0
    for edge in net.getEdges():
        etype = (edge.getType() or "").lower()
        if any(s in etype for s in SKIP_EDGE_TYPES) or etype.startswith(":"):
            continue
        lanes = edge.getLanes()
        total_w = edge_total_width(edge)
        shape = lanes[0].getShape()
        pts = [(p[0], p[1]) for p in shape]

        arcs = [0.0]
        for i in range(1, len(pts)):
            arcs.append(arcs[-1] + math.hypot(pts[i][0]-pts[i-1][0],
                                               pts[i][1]-pts[i-1][1]))
        total_len = arcs[-1]
        if total_len < BLDG_PROC_SPACING:
            continue

        s = BLDG_PROC_SPACING / 2
        while s < total_len - 5:
            for i in range(1, len(arcs)):
                if arcs[i] >= s - 1e-9:
                    seg = max(arcs[i]-arcs[i-1], 1e-9)
                    t = (s - arcs[i-1]) / seg
                    bx = pts[i-1][0] + t*(pts[i][0]-pts[i-1][0])
                    by = pts[i-1][1] + t*(pts[i][1]-pts[i-1][1])
                    heading = math.atan2(pts[i][1]-pts[i-1][1],
                                         pts[i][0]-pts[i-1][0])
                    break

            nx_v, ny_v = -math.sin(heading), math.cos(heading)
            bw = random.uniform(8, 22)
            bd = random.uniform(8, 18)
            bh = random.uniform(6, 28)
            yaw = heading + random.uniform(-0.12, 0.12)
            r, g, b = colours[idx % len(colours)]
            r += random.uniform(-0.04, 0.04)
            g += random.uniform(-0.04, 0.04)
            b += random.uniform(-0.04, 0.04)

            # Both sides
            for side in [+1, -1]:
                offset = side * (total_w/2 + random.uniform(BLDG_OFFSET_MIN,
                                                              BLDG_OFFSET_MAX)
                                 + bw/2)
                px = bx + nx_v * offset
                py = by + ny_v * offset
                models.append(building_sdf(idx,
                                           tx(px, cx), ty(py, cy),
                                           bw, bd, bh, yaw, r, g, b))
                idx += 1
            s += BLDG_PROC_SPACING

    print(f"  Buildings (proc): {len(models)}")
    return models


# ── poly.xml parser ───────────────────────────────────────────────────────────

def parse_poly_xml(path):
    """
    Returns (building_polys, park_polys) — each a list of [(x,y),...] polygons.
    Handles both SUMO poly.xml and OSM-exported additional files.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    buildings, parks = [], []
    building_types = {
        "building", "shop", "amenity", "tourism", "clinic",
        "residential", "university", "school", "man_made",
        "kindergarten", "historic",
    }
    park_types = {"natural", "landuse", "leisure", "sport"}
    skip_types = {"parking", "aeroway", "traffic_sign"}

    for poly in root.findall(".//poly"):
        ptype = (poly.get("type") or "").lower()
        shape_str = poly.get("shape","")
        if not shape_str or ptype in skip_types:
            continue
        try:
            pts = [tuple(map(float, p.split(","))) for p in shape_str.split()]
        except ValueError:
            continue
        if len(pts) < 3:
            continue
        if ptype in building_types:
            buildings.append(pts)
        elif ptype in park_types:
            parks.append(pts)
    return buildings, parks


# ── world SDF writer ──────────────────────────────────────────────────────────

def write_world(out_path, road_model_name, all_models):
    header = f"""<?xml version="1.0" ?>
<!--
  Generated by sumo_world_builder.py
  Road model: {road_model_name}
  Objects: {len(all_models)} models
-->
<sdf version="1.9">
  <world name="crossroad_world">

    <!-- ── Lighting ───────────────────────────────────────────── -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 500 0 0.3 0.6</pose>
      <diffuse>0.95 0.95 0.90 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>-0.5 0.3 -0.8</direction>
    </light>

    <light name="ambient_fill" type="directional">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 100 0 0 0</pose>
      <diffuse>0.35 0.38 0.42 1</diffuse>
      <direction>0.3 -0.5 -0.7</direction>
    </light>

    <!-- ── Physics ────────────────────────────────────────────── -->
    <physics name="default" type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- ── Ground plane (below road, catches shadows) ─────────── -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <pose>0 0 -0.01 0 0 0</pose>
          <geometry><plane><normal>0 0 1</normal>
            <size>1000 1000</size></plane></geometry>
          <material>
            <ambient>0.20 0.20 0.18 1</ambient>
            <diffuse>0.20 0.20 0.18 1</diffuse>
          </material>
        </visual>
        <collision name="col">
          <pose>0 0 -0.01 0 0 0</pose>
          <geometry><plane><normal>0 0 1</normal>
            <size>1000 1000</size></plane></geometry>
        </collision>
      </link>
    </model>

    <!-- ── Road network mesh ──────────────────────────────────── -->
    <include>
      <uri>model://{road_model_name}</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>

    <!-- ── Scene objects ──────────────────────────────────────── -->
"""
    footer = """
  </world>
</sdf>
"""
    with open(out_path, "w") as f:
        f.write(header)
        for m in all_models:
            f.write(m)
            f.write("\n")
        f.write(footer)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SUMO net.xml → Gazebo Harmonic world SDF")
    ap.add_argument("net",          help="SUMO net.xml path")
    ap.add_argument("--poly",       help="Optional SUMO poly.xml (buildings + parks)",
                    default=None)
    ap.add_argument("--road-model", help="Gazebo model name for road mesh",
                    default="crossroad_road")
    ap.add_argument("--out",        help="Output world SDF path",
                    default="crossroad_world.sdf")
    ap.add_argument("--no-tl",      action="store_true", help="Skip traffic lights")
    ap.add_argument("--no-gantry",  action="store_true", help="Skip gantry signs")
    ap.add_argument("--no-trees",   action="store_true", help="Skip trees")
    ap.add_argument("--no-buildings", action="store_true", help="Skip buildings")
    args = ap.parse_args()

    if not os.path.isfile(args.net):
        print(f"Error: {args.net} not found"); sys.exit(1)

    print(f"Loading {args.net} …")
    net = sumolib.net.readNet(args.net, withInternal=False)
    cx, cy = net_centre(net)
    print(f"  Network centre: ({cx:.1f}, {cy:.1f})")

    all_models = []

    # ── traffic lights ────────────────────────────────────────────
    if not args.no_tl:
        all_models += gen_traffic_lights(net, cx, cy)

    # ── gantry signs ──────────────────────────────────────────────
    if not args.no_gantry:
        all_models += gen_gantries(net, cx, cy)

    # ── trees ─────────────────────────────────────────────────────
    if not args.no_trees:
        if args.poly:
            _, park_polys = parse_poly_xml(args.poly)
            if park_polys:
                all_models += gen_trees_from_poly(park_polys, cx, cy)
            else:
                print("  No park polygons in poly.xml — using procedural trees")
                all_models += gen_trees_procedural(net, cx, cy)
        else:
            all_models += gen_trees_procedural(net, cx, cy)

    # ── buildings ─────────────────────────────────────────────────
    if not args.no_buildings:
        if args.poly:
            bldg_polys, _ = parse_poly_xml(args.poly)
            if bldg_polys:
                all_models += gen_buildings_from_poly(bldg_polys, cx, cy)
            else:
                print("  No building polygons in poly.xml — using procedural buildings")
                all_models += gen_buildings_procedural(net, cx, cy)
        else:
            all_models += gen_buildings_procedural(net, cx, cy)

    # ── write world ───────────────────────────────────────────────
    write_world(args.out, args.road_model, all_models)
    b = net.getBoundary()
    print(f"\n  Total objects: {len(all_models)}")
    print(f"  World written: {args.out}")
    print()
    print("── Next steps ────────────────────────────────────────────────────")
    print(f"  Make sure the road model is installed:")
    print(f"    cp -r crossroad_road ~/.gz/sim/models/")
    print(f"  Then launch:")
    print(f"    gz sim {args.out}")
    print("──────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()