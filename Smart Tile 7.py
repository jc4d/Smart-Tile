bl_info = {
    "name": "Smart Tile Generator",
    "author": "You",
    "version": (1, 2, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > Tile Gen",
    "description": "Procedural tile pattern generator with optimized per-row dynamic bounds and percentage-based offsets",
    "category": "Mesh",
}

import bpy
import bmesh
import mathutils
import math
import random
import json
from bpy.props import (
    FloatProperty,
    IntProperty,
    EnumProperty,
    StringProperty,
    BoolProperty,
    PointerProperty,
    CollectionProperty,
)
from bpy.types import PropertyGroup, Operator, Panel, UIList

# Guard flag: prevents realtime-update callbacks from re-entering / firing a
# storm of regenerations while we're programmatically setting several
# properties in a row (e.g. when the pattern dropdown changes its defaults,
# or when settings are copied onto a freshly generated batch object).
_suspend_realtime_update = False

# Same purpose as _suspend_realtime_update, but dedicated to the material-rule
# real-time re-application (see _trigger_material_rule_update below), kept
# separate so a pending geometry regen and a pending material-rule re-apply
# never suppress each other.
_suspend_material_rule_update = False


# ---------------------------------------------------------------------------
# CORE GEOMETRY FUNCTIONS
# ---------------------------------------------------------------------------

def encode_face_data_snapshot(face_data):
    """
    Serializes face_data (list of (normal, [verts])) into a JSON string that
    can be stored on the batch object's property group, so it can be
    reused verbatim by every future update instead of being re-derived from
    the live mesh (see face_snapshot on SmartTileSettings).
    """
    payload = [
        [round(n.x, 6), round(n.y, 6), round(n.z, 6),
         [[round(v.x, 6), round(v.y, 6), round(v.z, 6)] for v in verts]]
        for n, verts in face_data
    ]
    return json.dumps(payload)


def decode_face_data_snapshot(snapshot_str):
    """Inverse of encode_face_data_snapshot. Returns None if empty/invalid
    (e.g. objects generated before this feature existed), so callers can
    fall back to the old index-based re-derivation."""
    if not snapshot_str:
        return None
    try:
        payload = json.loads(snapshot_str)
    except (ValueError, TypeError):
        return None
    face_data = []
    for entry in payload:
        nx, ny, nz, verts = entry
        normal = mathutils.Vector((nx, ny, nz))
        vert_list = [mathutils.Vector((v[0], v[1], v[2])) for v in verts]
        face_data.append((normal, vert_list))
    return face_data


def group_faces_by_plane(face_data):
    """
    Groups (normal, verts) face entries by the plane they lie on (rounded
    normal + rounded plane distance), so faces belonging to the same flat
    wall/surface are grouped together, while faces on a different plane
    (e.g. the other side of a 90 degree corner) end up in a different group.

    Used both for tile placement AND for building the boolean cutter, so the
    two stay consistent: each planar group gets its own independent, flat
    cutter island instead of one continuous creased shell.
    """
    groups = {}
    for f_normal, f_verts in face_data:
        # Calculate plane offset: d = normal dot point
        # Use rounding to group faces that are effectively on the same plane
        plane_dist = round(f_normal.dot(f_verts[0]), 3)
        key = (round(f_normal.x, 2), round(f_normal.y, 2), round(f_normal.z, 2), plane_dist)
        groups.setdefault(key, {"normal": f_normal, "faces": []})["faces"].append(f_verts)
    return groups


def get_pattern_offset(pos, axis_u, axis_v, normal, offset_x, offset_y, offset_z):
    offset_vec = (offset_x * axis_u) + (offset_y * axis_v) + (offset_z * normal)
    return pos + offset_vec


def _fast_tile_bool(seed, tile_idx, salt):
    """
    Deterministic pseudo-random bit for per-tile UV-flip decisions, used in
    place of constructing a fresh random.Random(seed) per tile. Profiling
    showed ~24k Random() object constructions (seed + init) costing real
    time for something that only needed a single coin-flip each. This is a
    small integer hash (SplitMix-style bit mixing) instead: same seed+idx
    always gives the same result (still fully deterministic/reproducible),
    but with no object-construction overhead.
    """
    x = (seed * 2654435761 + tile_idx * 2246822519 + salt * 3266489917) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85ebca6b) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xc2b2ae35) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 1


def set_tile_edge_attributes(bm, depth, attr_names):
    """
    Faces that belong to the original, pristine (pre-clip) tile geometry
    carry a temporary "_orig_tile_face" marker. Faces contributed by the
    clip-cutter itself never had that marker, so any edge touching one of
    those faces is, by definition, an edge born from the boolean clip
    rather than a genuine tile edge.
    """
    # Clean up any existing layers with the current target names
    for key, name in attr_names.items():
        layer = bm.edges.layers.float.get(name)
        if layer:
            bm.edges.layers.float.remove(layer)

    # Create new layers using the dynamic names
    layers = {key: bm.edges.layers.float.new(name) for key, name in attr_names.items()}

    l_normal = bm.faces.layers.float_vector.get("tile_normal")
    lu = bm.faces.layers.float_vector.get("tile_axis_u")
    lv = bm.faces.layers.float_vector.get("tile_axis_v")

    orig_face_marker = bm.faces.layers.float.get("_orig_tile_face")

    if orig_face_marker is not None:
        marked = sum(1 for f in bm.faces if f[orig_face_marker] >= 0.5)
        if marked == 0:
            orig_face_marker = None

    # Precompute per-face data ONCE instead of inside the edge loop below.
    face_cache = {}
    if l_normal:
        for f in bm.faces:
            is_bool_f = orig_face_marker is not None and f[orig_face_marker] < 0.5
            tile_normal = mathutils.Vector(f[l_normal]).normalized()
            dot = f.normal.normalized().dot(tile_normal)
            is_top_f = dot > 0.9
            is_bot_f = dot < -0.9
            axis_u_f = mathutils.Vector(f[lu]) if (is_top_f and lu) else None
            axis_v_f = mathutils.Vector(f[lv]) if (is_top_f and lv) else None
            face_cache[f] = (is_bool_f, is_top_f, is_bot_f, axis_u_f, axis_v_f)
    elif orig_face_marker is not None:
        for f in bm.faces:
            face_cache[f] = (f[orig_face_marker] < 0.5, False, False, None, None)

    for edge in bm.edges:
        linked = edge.link_faces
        if not linked: continue

        is_boolean_edge = False
        if orig_face_marker is not None:
            is_boolean_edge = any(face_cache[f][0] for f in linked)

        is_top = is_bot = is_width_edge = is_length_edge = False

        if not is_boolean_edge:
            if l_normal:
                for f in linked:
                    _, is_top_f, is_bot_f, _, _ = face_cache[f]
                    if is_top_f: is_top = True
                    elif is_bot_f: is_bot = True

            if is_top and lu and lv:
                cap_face = next((f for f in linked if face_cache[f][1]), None)
                if cap_face:
                    axis_u, axis_v = face_cache[cap_face][3], face_cache[cap_face][4]
                    vec = (edge.verts[0].co - edge.verts[1].co).normalized()
                    is_width_edge = abs(vec.dot(axis_u)) > 0.9
                    is_length_edge = abs(vec.dot(axis_v)) > 0.9

        edge[layers["top"]] = 1.0 if is_top else 0.0
        edge[layers["bottom"]] = 1.0 if is_bot else 0.0
        edge[layers["side"]] = 1.0 if (not is_boolean_edge and not (is_top or is_bot)) else 0.0
        edge[layers["width"]] = 1.0 if is_width_edge else 0.0
        edge[layers["length"]] = 1.0 if is_length_edge else 0.0
        if "boolean" in layers:
            edge[layers["boolean"]] = 1.0 if is_boolean_edge else 0.0

    if orig_face_marker is not None:
        bm.faces.layers.float.remove(orig_face_marker)


def get_stretcher_matrices(groups, width, length, depth, rot_rad, row_offset, staggered_offset, max_random_offset,
                            width_gap, length_gap, random_offset_seed, offset_x, offset_y, offset_z, tiling_axis='BOTH'):
    all_matrices = []
    stretcher_rng = random.Random(random_offset_seed)
    random_row_shifts = {}
    e_width, e_length = width - width_gap, length - length_gap

    for key, data in groups.items():
        normal, faces = data["normal"], data["faces"]
        axis_u = (mathutils.Vector((0, 0, 1)) if abs(normal.dot(mathutils.Vector((0, 0, 1))) ) < 0.9 else mathutils.Vector((0, 1, 0)))
        axis_u = (axis_u - axis_u.dot(normal) * normal).normalized()
        axis_v = normal.cross(axis_u).normalized()
        if rot_rad != 0.0:
            rot_mat_axes = mathutils.Matrix.Rotation(rot_rad, 3, normal)
            axis_u, axis_v = rot_mat_axes @ axis_u, rot_mat_axes @ axis_v

        all_verts = [v for face in faces for v in face]
        anchor_base = all_verts[0]
        anchor = get_pattern_offset(anchor_base, axis_u, axis_v, normal, offset_x, offset_y, offset_z)
        projected = [((v - anchor_base).dot(axis_u), (v - anchor_base).dot(axis_v)) for v in all_verts]
        u_min, u_max = min(p[0] for p in projected), max(p[0] for p in projected)
        v_min, v_max = min(p[1] for p in projected), max(p[1] for p in projected)

        u_step, v_step = width + width_gap, length + length_gap
        u_off_step = math.floor(offset_x / u_step)
        v_off_step = math.floor(offset_y / v_step)
        
        # Determine strict vertical bounds
        j_min = math.floor((v_min - offset_y) / v_step) - 1
        j_max = math.ceil((v_max - offset_y) / v_step) + 1

        for j in range(j_min, j_max):
            if tiling_axis == 'X' and (j + v_off_step) != 0:
                continue
            if j not in random_row_shifts:
                random_row_shifts[j] = stretcher_rng.uniform(0, max_random_offset)
            
            # Row & Stagger offsets calculated as a percentage multiplier of the tile width
            row_shift = (row_offset * width if j % 2 != 0 else 0.0) + (j * staggered_offset * width) + random_row_shifts[j]
            
            # DYNAMIC OPTIMIZATION: Bounds are recalculated per row to track the shift sloped angle perfectly
            i_min = math.floor((u_min - offset_x - row_shift - width) / u_step)
            i_max = math.ceil((u_max - offset_x - row_shift + width) / u_step)

            for i in range(i_min, i_max + 1):
                if tiling_axis == 'Y' and (i + u_off_step) != 0:
                    continue
                u_pos = (i * u_step) + (width * 0.5) + row_shift
                v_pos = (j * v_step) + (length * 0.5)
                tile_center = anchor + (u_pos * axis_u) + (v_pos * axis_v)
                all_matrices.append(create_tile_matrix(tile_center, axis_u, axis_v, normal, e_width, e_length, depth) + (i, j))
    return all_matrices


def get_herringbone_matrices(groups, width, length, depth, rot_rad, row_offset, staggered_offset, max_random_offset,
                              width_gap, length_gap, offset_x, offset_y, offset_z, tiling_axis='BOTH'):
    all_matrices = []
    e_width = width - width_gap
    e_length = length - length_gap
    offset_u = length_gap / 2.0
    offset_v = width_gap / 2.0

    for key, data in groups.items():
        normal, faces = data["normal"], data["faces"]
        axis_u = (mathutils.Vector((0, 0, 1)) if abs(normal.dot(mathutils.Vector((0, 0, 1)))) < 0.9 else mathutils.Vector((0, 1, 0)))
        axis_u = (axis_u - axis_u.dot(normal) * normal).normalized()
        axis_v = normal.cross(axis_u).normalized()
        anchor_base = [v for f in faces for v in f][0]
        anchor = get_pattern_offset(anchor_base, axis_u, axis_v, normal, offset_x, offset_y, offset_z)

        rot_grid = mathutils.Matrix.Rotation(rot_rad, 3, normal)
        axis_u, axis_v = rot_grid @ axis_u, rot_grid @ axis_v

        step_u = length / math.sqrt(2)
        step_v = width / math.sqrt(2)
        row_step = 2 * step_v
        col_step = 2 * step_u

        rot_pos = mathutils.Matrix.Rotation(math.radians(45), 3, normal)
        u_pos, v_pos = rot_pos @ axis_u, rot_pos @ axis_v
        rot_neg = mathutils.Matrix.Rotation(math.radians(-45), 3, normal)
        u_neg, v_neg = rot_neg @ axis_u, rot_neg @ axis_v

        all_verts = [v for face in faces for v in face]
        projected_u = [(v - anchor_base).dot(axis_u) for v in all_verts]
        projected_v = [(v - anchor_base).dot(axis_v) for v in all_verts]
        u_min, u_max = min(projected_u), max(projected_u)
        v_min, v_max = min(projected_v), max(projected_v)

        row_off_step = math.floor(offset_y / row_step)
        col_off_step = math.floor(offset_x / col_step)
        
        # Tightened global grid footprint padding down to 1
        row_min = math.floor(v_min / row_step) - 1 - row_off_step
        row_max = math.ceil(v_max / row_step) + 1 - row_off_step
        col_min = math.floor(u_min / col_step) - 1 - col_off_step
        col_max = math.ceil(u_max / col_step) + 1 - col_off_step

        for row in range(row_min, row_max):
            if tiling_axis == 'X' and (row + row_off_step) != 0:
                continue
            row_origin = anchor + (row * row_step) * axis_v
            for col in range(col_min, col_max):
                if tiling_axis == 'Y' and (col + col_off_step) != 0:
                    continue
                pos_a = row_origin + (col * col_step) * axis_u
                centered_pos_a = pos_a + (offset_u * u_pos) + (offset_v * v_pos)
                all_matrices.append(create_tile_matrix(centered_pos_a, u_pos, v_pos, normal, e_length, e_width, depth) + (col, row))
                pos_b = pos_a + (step_u - step_v) * axis_u + (step_u - step_v) * axis_v
                centered_pos_b = pos_b + (offset_u * u_neg) + (offset_v * v_neg)
                all_matrices.append(create_tile_matrix(centered_pos_b, u_neg, v_neg, normal, e_length, e_width, depth) + (col, row))

    return all_matrices


def get_chevron_matrices(groups, width, length, depth, rot_rad, row_offset, staggered_offset, max_random_offset,
                          width_gap, length_gap, offset_x, offset_y, offset_z, tiling_axis='BOTH'):
    all_matrices = []
    for key, data in groups.items():
        normal, faces = data["normal"], data["faces"]
        axis_u = (mathutils.Vector((0, 0, 1)) if abs(normal.dot(mathutils.Vector((0, 0, 1)))) < 0.9 else mathutils.Vector((0, 1, 0)))
        axis_u = (axis_u - axis_u.dot(normal) * normal).normalized()
        axis_v = normal.cross(axis_u).normalized()
        anchor_base = [v for f in faces for v in f][0]
        anchor = get_pattern_offset(anchor_base, axis_u, axis_v, normal, offset_x, offset_y, offset_z)

        rot_grid = mathutils.Matrix.Rotation(rot_rad, 3, normal)
        axis_u, axis_v = rot_grid @ axis_u, rot_grid @ axis_v
        rot_pos = mathutils.Matrix.Rotation(math.radians(45), 3, normal)
        u_pos, v_pos = rot_pos @ axis_u, rot_pos @ axis_v
        rot_neg = mathutils.Matrix.Rotation(math.radians(-45), 3, normal)
        u_neg, v_neg = rot_neg @ axis_u, rot_neg @ axis_v

        shear = width / length
        shear_pos = mathutils.Matrix.Identity(4)
        shear_pos[0][1] = shear
        shear_neg = mathutils.Matrix.Identity(4)
        shear_neg[0][1] = -shear

        row_step = length * math.sqrt(2)
        leg_width = width * math.sqrt(2)
        correction = (length / 2) * (u_pos - u_neg) + (width / 2) * (v_pos - v_neg)

        all_verts = [v for face in faces for v in face]
        projected_u = [(v - anchor_base).dot(axis_u) for v in all_verts]
        projected_v = [(v - anchor_base).dot(axis_v) for v in all_verts]
        u_min, u_max = min(projected_u), max(projected_u)
        v_min, v_max = min(projected_v), max(projected_v)

        row_off_step = math.floor(offset_x / row_step)
        col_off_step = math.floor(offset_y / leg_width)
        
        # Reduced heavy flat safe padding (from 6 down to 1) to save thousands of redundant tiles
        pad = 1
        row_min = math.floor(u_min / row_step) - pad - row_off_step
        row_max = math.ceil(u_max / row_step) + pad - row_off_step
        col_min = math.floor(v_min / leg_width) - pad - col_off_step
        col_max = math.ceil(v_max / leg_width) + pad - col_off_step

        for row in range(row_min, row_max):
            if tiling_axis == 'Y' and (row + row_off_step) != 0:
                continue
            for col in range(col_min, col_max):
                if tiling_axis == 'X' and (col + col_off_step) != 0:
                    continue
                origin = anchor + (col * (leg_width + width_gap)) * axis_v + (row * (row_step + length_gap)) * axis_u
                mat, n, u, v = create_tile_matrix(origin, u_pos, v_pos, normal, length, width, depth)
                all_matrices.append((mat @ shear_pos, n, u, v, row, col))
                seam_push = (v_pos - v_neg).normalized() * -(width_gap * 0.5)
                mat, n, u, v = create_tile_matrix(origin + correction + (row_step / 2) * axis_u + seam_push, u_neg, v_neg, normal, length, width, depth)
                all_matrices.append((mat @ shear_neg, n, u, v, row, col))
    return all_matrices


def get_windmill_matrices(groups, width, length, depth, rot_rad, row_offset, staggered_offset, max_random_offset,
                           width_gap, length_gap, offset_x, offset_y, offset_z, tiling_axis='BOTH'):
    all_matrices = []
    for key, data in groups.items():
        normal, faces = data["normal"], data["faces"]
        axis_u = (mathutils.Vector((0, 0, 1)) if abs(normal.dot(mathutils.Vector((0, 0, 1)))) < 0.9 else mathutils.Vector((0, 1, 0)))
        axis_u = (axis_u - axis_u.dot(normal) * normal).normalized()
        axis_v = normal.cross(axis_u).normalized()
        anchor_base = [v for face in faces for v in face][0]
        anchor = get_pattern_offset(anchor_base, axis_u, axis_v, normal, offset_x, offset_y, offset_z)

        rot_grid = mathutils.Matrix.Rotation(rot_rad, 3, normal)
        axis_u, axis_v = rot_grid @ axis_u, rot_grid @ axis_v

        cell = length + width
        all_verts = [v for face in faces for v in face]
        projected_u = [(v - anchor_base).dot(axis_u) for v in all_verts]
        projected_v = [(v - anchor_base).dot(axis_v) for v in all_verts]
        u_min, u_max = min(projected_u), max(projected_u)
        v_min, v_max = min(projected_v), max(projected_v)

        row_off_step = math.floor(offset_x / cell)
        col_off_step = math.floor(offset_y / cell)
        
        # Tight padding buffer reduced to 1
        row_min = math.floor(u_min / cell) - 1 - row_off_step
        row_max = math.ceil(u_max / cell) + 1 - row_off_step
        col_min = math.floor(v_min / cell) - 1 - col_off_step
        col_max = math.ceil(v_max / cell) + 1 - col_off_step

        center_gap = (width_gap + length_gap) / 2.0

        for row in range(row_min, row_max):
            if tiling_axis == 'Y' and (row + row_off_step) != 0:
                continue
            for col in range(col_min, col_max):
                if tiling_axis == 'X' and (col + col_off_step) != 0:
                    continue
                origin = anchor + (col * cell) * axis_v + (row * cell) * axis_u
                all_matrices.append(create_tile_matrix(origin, axis_u, axis_v, normal, length - length_gap, width - width_gap, depth) + (row, col))
                all_matrices.append(create_tile_matrix(origin + (length + width) * axis_u, axis_v, -axis_u, normal, length - length_gap, width - width_gap, depth) + (row, col))
                all_matrices.append(create_tile_matrix(origin + width * axis_v + width * axis_u, axis_v, -axis_u, normal, length - length_gap, width - width_gap, depth) + (row, col))
                all_matrices.append(create_tile_matrix(origin + length * axis_v + width * axis_u, axis_u, axis_v, normal, length - length_gap, width - width_gap, depth) + (row, col))
                all_matrices.append(create_tile_matrix(origin + width * axis_v + width * axis_u, axis_u, axis_v, normal, length - width - center_gap, length - width - center_gap, depth) + (row, col))
    return all_matrices


def get_custom_tile_matrices(groups, width, length, depth, rot_rad, row_offset, staggered_offset, max_random_offset,
                              width_gap, length_gap, random_offset_seed, offset_x, offset_y, offset_z,
                              random_depth, random_depth_seed, tile_name="CustomTileTemplate", tiling_axis='BOTH'):
    all_matrices = []
    tile_obj = bpy.data.objects.get(tile_name)
    if not tile_obj:
        return all_matrices

    stretcher_rng = random.Random(random_offset_seed)
    t_dim = tile_obj.dimensions
    new_w = t_dim.x * width
    new_l = t_dim.y * length
    u_step = new_w + width_gap
    v_step = new_l + length_gap
    u_off_step = math.floor(offset_x / u_step)
    v_off_step = math.floor(offset_y / v_step)

    for key, data in groups.items():
        normal, faces = data["normal"], data["faces"]
        axis_u = (mathutils.Vector((0, 0, 1)) if abs(normal.dot(mathutils.Vector((0, 0, 1)))) < 0.9 else mathutils.Vector((0, 1, 0)))
        axis_u = (axis_u - axis_u.dot(normal) * normal).normalized()
        axis_v = normal.cross(axis_u).normalized()

        if rot_rad != 0.0:
            rot_mat_axes = mathutils.Matrix.Rotation(rot_rad, 3, normal)
            axis_u, axis_v = rot_mat_axes @ axis_u, rot_mat_axes @ axis_v

        all_verts = [v for face in faces for v in face]
        anchor_base = all_verts[0]
        anchor = get_pattern_offset(anchor_base, axis_u, axis_v, normal, offset_x, offset_y, offset_z)
        projected = [((v - anchor_base).dot(axis_u), (v - anchor_base).dot(axis_v)) for v in all_verts]
        u_min, u_max = min(p[0] for p in projected), max(p[0] for p in projected)
        v_min, v_max = min(p[1] for p in projected), max(p[1] for p in projected)

        j_min = math.floor((v_min - offset_y) / v_step) - 1
        j_max = math.ceil((v_max - offset_y) / v_step) + 1

        for j in range(j_min, j_max):
            if tiling_axis == 'X' and (j + v_off_step) != 0:
                continue
            random_off = stretcher_rng.uniform(0, max_random_offset)
            
            # Custom templates use percentual factor multipliers mapped directly to the evaluated width bounds (new_w)
            shift = (row_offset * new_w if j % 2 != 0 else 0.0) + (j * staggered_offset * new_w) + random_off
            
            # Recalculated dynamic internal loops to shrink horizontal footprint limits tightly per row
            i_min = math.floor((u_min - offset_x - shift - new_w) / u_step)
            i_max = math.ceil((u_max - offset_x - shift + new_w) / u_step)

            for i in range(i_min, i_max + 1):
                if tiling_axis == 'Y' and (i + u_off_step) != 0:
                    continue
                center_pos = anchor + ((i * u_step) + shift + (new_w * 0.5)) * axis_u + ((j * v_step) + (new_l * 0.5)) * axis_v

                scale_mat = mathutils.Matrix.Diagonal((width, length, depth, 1.0))
                rot_mat = mathutils.Matrix((axis_u, axis_v, normal)).transposed().to_4x4()
                trans = mathutils.Matrix.Translation(center_pos)
                all_matrices.append((trans @ rot_mat @ scale_mat, normal, axis_u, axis_v, i, j))
    return all_matrices


def create_tile_matrix(pos, u_dir, v_dir, normal, l, w, d, z_offset=0.0):
    rot_mat = mathutils.Matrix((u_dir, v_dir, normal)).transposed().to_4x4()
    scale_mat = mathutils.Matrix.Diagonal((l, w, d, 1.0))
    center = pos + (l * 0.5) * u_dir + (w * 0.5) * v_dir + (normal * z_offset)
    return (mathutils.Matrix.Translation(center + normal * (d * 0.5)) @ rot_mat @ scale_mat, normal, u_dir, v_dir)


def get_placement_matrices(pattern, groups, width, length, depth, rot_rad, row_offset, staggered_offset, max_random_offset,
                            width_gap, length_gap, offset_x, offset_y, offset_z, random_offset_seed,
                            random_depth, random_depth_seed, custom_obj_name="CustomTileTemplate", tiling_axis='BOTH'):
    args = (groups, width, length, depth, rot_rad, row_offset, staggered_offset, max_random_offset, width_gap, length_gap, offset_x, offset_y, offset_z)

    if pattern == "HERRINGBONE":
        return get_herringbone_matrices(*args, tiling_axis=tiling_axis)
    if pattern == "CHEVRON":
        return get_chevron_matrices(*args, tiling_axis=tiling_axis)
    if pattern == "WINDMILL":
        return get_windmill_matrices(*args, tiling_axis=tiling_axis)
    if pattern == "CUSTOM":
        return get_custom_tile_matrices(
            groups, width, length, depth, rot_rad, row_offset, staggered_offset, max_random_offset,
            width_gap, length_gap, random_offset_seed, offset_x, offset_y, offset_z,
            random_depth, random_depth_seed, tile_name=custom_obj_name, tiling_axis=tiling_axis
        )
    return get_stretcher_matrices(groups, width, length, depth, rot_rad, row_offset, staggered_offset, max_random_offset,
                                   width_gap, length_gap, random_offset_seed, offset_x, offset_y, offset_z, tiling_axis=tiling_axis)


def _build_single_plane_cutter(faces, obj_matrix_world, depth, offset_z, name="_tile_cutter_tmp"):
    cutter_bm = bmesh.new()
    vert_map = {}
    for face in faces:
        for v in face:
            key = (round(v.x, 4), round(v.y, 4), round(v.z, 4))
            if key not in vert_map:
                vert_map[key] = cutter_bm.verts.new(v)
    cutter_bm.verts.ensure_lookup_table()
    for face in faces:
        try:
            cutter_bm.faces.new([vert_map[(round(v.x, 4), round(v.y, 4), round(v.z, 4))] for v in face])
        except (ValueError, KeyError):
            pass

    cutter_mesh = bpy.data.meshes.new(name)
    cutter_bm.to_mesh(cutter_mesh)
    cutter_bm.free()
    cutter_obj = bpy.data.objects.new(name, cutter_mesh)
    bpy.context.collection.objects.link(cutter_obj)
    cutter_obj.matrix_world = obj_matrix_world

    solidify = cutter_obj.modifiers.new("Solidify", 'SOLIDIFY')
    solidify.thickness = (depth * 4.0) + abs(offset_z) + 2.0
    solidify.offset = 0.0
    solidify.use_even_offset = True

    depsgraph = bpy.context.evaluated_depsgraph_get()
    solid_mesh = bpy.data.meshes.new_from_object(cutter_obj.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
    old_mesh = cutter_obj.data
    cutter_obj.modifiers.clear()
    cutter_obj.data = solid_mesh
    bpy.data.meshes.remove(old_mesh)
    return cutter_obj


def _clip_mesh_to_faces(mesh, faces, obj_matrix_world, depth, offset_z):
    tmp_obj = bpy.data.objects.new("_tile_group_tmp", mesh)
    bpy.context.collection.objects.link(tmp_obj)
    tmp_obj.matrix_world = obj_matrix_world

    cutter_obj = _build_single_plane_cutter(faces, obj_matrix_world, depth, offset_z)

    bool_mod = tmp_obj.modifiers.new(name="BooleanClip", type='BOOLEAN')
    bool_mod.operation, bool_mod.solver, bool_mod.object, bool_mod.use_self = 'INTERSECT', 'MANIFOLD', cutter_obj, False
    bpy.context.view_layer.objects.active = tmp_obj
    bpy.ops.object.modifier_apply(modifier="BooleanClip")
    bpy.data.objects.remove(cutter_obj, do_unlink=True)

    clipped_mesh = tmp_obj.data
    bpy.data.objects.remove(tmp_obj, do_unlink=True)
    return clipped_mesh


def create_tile_batch(face_data, width, length, depth, rotation_angle, row_offset, staggered_offset, max_random_offset,
                       pattern, width_gap, length_gap, offset_x, offset_y, offset_z, uv_random_seed=0,
                       random_offset_seed=0, flip_mode="BOTH", random_depth=0.0, random_depth_seed=0,
                       custom_obj_name="CustomTileTemplate", obj_matrix_world=None, preserve_uv=False, tiling_axis='BOTH',
                       use_boolean_clip=True):
    templates = {}
    u_options = [False, True] if flip_mode in ["BOTH", "U"] else [False]
    v_options = [False, True] if flip_mode in ["BOTH", "V"] else [False]

    custom_src_obj = bpy.data.objects.get(custom_obj_name) if pattern == "CUSTOM" else None
    src_uv_name = None
    uv_fallback_warning = None
    if preserve_uv and custom_src_obj is not None:
        active_uv = custom_src_obj.data.uv_layers.active
        if active_uv:
            src_uv_name = active_uv.name
        else:
            uv_fallback_warning = (
                f"'{custom_src_obj.name}' has no UV map -- Preserve Source UVs "
                f"fell back to automatic box-projected UVs."
            )

    for flip_u in u_options:
        for flip_v in v_options:
            bm = bmesh.new()
            if custom_src_obj is not None:
                src_obj = custom_src_obj
                bm.from_mesh(src_obj.data)
            else:
                bmesh.ops.create_cube(bm, size=1.0)

            uv_layer = bm.loops.layers.uv.get("UVMap") or bm.loops.layers.uv.new("UVMap")
            src_uv_layer = bm.loops.layers.uv.get(src_uv_name) if src_uv_name else None

            for f in bm.faces:
                is_top = f.normal.z > 0.9
                is_bot = f.normal.z < -0.9
                for loop in f.loops:
                    if src_uv_layer is not None:
                        src_u, src_v = loop[src_uv_layer].uv
                        u = (1.0 - src_u) if flip_u else src_u
                        v = (1.0 - src_v) if flip_v else src_v
                    else:
                        co = loop.vert.co
                        u = (1.0 - (co.x + 0.5)) if flip_u else (co.x + 0.5)
                        v = (1.0 - (co.y + 0.5)) if flip_v else (co.y + 0.5)
                        if not (is_top or is_bot):
                            u = (co.y + 0.5) if abs(f.normal.x) > 0.5 else (co.x + 0.5)
                            v = co.z + 0.5
                    loop[uv_layer].uv = (u, v)
            templates[(flip_u, flip_v)] = bm

    template_z_mid = {}
    for key, t_bm in templates.items():
        zs = [v.co.z for v in t_bm.verts]
        z_min, z_max = min(zs), max(zs)
        template_z_mid[key] = (z_min + z_max) * 0.5

    master_bm = bmesh.new()
    m_uv = master_bm.loops.layers.uv.get("UVMap") or master_bm.loops.layers.uv.new("UVMap")
    m_tile_normal, m_axis_u, m_axis_v = [master_bm.faces.layers.float_vector.new(n) for n in ["tile_normal", "tile_axis_u", "tile_axis_v"]]
    m_tile_col = master_bm.faces.layers.float.new("tile_col")
    m_tile_row = master_bm.faces.layers.float.new("tile_row")

    groups = group_faces_by_plane(face_data)
    rot_rad = math.radians(rotation_angle)
    depth_rng = random.Random(random_depth_seed)
    tile_idx = 0

    for group_key, group_val in groups.items():
        group_matrices = get_placement_matrices(
            pattern, {group_key: group_val}, width, length, depth, rot_rad, row_offset, staggered_offset,
            max_random_offset, width_gap, length_gap, offset_x, offset_y, offset_z,
            random_offset_seed, random_depth, random_depth_seed, custom_obj_name=custom_obj_name, tiling_axis=tiling_axis
        )

        group_bm = bmesh.new()
        g_uv = group_bm.loops.layers.uv.get("UVMap") or group_bm.loops.layers.uv.new("UVMap")
        g_tile_normal, g_axis_u, g_axis_v = [group_bm.faces.layers.float_vector.new(n) for n in ["tile_normal", "tile_axis_u", "tile_axis_v"]]
        g_tile_col = group_bm.faces.layers.float.new("tile_col")
        g_tile_row = group_bm.faces.layers.float.new("tile_row")

        for final_mat, normal, axis_u, axis_v, tile_col_val, tile_row_val in group_matrices:
            f_u = bool(_fast_tile_bool(uv_random_seed, tile_idx, 1)) if flip_mode in ["BOTH", "U"] else False
            f_v = bool(_fast_tile_bool(uv_random_seed, tile_idx, 2)) if flip_mode in ["BOTH", "V"] else False
            tile_idx += 1
            template_bm = templates[(f_u, f_v)]
            t_uv = template_bm.loops.layers.uv.active

            tile_depth_offset = depth_rng.uniform(0.0, random_depth) if random_depth > 0.0 else 0.0
            normal_vec = mathutils.Vector(normal)
            z_mid = template_z_mid[(f_u, f_v)]

            vert_map = {}
            for v in template_bm.verts:
                world_co = final_mat @ v.co
                if v.co.z > z_mid:
                    world_co = world_co + normal_vec * tile_depth_offset
                vert_map[v] = group_bm.verts.new(world_co)

            for f_tmp in template_bm.faces:
                new_f = group_bm.faces.new([vert_map[v] for v in f_tmp.verts])
                for li, loop in enumerate(f_tmp.loops):
                    new_f.loops[li][g_uv].uv = loop[t_uv].uv
                new_f[g_tile_normal], new_f[g_axis_u], new_f[g_axis_v] = normal, axis_u, axis_v
                new_f[g_tile_col], new_f[g_tile_row] = float(tile_col_val), float(tile_row_val)

        g_orig_face = group_bm.faces.layers.float.new("_orig_tile_face")
        for f in group_bm.faces:
            f[g_orig_face] = 1.0

        group_mesh = bpy.data.meshes.new("_tile_group_tmp")
        group_bm.to_mesh(group_mesh)
        group_bm.free()

        if use_boolean_clip:
            clipped_mesh = _clip_mesh_to_faces(group_mesh, group_val["faces"], obj_matrix_world, depth, offset_z)
            master_bm.from_mesh(clipped_mesh)
            bpy.data.meshes.remove(clipped_mesh)
        else:
            master_bm.from_mesh(group_mesh)
            bpy.data.meshes.remove(group_mesh)

    for bm in templates.values():
        bm.free()
    me_batch = bpy.data.meshes.new("BatchTile")
    master_bm.to_mesh(me_batch)
    master_bm.free()
    return me_batch, uv_fallback_warning


def _tile_matches_pattern(col, row, mode, step_x, offset_x, step_y, offset_y, invert):
    """
    Shared match test for 'is this real tile (identified by its col/row grid
    index) picked by this pattern'. Used both by the interactive Select
    Tiles by Pattern operator and by the persisted material rules, so the
    two always agree on what a given pattern selects.
    """
    if mode == 'CHECKER':
        block_col = math.floor((col - offset_x) / step_x)
        block_row = math.floor((row - offset_y) / step_y)
        parity = (block_col + block_row) % 2
        return (parity == 1) if invert else (parity == 0)
    return ((col - offset_x) % step_x == 0) and ((row - offset_y) % step_y == 0)


def _apply_material_rules(batch_obj):
    """
    Re-applies the persisted pattern -> material rules (settings.material_rules)
    onto the batch mesh. Called automatically after every regeneration so a
    material assigned via a pattern survives live-updates instead of being
    lost when the mesh is rebuilt from scratch (regeneration replaces
    batch_obj.data entirely, so any one-off manual face.material_index
    edit can't survive it -- only a remembered rule can).
    """
    settings = batch_obj.smart_tile_props
    rules = [r for r in settings.material_rules if r.material is not None]
    if not rules:
        return

    mesh = batch_obj.data
    rule_slots = []
    for rule in rules:
        if rule.material.name not in mesh.materials:
            mesh.materials.append(rule.material)
        rule_slots.append(mesh.materials.find(rule.material.name))

    bm = bmesh.new()
    bm.from_mesh(mesh)
    l_col = bm.faces.layers.float.get("tile_col")
    l_row = bm.faces.layers.float.get("tile_row")
    if l_col is None or l_row is None:
        bm.free()
        return

    for f in bm.faces:
        col = round(f[l_col])
        row = round(f[l_row])
        for rule, slot in zip(rules, rule_slots):
            if _tile_matches_pattern(col, row, rule.mode, rule.step_x, rule.offset_x, rule.step_y, rule.offset_y, rule.invert):
                f.material_index = slot

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _finalize_batch_mesh(batch_obj, depth):
    bm_final = bmesh.new()
    bm_final.from_mesh(batch_obj.data)
    
    l_normal = bm_final.faces.layers.float_vector.get("tile_normal")
    if l_normal:
        for f in bm_final.faces:
            if f.normal.dot(mathutils.Vector(f[l_normal])) < 0:
                f.normal_flip()
    bmesh.ops.recalc_face_normals(bm_final, faces=bm_final.faces[:])
    
    settings = batch_obj.smart_tile_props
    attr_names = {
        "top": settings.attr_bevel_top,
        "bottom": settings.attr_bevel_bottom,
        "side": settings.attr_bevel_side,
        "width": settings.attr_width_edge,
        "length": settings.attr_length_edge,
        "boolean": settings.attr_boolean_edge,
    }
    
    set_tile_edge_attributes(bm_final, depth, attr_names)
    
    bm_final.to_mesh(batch_obj.data)
    bm_final.free()
    batch_obj.data.update()


# ---------------------------------------------------------------------------
# PER-OBJECT SETTINGS
# ---------------------------------------------------------------------------

PATTERN_ITEMS = [
    ('STRETCHER', "Stretcher", "Running bond / brick pattern"),
    ('HERRINGBONE', "Herringbone", "Herringbone pattern"),
    ('CHEVRON', "Chevron", "Chevron / V pattern"),
    ('WINDMILL', "Windmill", "Windmill / pinwheel pattern"),
    ('CUSTOM', "Custom Tile", "Use a custom object as the tile"),
]

FLIP_ITEMS = [
    ('BOTH', "Both", "Randomly flip U and V"),
    ('U', "U Only", "Randomly flip U only"),
    ('V', "V Only", "Randomly flip V only"),
    ('NONE', "None", "No random flipping"),
]

TILING_AXIS_ITEMS = [
    ('BOTH', "Both Axes", "Tile along both X and Y axes"),
    ('X', "X Axis Only", "Tile only along the X axis"),
    ('Y', "Y Axis Only", "Tile only along the Y axis"),
]


def _copy_settings(src, dst):
    global _suspend_realtime_update
    prev = _suspend_realtime_update
    _suspend_realtime_update = True
    try:
        for key in src.__annotations__.keys():
            if key in ("is_tile_batch", "source_object", "face_indices", "face_snapshot", "material_rules", "material_rules_index"):
                continue
            setattr(dst, key, getattr(src, key))
    finally:
        _suspend_realtime_update = prev


def _run_deferred_tile_update(obj_name):
    global _suspend_realtime_update
    try:
        obj = bpy.data.objects.get(obj_name)
        if obj is not None and obj.smart_tile_props.is_tile_batch:
            _perform_tile_update(bpy.context, obj)
    finally:
        _suspend_realtime_update = False
    return None


def _trigger_realtime_update(self, context):
    global _suspend_realtime_update
    if _suspend_realtime_update:
        return

    obj = self.id_data
    if not isinstance(obj, bpy.types.Object):
        return
    if not self.is_tile_batch or self.source_object is None:
        return

    _suspend_realtime_update = True
    bpy.app.timers.register(lambda name=obj.name: _run_deferred_tile_update(name), first_interval=0.0)


def _run_deferred_material_rule_update(obj_name):
    global _suspend_material_rule_update
    try:
        obj = bpy.data.objects.get(obj_name)
        if obj is not None and obj.smart_tile_props.is_tile_batch:
            was_edit = (obj.mode == 'EDIT')
            if was_edit:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='OBJECT')
            _apply_material_rules(obj)
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
            wm = bpy.context.window_manager
            if wm:
                for window in wm.windows:
                    for area in window.screen.areas:
                        area.tag_redraw()
    finally:
        _suspend_material_rule_update = False
    return None


def _trigger_material_rule_update(self, context):
    """Update callback for SmartTileMaterialRule fields: re-applies all
    material rules a moment after any rule is edited, so changing a rule's
    pattern or material shows up in real time instead of requiring a manual
    Apply Material Rules click."""
    global _suspend_material_rule_update
    if _suspend_material_rule_update:
        return

    obj = self.id_data
    if not isinstance(obj, bpy.types.Object):
        return
    if not obj.smart_tile_props.is_tile_batch:
        return

    _suspend_material_rule_update = True
    bpy.app.timers.register(lambda name=obj.name: _run_deferred_material_rule_update(name), first_interval=0.0)


def update_pattern_defaults(self, context):
    global _suspend_realtime_update
    prev = _suspend_realtime_update
    _suspend_realtime_update = True

    pattern = self.pattern

    if pattern == 'CUSTOM':
        self.width = 1.0
        self.length = 1.0
        self.depth = 1.0
        self.rotation_angle = 0.0
        self.row_offset = 0.0
        self.staggered_offset = 0.0
        self.max_random_offset = 0.0
        self.random_offset_seed = 5
        self.width_gap = 0.0
        self.length_gap = 0.0
        self.uv_random_seed = 235
        self.flip_mode = 'BOTH'
        self.random_depth = 0.0
        self.random_depth_seed = 20
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.tiling_axis = 'BOTH'

    elif pattern == 'STRETCHER':
        self.width = 2.0
        self.length = 0.2
        self.depth = 0.2
        self.rotation_angle = 0.0
        self.row_offset = 0.5  # Formats out of the box into a perfect standard 50% half-brick lap
        self.staggered_offset = 0.0
        self.max_random_offset = 0.0
        self.random_offset_seed = 5
        self.width_gap = 0.0
        self.length_gap = 0.0
        self.uv_random_seed = 235
        self.flip_mode = 'BOTH'
        self.random_depth = 0.0
        self.random_depth_seed = 42
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.tiling_axis = 'BOTH'

    elif pattern == 'HERRINGBONE':
        self.width = 0.25
        self.length = 1.0
        self.depth = 0.1
        self.rotation_angle = 0.0
        self.row_offset = 0.0
        self.staggered_offset = 0.0
        self.max_random_offset = 0.0
        self.random_offset_seed = 0
        self.width_gap = 0.0
        self.length_gap = 0.0
        self.uv_random_seed = 123
        self.flip_mode = 'BOTH'
        self.random_depth = 0.0
        self.random_depth_seed = 42
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.tiling_axis = 'BOTH'

    elif pattern == 'CHEVRON':
        self.width = 0.2
        self.length = 2.0
        self.depth = 0.1
        self.rotation_angle = 0.0
        self.row_offset = 0.0
        self.staggered_offset = 0.0
        self.max_random_offset = 0.0
        self.random_offset_seed = 0
        self.width_gap = 0.0
        self.length_gap = 0.0
        self.uv_random_seed = 10
        self.flip_mode = 'BOTH'
        self.random_depth = 0.0
        self.random_depth_seed = 42
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.tiling_axis = 'BOTH'

    elif pattern == 'WINDMILL':
        self.width = 0.2
        self.length = 1.0
        self.depth = 0.1
        self.rotation_angle = 0.0
        self.row_offset = 0.0
        self.staggered_offset = 0.0
        self.max_random_offset = 0.0
        self.random_offset_seed = 0
        self.width_gap = 0.0
        self.length_gap = 0.0
        self.uv_random_seed = 123
        self.flip_mode = 'BOTH'
        self.random_depth = 0.0
        self.random_depth_seed = 40
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.tiling_axis = 'BOTH'

    _suspend_realtime_update = prev
    _trigger_realtime_update(self, context)


class SmartTileMaterialRule(PropertyGroup):
    """One persisted 'tiles matching this pattern get this material' rule.
    A list of these lives on SmartTileSettings.material_rules and is
    automatically re-applied after every regeneration (see
    _apply_material_rules), so pattern-based material assignments survive
    live property edits instead of being lost when the mesh is rebuilt."""
    mode: EnumProperty(
        items=[
            ('CHECKER', "Checkerboard", "Alternating blocks of tiles"),
            ('STEP', "Step (X/Y)", "Every Nth tile along X and Y, independently"),
        ],
        name="Mode",
        default='CHECKER',
        update=_trigger_material_rule_update,
    )
    invert: BoolProperty(name="Invert", default=False, update=_trigger_material_rule_update)
    step_x: IntProperty(name="Step X", default=1, min=1, update=_trigger_material_rule_update)
    offset_x: IntProperty(name="Offset X", default=0, update=_trigger_material_rule_update)
    step_y: IntProperty(name="Step Y", default=1, min=1, update=_trigger_material_rule_update)
    offset_y: IntProperty(name="Offset Y", default=0, update=_trigger_material_rule_update)
    material: PointerProperty(type=bpy.types.Material, name="Material", update=_trigger_material_rule_update)


class SmartTileSettings(PropertyGroup):
    is_tile_batch: BoolProperty(default=False)
    source_object: PointerProperty(type=bpy.types.Object)
    face_indices: StringProperty(default="")
    face_snapshot: StringProperty(default="")

    use_boolean_clip: BoolProperty(
        name="Enable Boolean Clip",
        description="Clip generated tiles to face borders. Disable for faster performance",
        default=True,
        update=_trigger_realtime_update
    )

    pattern: EnumProperty(
        items=PATTERN_ITEMS, 
        name="Pattern", 
        default='STRETCHER',
        update=update_pattern_defaults
    )
    tiling_axis: EnumProperty(
        items=TILING_AXIS_ITEMS,
        name="Tiling Axis",
        default='BOTH',
        update=_trigger_realtime_update
    )
    width: FloatProperty(name="Width", default=2.0, min=0.0001, unit='LENGTH', update=_trigger_realtime_update)
    length: FloatProperty(name="Length", default=0.2, min=0.0001, unit='LENGTH', update=_trigger_realtime_update)
    depth: FloatProperty(name="Depth", default=0.2, min=0.0001, unit='LENGTH', update=_trigger_realtime_update)
    rotation_angle: FloatProperty(name="Rotation", default=0.0, subtype='ANGLE', update=_trigger_realtime_update)
    
    row_offset: FloatProperty(
        name="Row Offset",
        default=0.0,
        min=-1.0,
        max=1.0,
        soft_min=-1.0,
        soft_max=1.0,
        update=_trigger_realtime_update,
    )
    staggered_offset: FloatProperty(
        name="Staggered Offset", 
        default=0.0,
        min=-1.0,
        max=1.0,
        soft_min=-1.0,
        soft_max=1.0,
        description="Progressively shift subsequent rows as a fraction of the tile width",
        update=_trigger_realtime_update
    )
    
    max_random_offset: FloatProperty(
        name="Max Random Offset", 
        default=0.0, 
        min=-10.0,
        max=10.0,
        subtype='FACTOR',
        update=_trigger_realtime_update
    )
    width_gap: FloatProperty(name="Width Gap", default=0.0, unit='LENGTH', update=_trigger_realtime_update)
    length_gap: FloatProperty(name="Length Gap", default=0.0, unit='LENGTH', update=_trigger_realtime_update)
    uv_random_seed: IntProperty(name="UV Seed", default=235, update=_trigger_realtime_update)
    random_offset_seed: IntProperty(name="Offset Seed", default=5, update=_trigger_realtime_update)
    flip_mode: EnumProperty(items=FLIP_ITEMS, name="Flip Mode", default='BOTH', update=_trigger_realtime_update)
    random_depth: FloatProperty(name="Random Depth", default=0.0, min=0.0, unit='LENGTH', update=_trigger_realtime_update)
    random_depth_seed: IntProperty(name="Depth Seed", default=42, update=_trigger_realtime_update)
    offset_x: FloatProperty(name="Offset X", default=0.0, unit='LENGTH', update=_trigger_realtime_update)
    offset_y: FloatProperty(name="Offset Y", default=0.0, unit='LENGTH', update=_trigger_realtime_update)
    offset_z: FloatProperty(name="Offset Z", default=0.0, unit='LENGTH', update=_trigger_realtime_update)
    attr_bevel_top: StringProperty(name="Top Bevel", default="bevel_top", update=_trigger_realtime_update)
    attr_bevel_bottom: StringProperty(name="Bottom Bevel", default="bevel_bottom", update=_trigger_realtime_update)
    attr_bevel_side: StringProperty(name="Side Bevel", default="bevel_side", update=_trigger_realtime_update)
    attr_width_edge: StringProperty(name="Width Edge", default="top_width_edge", update=_trigger_realtime_update)
    attr_length_edge: StringProperty(name="Length Edge", default="top_length_edge", update=_trigger_realtime_update)
    attr_boolean_edge: StringProperty(name="Boolean Cut Edge", default="boolean_edge", update=_trigger_realtime_update)

    def _poll_custom_tile_object(self, obj):
        return obj.type == 'MESH' and not obj.smart_tile_props.is_tile_batch

    custom_tile_object: PointerProperty(
        type=bpy.types.Object,
        name="Custom Tile Object",
        update=_trigger_realtime_update,
        description="Mesh object to use as the repeating tile",
        poll=_poll_custom_tile_object,
    )
    preserve_custom_uv: BoolProperty(
        name="Preserve Source UVs",
        default=False,
        update=_trigger_realtime_update,
        description=(
            "Keep the Custom Tile Object's own UV mapping instead of the "
            "automatic box-projected UVs. Per-tile random UV mirroring "
            "(Flip Mode) still applies on top of the preserved UVs. Falls "
            "back to box-projected UVs if the object has no UV map"
        ),
    )

    select_mode: EnumProperty(
        items=[
            ('CHECKER', "Checkerboard", "Select alternating blocks of tiles (block size/shift set by Step/Offset below)"),
            ('STEP', "Step (X/Y)", "Select every Nth tile along X and every Nth tile along Y, independently"),
        ],
        name="Select Mode",
        default='CHECKER',
        description="How tiles are picked when using Select Tiles by Pattern",
    )
    select_checker_invert: BoolProperty(
        name="Invert",
        default=False,
        description="Select the other half of the checkerboard",
    )
    select_step_x: IntProperty(name="Step X", default=1, min=1, description="Step mode: select every Nth column. Checker mode: checker block width in tiles")
    select_offset_x: IntProperty(name="Offset X", default=0, description="Shifts the pattern along X (columns)")
    select_step_y: IntProperty(name="Step Y", default=1, min=1, description="Step mode: select every Nth row. Checker mode: checker block height in tiles")
    select_offset_y: IntProperty(name="Offset Y", default=0, description="Shifts the pattern along Y (rows)")

    material_rules: CollectionProperty(type=SmartTileMaterialRule)
    material_rules_index: IntProperty(default=0)

# ---------------------------------------------------------------------------
# OPERATORS
# ---------------------------------------------------------------------------

class SMARTTILE_OT_generate(Operator):
    """Generate a new tile pattern from the selected faces (Edit Mode)"""
    bl_idname = "object.smart_tile_generate"
    bl_label = "Generate Tile Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and obj.mode == 'EDIT'
    
    def execute(self, context):
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']

        if not selected_meshes:
            self.report({'WARNING'}, "No mesh objects selected")
            return {'CANCELLED'}

        generated_batches = []

        for obj in selected_meshes:
            if obj.mode != 'EDIT':
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

            settings = obj.smart_tile_props
            bm_src = bmesh.from_edit_mesh(obj.data)
            sel_faces = [f for f in bm_src.faces if f.select]

            if not sel_faces:
                continue
            
            if settings.pattern == 'CUSTOM' and settings.custom_tile_object is None:
                self.report({'WARNING'}, f"Pick a Custom Tile Object for {obj.name}")
                continue

            face_indices = [f.index for f in sel_faces]
            face_data = [(f.normal.copy(), [v.co.copy() for v in f.verts]) for f in sel_faces]

            bpy.ops.object.mode_set(mode='OBJECT')

            custom_name = settings.custom_tile_object.name if settings.custom_tile_object else ""
            me_batch, uv_warning = create_tile_batch(
                face_data, settings.width, settings.length, settings.depth, 
                math.degrees(settings.rotation_angle), settings.row_offset, settings.staggered_offset,
                settings.max_random_offset, settings.pattern, settings.width_gap, 
                settings.length_gap, settings.offset_x, settings.offset_y, settings.offset_z, 
                uv_random_seed=settings.uv_random_seed, random_offset_seed=settings.random_offset_seed, 
                flip_mode=settings.flip_mode, random_depth=settings.random_depth, 
                random_depth_seed=settings.random_depth_seed, custom_obj_name=custom_name, 
                obj_matrix_world=obj.matrix_world, preserve_uv=settings.preserve_custom_uv,
                tiling_axis=settings.tiling_axis,
                use_boolean_clip=settings.use_boolean_clip
            )
            if uv_warning:
                self.report({'WARNING'}, uv_warning)

            batch_obj = bpy.data.objects.new(f"BatchTile_{obj.name}", me_batch)
            context.collection.objects.link(batch_obj)
            batch_obj.matrix_world = obj.matrix_world

            _finalize_batch_mesh(batch_obj, settings.depth)

            batch_obj.smart_tile_props.is_tile_batch = True
            batch_obj.smart_tile_props.source_object = obj
            batch_obj.smart_tile_props.face_indices = ",".join(str(i) for i in face_indices)
            _copy_settings(settings, batch_obj.smart_tile_props)
            batch_obj.smart_tile_props.face_snapshot = encode_face_data_snapshot(face_data)
            
            generated_batches.append(batch_obj)

        bpy.ops.object.select_all(action='DESELECT')
        for batch in generated_batches:
            batch.select_set(True)
        
        if generated_batches:
            context.view_layer.objects.active = generated_batches[-1]

        return {'FINISHED'}

def _perform_tile_update(context, batch_obj):
    """Regenerate an existing tile-batch object from its currently stored
    settings. Returns (success: bool, message: str). Shared by the manual
    'Update Tile Pattern' operator and the realtime property callbacks."""
    settings = batch_obj.smart_tile_props
    src_obj = settings.source_object

    if src_obj is None or src_obj.name not in bpy.data.objects:
        return False, "Source object no longer exists"

    prev_active = context.view_layer.objects.active
    prev_selected_names = {o.name for o in context.view_layer.objects if o.select_get()}

    was_edit_mode = (batch_obj.mode == 'EDIT')
    if was_edit_mode:
        context.view_layer.objects.active = batch_obj
        bpy.ops.object.mode_set(mode='OBJECT')

    old_mesh = batch_obj.data
    old_materials = list(old_mesh.materials)

    modifier_data = []
    for mod in batch_obj.modifiers:
        if mod.name == "BooleanClip":
            continue

        mod_props = {}
        for prop in mod.bl_rna.properties:
            if not prop.is_readonly and not prop.is_skip_save:
                try:
                    mod_props[prop.identifier] = getattr(mod, prop.identifier)
                except (AttributeError, TypeError, ValueError):
                    continue
        modifier_data.append((mod.name, mod.type, mod_props))

    face_data = decode_face_data_snapshot(settings.face_snapshot)
    if face_data is None:
        try:
            idx_list = [int(i) for i in settings.face_indices.split(",") if i]
        except ValueError:
            return False, "Stored face indices are corrupted"

        bm_src = bmesh.new()
        bm_src.from_mesh(src_obj.data)
        bm_src.normal_update()
        bm_src.faces.ensure_lookup_table()
        try:
            sel_faces = [bm_src.faces[i] for i in idx_list]
        except IndexError:
            bm_src.free()
            return False, "Source mesh topology changed"

        face_data = [(f.normal.copy(), [v.co.copy() for v in f.verts]) for f in sel_faces]
        bm_src.free()
        settings.face_snapshot = encode_face_data_snapshot(face_data)

    custom_name = settings.custom_tile_object.name if settings.custom_tile_object else ""
    me_batch, uv_warning = create_tile_batch(
        face_data, settings.width, settings.length, settings.depth,
        math.degrees(settings.rotation_angle), settings.row_offset, settings.staggered_offset, settings.max_random_offset,
        settings.pattern, settings.width_gap, settings.length_gap,
        settings.offset_x, settings.offset_y, settings.offset_z,
        uv_random_seed=settings.uv_random_seed,
        random_offset_seed=settings.random_offset_seed,
        flip_mode=settings.flip_mode,
        random_depth=settings.random_depth,
        random_depth_seed=settings.random_depth_seed,
        custom_obj_name=custom_name,
        obj_matrix_world=src_obj.matrix_world,
        preserve_uv=settings.preserve_custom_uv,
        tiling_axis=settings.tiling_axis,
        use_boolean_clip=settings.use_boolean_clip
    )

    batch_obj.modifiers.clear()
    batch_obj.data = me_batch
    bpy.data.meshes.remove(old_mesh)

    for mat in old_materials:
        me_batch.materials.append(mat)

    _finalize_batch_mesh(batch_obj, settings.depth)
    _apply_material_rules(batch_obj)

    for name, m_type, props in modifier_data:
        new_mod = batch_obj.modifiers.new(name=name, type=m_type)
        for key, val in props.items():
            try:
                setattr(new_mod, key, val)
            except (AttributeError, TypeError, ValueError):
                continue

    for o in context.view_layer.objects:
        o.select_set(o.name in prev_selected_names)
    if prev_active is not None and prev_active.name in bpy.data.objects:
        context.view_layer.objects.active = prev_active

    if was_edit_mode and batch_obj.name in bpy.data.objects:
        context.view_layer.objects.active = batch_obj
        bpy.ops.object.mode_set(mode='EDIT')
        if prev_active is not None and prev_active.name in bpy.data.objects:
            context.view_layer.objects.active = prev_active

    return True, (uv_warning or "")


class SMARTTILE_OT_update(Operator):
    """Regenerate an existing tile pattern using its (edited) remembered settings"""
    bl_idname = "object.smart_tile_update"
    bl_label = "Update Tile Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.smart_tile_props.is_tile_batch
            and obj.smart_tile_props.source_object is not None
        )

    def execute(self, context):
        batch_obj = context.active_object
        ok, message = _perform_tile_update(context, batch_obj)
        if not ok:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}
        if message:
            self.report({'WARNING'}, message)
        return {'FINISHED'}


class MESH_OT_sync_tile_settings(Operator):
    """Synchronize tile settings from the active object to all other selected tile batches"""
    bl_idname = "mesh.sync_tile_settings"
    bl_label = "Sync All Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        active = context.active_object
        return (active is not None and 
                hasattr(active, "smart_tile_props") and 
                active.smart_tile_props.is_tile_batch and 
                len(context.selected_objects) > 1)

    def execute(self, context):
        global _suspend_realtime_update
        source_obj = context.active_object
        src_props = source_obj.smart_tile_props
        
        props_to_sync = [
            "pattern", "tiling_axis", "width", "length", "depth", "rotation_angle",
            "row_offset", "staggered_offset", "max_random_offset", "random_offset_seed",
            "width_gap", "length_gap", "uv_random_seed", "flip_mode",
            "random_depth", "random_depth_seed", "offset_x", "offset_y", "offset_z",
            "attr_bevel_top", "attr_bevel_bottom", "attr_bevel_side", 
            "attr_width_edge", "attr_length_edge", "attr_boolean_edge",
            "use_boolean_clip"
        ]
        if hasattr(src_props, "custom_tile_object"):
            props_to_sync.append("custom_tile_object")
        if hasattr(src_props, "preserve_custom_uv"):
            props_to_sync.append("preserve_custom_uv")
        
        targets = [obj for obj in context.selected_objects 
                   if obj != source_obj and hasattr(obj, "smart_tile_props") and obj.smart_tile_props.is_tile_batch]
        
        if not targets:
            self.report({'WARNING'}, "No other valid tile batches selected")
            return {'CANCELLED'}

        _suspend_realtime_update = True
        try:
            for obj in targets:
                target_props = obj.smart_tile_props
                for prop in props_to_sync:
                    setattr(target_props, prop, getattr(src_props, prop))
                
                _perform_tile_update(context, obj)
        finally:
            _suspend_realtime_update = False
            context.view_layer.update()
        
        self.report({'INFO'}, f"Synced settings to {len(targets)} tiles.")
        return {'FINISHED'}


class SMARTTILE_OT_select_tiles_by_pattern(Operator):
    """Select whole real tiles (not individual faces) on a generated tile batch, by an X/Y pattern"""
    bl_idname = "mesh.smart_tile_select_by_pattern"
    bl_label = "Select Tiles by Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == 'MESH'
            and hasattr(obj, "smart_tile_props")
            and obj.smart_tile_props.is_tile_batch
        )

    def execute(self, context):
        obj = context.active_object
        settings = obj.smart_tile_props

        prev_mode = obj.mode
        if prev_mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        l_col = bm.faces.layers.float.get("tile_col")
        l_row = bm.faces.layers.float.get("tile_row")

        if l_col is None or l_row is None:
            if prev_mode != 'EDIT':
                bpy.ops.object.mode_set(mode=prev_mode)
            self.report({'ERROR'}, "No tile position data on this mesh -- use Force Refresh to regenerate it first")
            return {'CANCELLED'}

        step_x, offset_x = settings.select_step_x, settings.select_offset_x
        step_y, offset_y = settings.select_step_y, settings.select_offset_y
        checker_invert = settings.select_checker_invert

        for f in bm.faces:
            col = round(f[l_col])
            row = round(f[l_row])
            match = _tile_matches_pattern(col, row, settings.select_mode, step_x, offset_x, step_y, offset_y, checker_invert)
            f.select = match

        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)

        if prev_mode != 'EDIT':
            bpy.ops.object.mode_set(mode=prev_mode)

        return {'FINISHED'}


class SMARTTILE_UL_material_rules(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "mode", text="")
        row.prop(item, "material", text="")


class SMARTTILE_OT_add_material_rule(Operator):
    """Add a persistent pattern->material rule, using the current pattern settings above and the object's active material"""
    bl_idname = "object.smart_tile_add_material_rule"
    bl_label = "Add Material Rule"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and hasattr(obj, "smart_tile_props") and obj.smart_tile_props.is_tile_batch

    def execute(self, context):
        global _suspend_material_rule_update
        obj = context.active_object
        settings = obj.smart_tile_props

        _suspend_material_rule_update = True
        try:
            rule = settings.material_rules.add()
            rule.mode = settings.select_mode
            rule.step_x = settings.select_step_x
            rule.offset_x = settings.select_offset_x
            rule.step_y = settings.select_step_y
            rule.offset_y = settings.select_offset_y
            rule.invert = settings.select_checker_invert
            rule.material = obj.active_material
            settings.material_rules_index = len(settings.material_rules) - 1
        finally:
            _suspend_material_rule_update = False

        was_edit = (obj.mode == 'EDIT')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        _apply_material_rules(obj)
        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class SMARTTILE_OT_remove_material_rule(Operator):
    """Remove the selected material rule"""
    bl_idname = "object.smart_tile_remove_material_rule"
    bl_label = "Remove Material Rule"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and hasattr(obj, "smart_tile_props")
            and obj.smart_tile_props.is_tile_batch
            and len(obj.smart_tile_props.material_rules) > 0
        )

    def execute(self, context):
        settings = context.active_object.smart_tile_props
        settings.material_rules.remove(settings.material_rules_index)
        settings.material_rules_index = max(0, settings.material_rules_index - 1)
        return {'FINISHED'}


class SMARTTILE_OT_apply_material_rules(Operator):
    """Re-apply all persisted pattern->material rules onto this tile batch now"""
    bl_idname = "object.smart_tile_apply_material_rules"
    bl_label = "Apply Material Rules"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == 'MESH'
            and hasattr(obj, "smart_tile_props")
            and obj.smart_tile_props.is_tile_batch
        )

    def execute(self, context):
        obj = context.active_object
        was_edit = (obj.mode == 'EDIT')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        _apply_material_rules(obj)
        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}

# ---------------------------------------------------------------------------
# PANEL
# ---------------------------------------------------------------------------

class SMARTTILE_PT_panel(Panel):
    bl_label = "Smart Tile Generator"
    bl_idname = "SMARTTILE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tile Gen"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.smart_tile_props
        is_result = settings.is_tile_batch

        if is_result:
            src_obj = settings.source_object
            src_name = src_obj.name if src_obj else "(missing)"
            
            box = layout.box()
            box.label(text=f"Pattern on: {obj.name}", icon='MOD_ARRAY')
            box.label(text=f"Source: {src_name}")
        else:
            layout.label(text="Select faces in Edit Mode, then Generate", icon='INFO')

        layout.prop(settings, "pattern")
        layout.prop(settings, "tiling_axis")
        layout.prop(settings, "use_boolean_clip", text="Boolean Clip (Performance)")

        if settings.pattern == 'CUSTOM':
            layout.prop(settings, "custom_tile_object")
            if settings.custom_tile_object is None:
                layout.label(text="Pick an object above", icon='ERROR')
            layout.prop(settings, "preserve_custom_uv")

        col = layout.column(align=True)
        if settings.pattern == 'CUSTOM':
            col.label(text="Scale multipliers (1.0 = object's own size):")
        col.prop(settings, "width")
        col.prop(settings, "length")
        col.prop(settings, "depth")

        layout.prop(settings, "rotation_angle")

        if settings.pattern in ('STRETCHER', 'CUSTOM'):
            col = layout.column(align=True)
            col.prop(settings, "row_offset", slider=True)
#            col.prop(settings, "row_offset")
            col.prop(settings, "staggered_offset", slider=True)
            col.prop(settings, "max_random_offset")
            col.prop(settings, "random_offset_seed")

        col = layout.column(align=True)
        col.prop(settings, "width_gap")
        col.prop(settings, "length_gap")

        layout.prop(settings, "flip_mode")
        layout.prop(settings, "uv_random_seed")

        col = layout.column(align=True)
        col.prop(settings, "random_depth")
        col.prop(settings, "random_depth_seed")

        col = layout.column(align=True)
        col.label(text="Pattern Offset")
        col.prop(settings, "offset_x")
        col.prop(settings, "offset_y")
        col.prop(settings, "offset_z")

        box = layout.box()
        box.label(text="Edge Attribute Names:", icon='FILE_TEXT')
        box.prop(settings, "attr_bevel_top")
        box.prop(settings, "attr_bevel_bottom")
        box.prop(settings, "attr_bevel_side")
        box.prop(settings, "attr_width_edge")
        box.prop(settings, "attr_length_edge")
        box.prop(settings, "attr_boolean_edge")
        
        if is_result:
            box = layout.box()
            box.label(text="Select Tiles by Pattern:", icon='SELECT_SET')
            box.prop(settings, "select_mode")
            row = box.row(align=True)
            row.prop(settings, "select_step_x")
            row.prop(settings, "select_offset_x")
            row = box.row(align=True)
            row.prop(settings, "select_step_y")
            row.prop(settings, "select_offset_y")
            if settings.select_mode == 'CHECKER':
                box.prop(settings, "select_checker_invert")
            box.operator("mesh.smart_tile_select_by_pattern", icon='SELECT_SET')

            box = layout.box()
            box.label(text="Pattern -> Material Rules (persists on regen):", icon='MATERIAL')
            row = box.row()
            row.template_list(
                "SMARTTILE_UL_material_rules", "", settings, "material_rules",
                settings, "material_rules_index", rows=3
            )
            col = row.column(align=True)
            col.operator("object.smart_tile_add_material_rule", text="", icon='ADD')
            col.operator("object.smart_tile_remove_material_rule", text="", icon='REMOVE')
            if settings.material_rules:
                idx = min(settings.material_rules_index, len(settings.material_rules) - 1)
                active_rule = settings.material_rules[idx]
                sub = box.column(align=True)
                r = sub.row(align=True)
                r.prop(active_rule, "step_x")
                r.prop(active_rule, "offset_x")
                r = sub.row(align=True)
                r.prop(active_rule, "step_y")
                r.prop(active_rule, "offset_y")
                if active_rule.mode == 'CHECKER':
                    sub.prop(active_rule, "invert")
            box.operator("object.smart_tile_apply_material_rules", icon='FILE_REFRESH')

        layout.separator()
        layout.operator("mesh.sync_tile_settings", icon='COPY_ID')

        layout.separator()
        if is_result:
            layout.operator("object.smart_tile_update", text="Force Refresh", icon='FILE_REFRESH')
            layout.label(text="Settings above update live", icon='CHECKMARK')
        else:
            layout.operator("object.smart_tile_generate", icon='MESH_GRID')


classes = (
    SmartTileMaterialRule,
    SmartTileSettings,
    MESH_OT_sync_tile_settings,
    SMARTTILE_OT_generate,
    SMARTTILE_OT_update,
    SMARTTILE_OT_select_tiles_by_pattern,
    SMARTTILE_UL_material_rules,
    SMARTTILE_OT_add_material_rule,
    SMARTTILE_OT_remove_material_rule,
    SMARTTILE_OT_apply_material_rules,
    SMARTTILE_PT_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.smart_tile_props = PointerProperty(type=SmartTileSettings)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Object.smart_tile_props


if __name__ == "__main__":
    register()
