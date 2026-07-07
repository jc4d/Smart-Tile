bl_info = {
    "name": "Smart Tile Generator",
    "author": "You",
    "version": (1, 0, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > Tile Gen",
    "description": "Procedural tile pattern generator (Stretcher, Herringbone, Chevron, Windmill, Custom) with per-object remembered parameters",
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
)
from bpy.types import PropertyGroup, Operator, Panel

# Guard flag: prevents realtime-update callbacks from re-entering / firing a
# storm of regenerations while we're programmatically setting several
# properties in a row (e.g. when the pattern dropdown changes its defaults,
# or when settings are copied onto a freshly generated batch object).
_suspend_realtime_update = False


# ---------------------------------------------------------------------------
# CORE GEOMETRY FUNCTIONS
# (unchanged logic from the working script, with the previously agreed fixes:
#  - reliable top/bottom split for random_depth using bbox midpoint, not a
#    hardcoded local z > 0 check
#  - CUSTOM tile width/length/depth treated as scale multipliers
#  - CHEVRON padding increased so patterns fully fill selected faces)
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


def set_tile_edge_attributes(bm, depth, attr_names):
    """
    attr_names is a dictionary mapping logical roles to the string names 
    stored in your UI properties.
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

    for edge in bm.edges:
        linked = edge.link_faces
        if not linked: continue
        is_top = is_bot = is_width_edge = is_length_edge = False
        
        if l_normal:
            for f in linked:
                tile_normal = mathutils.Vector(f[l_normal]).normalized()
                dot = f.normal.normalized().dot(tile_normal)
                if dot > 0.9: is_top = True
                elif dot < -0.9: is_bot = True
        
        if is_top and lu and lv:
            cap_face = next((f for f in linked if f.normal.normalized().dot(mathutils.Vector(f[l_normal]).normalized()) > 0.9), None)
            if cap_face:
                axis_u, axis_v = mathutils.Vector(cap_face[lu]), mathutils.Vector(cap_face[lv])
                vec = (edge.verts[0].co - edge.verts[1].co).normalized()
                is_width_edge = abs(vec.dot(axis_u)) > 0.9
                is_length_edge = abs(vec.dot(axis_v)) > 0.9
        
        edge[layers["top"]] = 1.0 if is_top else 0.0
        edge[layers["bottom"]] = 1.0 if is_bot else 0.0
        edge[layers["side"]] = 1.0 if not (is_top or is_bot) else 0.0
        edge[layers["width"]] = 1.0 if is_width_edge else 0.0
        edge[layers["length"]] = 1.0 if is_length_edge else 0.0


def get_stretcher_matrices(groups, width, length, depth, rot_rad, row_offset, max_random_offset,
                            width_gap, length_gap, random_offset_seed, offset_x, offset_y, offset_z):
    all_matrices = []
    stretcher_rng = random.Random(random_offset_seed)
    random_row_shifts = {}
    e_width, e_length = width - width_gap, length - length_gap

    for key, data in groups.items():
        normal, faces = data["normal"], data["faces"]
        axis_u = (mathutils.Vector((0, 0, 1)) if abs(normal.dot(mathutils.Vector((0, 0, 1)))) < 0.9 else mathutils.Vector((0, 1, 0)))
        axis_u = (axis_u - axis_u.dot(normal) * normal).normalized()
        axis_v = normal.cross(axis_u).normalized()
        if rot_rad != 0.0:
            rot_mat_axes = mathutils.Matrix.Rotation(rot_rad, 3, normal)
            axis_u, axis_v = rot_mat_axes @ axis_u, rot_mat_axes @ axis_v

        all_verts = [v for face in faces for v in face]
        anchor = get_pattern_offset(all_verts[0], axis_u, axis_v, normal, offset_x, offset_y, offset_z)
        projected = [((v - all_verts[0]).dot(axis_u), (v - all_verts[0]).dot(axis_v)) for v in all_verts]
        u_min, u_max = min(p[0] for p in projected), max(p[0] for p in projected)
        v_min, v_max = min(p[1] for p in projected), max(p[1] for p in projected)

        u_step, v_step = width + width_gap, length + length_gap
        u_off_step, v_off_step = math.floor(offset_x / u_step), math.floor(offset_y / v_step)
        for i in range(math.floor(u_min / u_step) - 5 - u_off_step, math.ceil(u_max / u_step) + 5 - u_off_step):
            for j in range(math.floor(v_min / v_step) - 5 - v_off_step, math.ceil(v_max / v_step) + 5 - v_off_step):
                if j not in random_row_shifts:
                    random_row_shifts[j] = stretcher_rng.uniform(0, max_random_offset)
                u_pos = (i * u_step) + (width * 0.5) + (row_offset if j % 2 != 0 else 0.0) + random_row_shifts[j]
                v_pos = (j * v_step) + (length * 0.5)
                tile_center = anchor + (u_pos * axis_u) + (v_pos * axis_v)
                all_matrices.append(create_tile_matrix(tile_center, axis_u, axis_v, normal, e_width, e_length, depth))
    return all_matrices


def get_herringbone_matrices(groups, width, length, depth, rot_rad, row_offset, max_random_offset,
                              width_gap, length_gap, offset_x, offset_y, offset_z):
    all_matrices = []
    # Tiles are built with length running along the u-axis and width along the
    # v-axis (see the create_tile_matrix calls below), so length_gap shrinks/
    # shifts along u and width_gap shrinks/shifts along v.
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
        row_min = math.floor(v_min / row_step) - 2 - row_off_step
        row_max = math.ceil(v_max / row_step) + 2 - row_off_step
        col_min = math.floor(u_min / col_step) - 2 - col_off_step
        col_max = math.ceil(u_max / col_step) + 2 - col_off_step

        for row in range(row_min, row_max):
            row_origin = anchor + (row * row_step) * axis_v
            for col in range(col_min, col_max):
                pos_a = row_origin + (col * col_step) * axis_u
                centered_pos_a = pos_a + (offset_u * u_pos) + (offset_v * v_pos)
                all_matrices.append(create_tile_matrix(centered_pos_a, u_pos, v_pos, normal, e_length, e_width, depth))
                pos_b = pos_a + (step_u - step_v) * axis_u + (step_u - step_v) * axis_v
                centered_pos_b = pos_b + (offset_u * u_neg) + (offset_v * v_neg)
                all_matrices.append(create_tile_matrix(centered_pos_b, u_neg, v_neg, normal, e_length, e_width, depth))

    return all_matrices


def get_chevron_matrices(groups, width, length, depth, rot_rad, row_offset, max_random_offset,
                          width_gap, length_gap, offset_x, offset_y, offset_z):
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
        # Chevron places two sheared tiles per cell, one shifted an extra half-row
        # out, so its true footprint reaches further than plain length x width.
        # Use a wider safety margin than the other patterns to avoid unfilled edges.
        pad = 6
        row_min = math.floor(u_min / row_step) - pad - row_off_step
        row_max = math.ceil(u_max / row_step) + pad - row_off_step
        col_min = math.floor(v_min / leg_width) - pad - col_off_step
        col_max = math.ceil(v_max / leg_width) + pad - col_off_step

        for row in range(row_min, row_max):
            for col in range(col_min, col_max):
                # row_step runs along axis_u (the length direction) so it uses
                # length_gap; leg_width runs along axis_v (the width direction)
                # so it uses width_gap. The seam between the two chevron legs
                # sits along the width direction too, so it also uses width_gap.
                origin = anchor + (col * (leg_width + width_gap)) * axis_v + (row * (row_step + length_gap)) * axis_u
                mat, n, u, v = create_tile_matrix(origin, u_pos, v_pos, normal, length, width, depth)
                all_matrices.append((mat @ shear_pos, n, u, v))
                seam_push = (v_pos - v_neg).normalized() * -(width_gap * 0.5)
                mat, n, u, v = create_tile_matrix(origin + correction + (row_step / 2) * axis_u + seam_push, u_neg, v_neg, normal, length, width, depth)
                all_matrices.append((mat @ shear_neg, n, u, v))
    return all_matrices


def get_windmill_matrices(groups, width, length, depth, rot_rad, row_offset, max_random_offset,
                           width_gap, length_gap, offset_x, offset_y, offset_z):
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
        row_min = math.floor(u_min / cell) - 2 - row_off_step
        row_max = math.ceil(u_max / cell) + 2 - row_off_step
        col_min = math.floor(v_min / cell) - 2 - col_off_step
        col_max = math.ceil(v_max / cell) + 2 - col_off_step

        center_gap = (width_gap + length_gap) / 2.0

        for row in range(row_min, row_max):
            for col in range(col_min, col_max):
                origin = anchor + (col * cell) * axis_v + (row * cell) * axis_u
                all_matrices.append(create_tile_matrix(origin, axis_u, axis_v, normal, length - length_gap, width - width_gap, depth))
                all_matrices.append(create_tile_matrix(origin + (length + width) * axis_u, axis_v, -axis_u, normal, length - length_gap, width - width_gap, depth))
                all_matrices.append(create_tile_matrix(origin + width * axis_v + width * axis_u, axis_v, -axis_u, normal, length - length_gap, width - width_gap, depth))
                all_matrices.append(create_tile_matrix(origin + length * axis_v + width * axis_u, axis_u, axis_v, normal, length - length_gap, width - width_gap, depth))
                all_matrices.append(create_tile_matrix(origin + width * axis_v + width * axis_u, axis_u, axis_v, normal, length - width - center_gap, length - width - center_gap, depth))
    return all_matrices


def get_custom_tile_matrices(groups, width, length, depth, rot_rad, row_offset, max_random_offset,
                              width_gap, length_gap, random_offset_seed, offset_x, offset_y, offset_z,
                              random_depth, random_depth_seed, tile_name="CustomTileTemplate"):
    all_matrices = []
    tile_obj = bpy.data.objects.get(tile_name)
    if not tile_obj:
        return all_matrices

    stretcher_rng = random.Random(random_offset_seed)

    t_dim = tile_obj.dimensions
    # width / length / depth act as SCALE MULTIPLIERS of the custom object's own
    # size (1.0 = original size, 2.0 = double, 0.5 = half), not absolute sizes.
    new_w = t_dim.x * width
    new_l = t_dim.y * length

    u_step = new_w + width_gap
    v_step = new_l + length_gap

    for key, data in groups.items():
        normal, faces = data["normal"], data["faces"]
        axis_u = (mathutils.Vector((0, 0, 1)) if abs(normal.dot(mathutils.Vector((0, 0, 1)))) < 0.9 else mathutils.Vector((0, 1, 0)))
        axis_u = (axis_u - axis_u.dot(normal) * normal).normalized()
        axis_v = normal.cross(axis_u).normalized()

        if rot_rad != 0.0:
            rot_mat_axes = mathutils.Matrix.Rotation(rot_rad, 3, normal)
            axis_u, axis_v = rot_mat_axes @ axis_u, rot_mat_axes @ axis_v

        all_verts = [v for face in faces for v in face]
        anchor = get_pattern_offset(all_verts[0], axis_u, axis_v, normal, offset_x, offset_y, offset_z)

        projected = [((v - all_verts[0]).dot(axis_u), (v - all_verts[0]).dot(axis_v)) for v in all_verts]
        u_min, u_max = min(p[0] for p in projected), max(p[0] for p in projected)
        v_min, v_max = min(p[1] for p in projected), max(p[1] for p in projected)

        for i in range(math.floor((u_min - offset_x) / u_step) - 2, math.ceil((u_max - offset_x) / u_step) + 2):
            for j in range(math.floor((v_min - offset_y) / v_step) - 2, math.ceil((v_max - offset_y) / v_step) + 2):

                random_off = stretcher_rng.uniform(0, max_random_offset)
                shift = row_offset if j % 2 != 0 else 0.0

                center_pos = anchor + ((i * u_step) + shift + random_off + (new_w * 0.5)) * axis_u + ((j * v_step) + (new_l * 0.5)) * axis_v

                scale_mat = mathutils.Matrix.Diagonal((width, length, depth, 1.0))
                rot_mat = mathutils.Matrix((axis_u, axis_v, normal)).transposed().to_4x4()
                trans = mathutils.Matrix.Translation(center_pos)

                all_matrices.append((trans @ rot_mat @ scale_mat, normal, axis_u, axis_v))

    return all_matrices


def create_tile_matrix(pos, u_dir, v_dir, normal, l, w, d, z_offset=0.0):
    rot_mat = mathutils.Matrix((u_dir, v_dir, normal)).transposed().to_4x4()
    scale_mat = mathutils.Matrix.Diagonal((l, w, d, 1.0))
    center = pos + (l * 0.5) * u_dir + (w * 0.5) * v_dir + (normal * z_offset)
    return (mathutils.Matrix.Translation(center + normal * (d * 0.5)) @ rot_mat @ scale_mat, normal, u_dir, v_dir)


def get_placement_matrices(pattern, groups, width, length, depth, rot_rad, row_offset, max_random_offset,
                            width_gap, length_gap, offset_x, offset_y, offset_z, random_offset_seed,
                            random_depth, random_depth_seed, custom_obj_name="CustomTileTemplate"):
    args = (groups, width, length, depth, rot_rad, row_offset, max_random_offset, width_gap, length_gap, offset_x, offset_y, offset_z)

    if pattern == "HERRINGBONE":
        return get_herringbone_matrices(*args)
    if pattern == "CHEVRON":
        return get_chevron_matrices(*args)
    if pattern == "WINDMILL":
        return get_windmill_matrices(*args)
    if pattern == "CUSTOM":
        return get_custom_tile_matrices(
            groups, width, length, depth, rot_rad, row_offset, max_random_offset,
            width_gap, length_gap, random_offset_seed, offset_x, offset_y, offset_z,
            random_depth, random_depth_seed, tile_name=custom_obj_name,
        )
    return get_stretcher_matrices(groups, width, length, depth, rot_rad, row_offset, max_random_offset,
                                   width_gap, length_gap, random_offset_seed, offset_x, offset_y, offset_z)


def _build_single_plane_cutter(faces, obj_matrix_world, depth, offset_z, name="_tile_cutter_tmp"):
    """
    Builds a solidified cutter object for ONE flat group of faces only.
    Because every face here shares the same plane, the resulting shell is a
    single straight prism -- there's no other plane involved, so there's
    nothing for Solidify to crease against and no other island for it to
    overlap/self-intersect with. This is what makes the per-group boolean
    below safe to run with the fast MANIFOLD solver.
    """
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
    solidify.thickness = (depth * 4.0) + abs(offset_z) + 0.05
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
    """
    Boolean-intersects `mesh` (a single-plane-group's own tile geometry)
    down to the exact boundary of `faces` (that same group's selected
    source faces). Runs as its own tiny scene object + its own single-plane
    cutter, entirely independent of any other group -- nothing here ever
    sees or interacts with another plane's geometry.
    """
    tmp_obj = bpy.data.objects.new("_tile_group_tmp", mesh)
    bpy.context.collection.objects.link(tmp_obj)
    tmp_obj.matrix_world = obj_matrix_world

    cutter_obj = _build_single_plane_cutter(faces, obj_matrix_world, depth, offset_z)

    bool_mod = tmp_obj.modifiers.new(name="BooleanClip", type='BOOLEAN')
    # MANIFOLD is safe here: a single-plane cutter is a simple, clean, non
    # self-intersecting prism, which is exactly what MANIFOLD is fast and
    # reliable at. There's no other group's geometry in this operation at
    # all, so there's nothing left for it to leak through.
    bool_mod.operation, bool_mod.solver, bool_mod.object, bool_mod.use_self = 'INTERSECT', 'MANIFOLD', cutter_obj, False
    bpy.context.view_layer.objects.active = tmp_obj
    bpy.ops.object.modifier_apply(modifier="BooleanClip")
    bpy.data.objects.remove(cutter_obj, do_unlink=True)

    clipped_mesh = tmp_obj.data
    bpy.data.objects.remove(tmp_obj, do_unlink=True)
    return clipped_mesh


def create_tile_batch(face_data, width, length, depth, rotation_angle, row_offset, max_random_offset,
                       pattern, width_gap, length_gap, offset_x, offset_y, offset_z, uv_random_seed=0,
                       random_offset_seed=0, flip_mode="BOTH", random_depth=0.0, random_depth_seed=0,
                       custom_obj_name="CustomTileTemplate", obj_matrix_world=None):
    templates = {}
    u_options = [False, True] if flip_mode in ["BOTH", "U"] else [False]
    v_options = [False, True] if flip_mode in ["BOTH", "V"] else [False]

    for flip_u in u_options:
        for flip_v in v_options:
            bm = bmesh.new()
            if pattern == "CUSTOM" and bpy.data.objects.get(custom_obj_name):
                src_obj = bpy.data.objects[custom_obj_name]
                bm.from_mesh(src_obj.data)
            else:
                bmesh.ops.create_cube(bm, size=1.0)

            uv_layer = bm.loops.layers.uv.verify()
            for f in bm.faces:
                is_top = f.normal.z > 0.9
                is_bot = f.normal.z < -0.9
                for loop in f.loops:
                    co = loop.vert.co
                    u = (1.0 - (co.x + 0.5)) if flip_u else (co.x + 0.5)
                    v = (1.0 - (co.y + 0.5)) if flip_v else (co.y + 0.5)
                    if is_top or is_bot:
                        loop[uv_layer].uv = (u, v)
                    else:
                        u_coord = (co.y + 0.5) if abs(f.normal.x) > 0.5 else (co.x + 0.5)
                        loop[uv_layer].uv = (u_coord, co.z + 0.5)
            templates[(flip_u, flip_v)] = bm

    template_z_mid = {}
    for key, t_bm in templates.items():
        zs = [v.co.z for v in t_bm.verts]
        z_min, z_max = min(zs), max(zs)
        template_z_mid[key] = (z_min + z_max) * 0.5

    master_bm = bmesh.new()
    m_uv = master_bm.loops.layers.uv.verify()
    m_tile_normal, m_axis_u, m_axis_v = [master_bm.faces.layers.float_vector.new(n) for n in ["tile_normal", "tile_axis_u", "tile_axis_v"]]

    groups = group_faces_by_plane(face_data)
    rot_rad = math.radians(rotation_angle)
    depth_rng = random.Random(random_depth_seed)
    tile_idx = 0  # global counter, kept across groups so uv_random_seed behavior is unchanged

    for group_key, group_val in groups.items():
        group_matrices = get_placement_matrices(
            pattern, {group_key: group_val}, width, length, depth, rot_rad, row_offset,
            max_random_offset, width_gap, length_gap, offset_x, offset_y, offset_z,
            random_offset_seed, random_depth, random_depth_seed, custom_obj_name=custom_obj_name,
        )

        # Build this group's tiles into their OWN bmesh -- nothing from any
        # other plane-group is ever present here.
        group_bm = bmesh.new()
        g_uv = group_bm.loops.layers.uv.verify()
        g_tile_normal, g_axis_u, g_axis_v = [group_bm.faces.layers.float_vector.new(n) for n in ["tile_normal", "tile_axis_u", "tile_axis_v"]]

        for final_mat, normal, axis_u, axis_v in group_matrices:
            tile_uv_rng = random.Random(uv_random_seed + tile_idx)
            tile_idx += 1
            f_u = tile_uv_rng.choice([True, False]) if flip_mode in ["BOTH", "U"] else False
            f_v = tile_uv_rng.choice([True, False]) if flip_mode in ["BOTH", "V"] else False
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
            group_bm.verts.ensure_lookup_table()

            for f_tmp in template_bm.faces:
                new_f = group_bm.faces.new([vert_map[v] for v in f_tmp.verts])
                for li, loop in enumerate(f_tmp.loops):
                    new_f.loops[li][g_uv].uv = loop[t_uv].uv
                new_f[g_tile_normal], new_f[g_axis_u], new_f[g_axis_v] = normal, axis_u, axis_v

        group_mesh = bpy.data.meshes.new("_tile_group_tmp")
        group_bm.to_mesh(group_mesh)
        group_bm.free()

        # Clip THIS group's tiles to THIS group's own face boundary only --
        # fully independent of every other group, so corners can't leak.
        clipped_mesh = _clip_mesh_to_faces(group_mesh, group_val["faces"], obj_matrix_world, depth, offset_z)

        # Merge the already-clipped group straight into the master bmesh.
        # This is a plain append (matching layers by name), not a boolean --
        # each group arrives pre-clipped, so there's nothing left to combine
        # geometrically.
        master_bm.from_mesh(clipped_mesh)
        bpy.data.meshes.remove(clipped_mesh)

    for bm in templates.values():
        bm.free()
    me_batch = bpy.data.meshes.new("BatchTile")
    master_bm.to_mesh(me_batch)
    master_bm.free()
    return me_batch


def _finalize_batch_mesh(batch_obj, depth):
    """
    Fix normals and (re)write the edge-attribute layers on a generated 
    batch object using the attribute names defined in the object's settings.
    """
    bm_final = bmesh.new()
    bm_final.from_mesh(batch_obj.data)
    
    # 1. Fix Normals
    l_normal = bm_final.faces.layers.float_vector.get("tile_normal")
    if l_normal:
        for f in bm_final.faces:
            if f.normal.dot(mathutils.Vector(f[l_normal])) < 0:
                f.normal_flip()
    bmesh.ops.recalc_face_normals(bm_final, faces=bm_final.faces[:])
    
    # 2. Collect dynamic attribute names from settings
    settings = batch_obj.smart_tile_props
    attr_names = {
        "top": settings.attr_bevel_top,
        "bottom": settings.attr_bevel_bottom,
        "side": settings.attr_bevel_side,
        "width": settings.attr_width_edge,
        "length": settings.attr_length_edge,
    }
    
    # 3. Apply edge attributes using the collected names
    set_tile_edge_attributes(bm_final, depth, attr_names)
    
    # 4. Finalize
    bm_final.to_mesh(batch_obj.data)
    bm_final.free()
    batch_obj.data.update()


# ---------------------------------------------------------------------------
# PER-OBJECT SETTINGS (this is what gives the addon its "memory")
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

# ---------------------------------------------------------------------------
# PER-OBJECT SETTINGS
# ---------------------------------------------------------------------------

def _copy_settings(src, dst):
    """Helper to copy settings from one property group to another."""
    global _suspend_realtime_update
    prev = _suspend_realtime_update
    _suspend_realtime_update = True
    try:
        for key in src.__annotations__.keys():
            if key in ("is_tile_batch", "source_object", "face_indices", "face_snapshot"):
                continue
            setattr(dst, key, getattr(src, key))
    finally:
        _suspend_realtime_update = prev


def _run_deferred_tile_update(obj_name):
    """
    Runs on the next event-loop tick (via bpy.app.timers), NOT inside the
    property update callback that scheduled it. This matters: the per-group
    clip in create_tile_batch creates+deletes a temp object and applies a
    boolean modifier once per plane-group, and doing that repeatedly from
    directly inside an RNA property `update` callback is a known-fragile
    combination in Blender (that callback runs in a restricted context
    without guaranteed full window/depsgraph state) -- it can silently apply
    incorrectly or skip a group, leaving that group's tiles unclipped/
    oversized. Running it here instead gives it a normal, unrestricted
    execution context every time.
    """
    global _suspend_realtime_update
    try:
        obj = bpy.data.objects.get(obj_name)
        if obj is not None and obj.smart_tile_props.is_tile_batch:
            _perform_tile_update(bpy.context, obj)
    finally:
        _suspend_realtime_update = False
    return None  # one-shot timer, don't reschedule


def _trigger_realtime_update(self, context):
    """Property `update` callback: schedule a regeneration (deferred, see
    _run_deferred_tile_update) if this property group belongs to an
    already-generated batch object."""
    global _suspend_realtime_update
    if _suspend_realtime_update:
        return

    obj = self.id_data
    if not isinstance(obj, bpy.types.Object):
        return
    if not self.is_tile_batch or self.source_object is None:
        return

    # Set the guard immediately so rapid-fire property changes (e.g.
    # dragging a slider) don't queue up multiple overlapping regenerations
    # before the first deferred one has even run. _run_deferred_tile_update
    # clears it once that regeneration completes, and will pick up whatever
    # the property values are AT THAT TIME, so the final value always wins.
    _suspend_realtime_update = True
    bpy.app.timers.register(lambda name=obj.name: _run_deferred_tile_update(name), first_interval=0.0)


def update_pattern_defaults(self, context):
    """Callback to set default values when the pattern is changed."""
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

    elif pattern == 'STRETCHER':
        self.width = 2.0
        self.length = 0.2
        self.depth = 0.2
        self.rotation_angle = 0.0
        self.row_offset = 0.0
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

    elif pattern == 'HERRINGBONE':
        self.width = 0.25
        self.length = 1.0
        self.depth = 0.1
        self.rotation_angle = 0.0
        self.row_offset = 0.0
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

    elif pattern == 'CHEVRON':
        self.width = 0.2
        self.length = 2.0
        self.depth = 0.1
        self.rotation_angle = 0.0
        self.row_offset = 0.0
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

    elif pattern == 'WINDMILL':
        self.width = 0.2
        self.length = 1.0
        self.depth = 0.1
        self.rotation_angle = 0.0
        self.row_offset = 0.0
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

    _suspend_realtime_update = prev
    _trigger_realtime_update(self, context)


class SmartTileSettings(PropertyGroup):
    is_tile_batch: BoolProperty(default=False)
    source_object: PointerProperty(type=bpy.types.Object)
    face_indices: StringProperty(default="")
    # JSON snapshot of the exact (normal, verts) face_data captured at
    # Generate time -- see encode_face_data_snapshot / decode_face_data_snapshot.
    # This is the authoritative source for updates: re-deriving face_data
    # from the live mesh on every update (via face_indices + a fresh bmesh)
    # was what made grouping/normals fragile, since a freshly-built bmesh
    # isn't guaranteed to match the live edit-mode bmesh used at Generate
    # time bit-for-bit. Storing the real snapshot once removes that
    # dependency entirely -- every update reuses precisely what Generate saw.
    face_snapshot: StringProperty(default="")

    pattern: EnumProperty(
        items=PATTERN_ITEMS, 
        name="Pattern", 
        default='STRETCHER',
        update=update_pattern_defaults
    )
    width: FloatProperty(name="Width", default=2.0, min=0.0001, unit='LENGTH', update=_trigger_realtime_update)
    length: FloatProperty(name="Length", default=0.2, min=0.0001, unit='LENGTH', update=_trigger_realtime_update)
    depth: FloatProperty(name="Depth", default=0.2, min=0.0001, unit='LENGTH', update=_trigger_realtime_update)
    rotation_angle: FloatProperty(name="Rotation", default=0.0, subtype='ANGLE', update=_trigger_realtime_update)
    row_offset: FloatProperty(name="Row Offset", default=0.0, unit='LENGTH', update=_trigger_realtime_update)
    max_random_offset: FloatProperty(name="Max Random Offset", default=1.0, min=0.0, unit='LENGTH', update=_trigger_realtime_update)
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

    def _poll_custom_tile_object(self, obj):
        return obj.type == 'MESH' and not obj.smart_tile_props.is_tile_batch

    custom_tile_object: PointerProperty(
        type=bpy.types.Object,
        name="Custom Tile Object",
        update=_trigger_realtime_update,
        description="Mesh object to use as the repeating tile",
        poll=_poll_custom_tile_object,
    )

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
        # 1. Identify all valid, selected mesh objects
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        
        if not selected_meshes:
            self.report({'WARNING'}, "No mesh objects selected")
            return {'CANCELLED'}

        # 2. Iterate through each selected object
        for obj in selected_meshes:
            # Ensure the object is in EDIT mode to get face data
            if obj.mode != 'EDIT':
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
            
            settings = obj.smart_tile_props
            bm_src = bmesh.from_edit_mesh(obj.data)
            sel_faces = [f for f in bm_src.faces if f.select]
            
            if not sel_faces:
                continue # Skip objects with no selected faces

            if settings.pattern == 'CUSTOM' and settings.custom_tile_object is None:
                self.report({'WARNING'}, f"Pick a Custom Tile Object for {obj.name}")
                continue

            face_indices = [f.index for f in sel_faces]
            face_data = [(f.normal.copy(), [v.co.copy() for v in f.verts]) for f in sel_faces]

            # Move to object mode to generate the batch
            bpy.ops.object.mode_set(mode='OBJECT')

            custom_name = settings.custom_tile_object.name if settings.custom_tile_object else ""
            me_batch = create_tile_batch(
                face_data, settings.width, settings.length, settings.depth,
                math.degrees(settings.rotation_angle), settings.row_offset, settings.max_random_offset,
                settings.pattern, settings.width_gap, settings.length_gap,
                settings.offset_x, settings.offset_y, settings.offset_z,
                uv_random_seed=settings.uv_random_seed,
                random_offset_seed=settings.random_offset_seed,
                flip_mode=settings.flip_mode,
                random_depth=settings.random_depth,
                random_depth_seed=settings.random_depth_seed,
                custom_obj_name=custom_name,
                obj_matrix_world=obj.matrix_world,
            )

            batch_obj = bpy.data.objects.new(f"BatchTile_{obj.name}", me_batch)
            context.collection.objects.link(batch_obj)
            batch_obj.matrix_world = obj.matrix_world

            _finalize_batch_mesh(batch_obj, settings.depth)

            # Store the settings and metadata
            batch_obj.smart_tile_props.is_tile_batch = True
            batch_obj.smart_tile_props.source_object = obj
            batch_obj.smart_tile_props.face_indices = ",".join(str(i) for i in face_indices)
            _copy_settings(settings, batch_obj.smart_tile_props)
            batch_obj.smart_tile_props.face_snapshot = encode_face_data_snapshot(face_data)

            # Keep the original object selected
            batch_obj.select_set(False)
            obj.select_set(True)

        return {'FINISHED'}

def _perform_tile_update(context, batch_obj):
    """Regenerate an existing tile-batch object from its currently stored
    settings. Returns (success: bool, message: str). Shared by the manual
    'Update Tile Pattern' operator and the realtime property callbacks."""
    settings = batch_obj.smart_tile_props
    src_obj = settings.source_object

    if src_obj is None or src_obj.name not in bpy.data.objects:
        return False, "Source object no longer exists"

    # 1. PRESERVE EXISTING STATE
    # create_tile_batch below creates + deletes a temp object per plane-group
    # and repeatedly reassigns view_layer.objects.active while clipping each
    # one. Nothing else resets it afterwards on this path (unlike the
    # Generate operator, which explicitly restores selection at the end), so
    # capture it now and restore it before returning.
    prev_active = context.view_layer.objects.active
    prev_selected_names = {o.name for o in context.view_layer.objects if o.select_get()}

    old_mesh = batch_obj.data
    old_materials = list(old_mesh.materials)

    # Capture all modifiers
    modifier_data = []
    for mod in batch_obj.modifiers:
        if mod.name == "BooleanClip":
            continue

        # Capture properties via RNA
        mod_props = {}
        for prop in mod.bl_rna.properties:
            if not prop.is_readonly and not prop.is_skip_save:
                try:
                    mod_props[prop.identifier] = getattr(mod, prop.identifier)
                except (AttributeError, TypeError, ValueError):
                    continue
        modifier_data.append((mod.name, mod.type, mod_props))

    # 2. GENERATE NEW MESH
    # Prefer the exact snapshot captured at Generate time -- this is what
    # actually fixes the update-only grouping/normal glitches, since it means
    # update never has to re-derive face_data from the live mesh at all, and
    # therefore can't disagree with what Generate originally saw. Only fall
    # back to re-deriving from the mesh (via the stored face indices) for
    # objects generated before this snapshot existed.
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
        # Backfill the snapshot so subsequent updates use it directly.
        settings.face_snapshot = encode_face_data_snapshot(face_data)

    custom_name = settings.custom_tile_object.name if settings.custom_tile_object else ""
    me_batch = create_tile_batch(
        face_data, settings.width, settings.length, settings.depth,
        math.degrees(settings.rotation_angle), settings.row_offset, settings.max_random_offset,
        settings.pattern, settings.width_gap, settings.length_gap,
        settings.offset_x, settings.offset_y, settings.offset_z,
        uv_random_seed=settings.uv_random_seed,
        random_offset_seed=settings.random_offset_seed,
        flip_mode=settings.flip_mode,
        random_depth=settings.random_depth,
        random_depth_seed=settings.random_depth_seed,
        custom_obj_name=custom_name,
        obj_matrix_world=src_obj.matrix_world,
    )

    # 3. SWAP DATA
    # Clear modifiers only after copying their state
    batch_obj.modifiers.clear()
    batch_obj.data = me_batch
    bpy.data.meshes.remove(old_mesh)

    # 4. RESTORE MATERIALS
    for mat in old_materials:
        me_batch.materials.append(mat)

    # 5. RE-APPLY LOGIC AND RESTORE MODIFIERS
    _finalize_batch_mesh(batch_obj, settings.depth)

    # Re-apply modifiers in original order
    for name, m_type, props in modifier_data:
        new_mod = batch_obj.modifiers.new(name=name, type=m_type)
        for key, val in props.items():
            try:
                setattr(new_mod, key, val)
            except (AttributeError, TypeError, ValueError):
                continue

    # Restore selection/active state to what it was before this update --
    # undoes any lingering effect of the per-group temp objects used during
    # clipping.
    for o in context.view_layer.objects:
        o.select_set(o.name in prev_selected_names)
    if prev_active is not None and prev_active.name in bpy.data.objects:
        context.view_layer.objects.active = prev_active

    return True, ""


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
            "pattern", "width", "length", "depth", "rotation_angle",
            "row_offset", "max_random_offset", "random_offset_seed",
            "width_gap", "length_gap", "uv_random_seed", "flip_mode",
            "random_depth", "random_depth_seed", "offset_x", "offset_y", "offset_z"
        ]
        if hasattr(src_props, "custom_tile_object"):
            props_to_sync.append("custom_tile_object")
        
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
                
                # IMPORTANT: Clear the stale snapshot so _perform_tile_update 
                # is forced to re-derive the face data from the live source mesh
                target_props.face_snapshot = "" 
                
                if target_props.pattern == 'CUSTOM' and hasattr(target_props, 'update_pattern_defaults'):
                    target_props.update_pattern_defaults(context)
            
            for obj in targets:
                context.view_layer.objects.active = obj
                _perform_tile_update(context, obj)
                
            context.view_layer.objects.active = source_obj
        finally:
            _suspend_realtime_update = False
        
        self.report({'INFO'}, f"Synced and refreshed {len(targets)} tiles.")
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
            # Only define and use these variables if it's a batch result
            src_obj = settings.source_object
            src_name = src_obj.name if src_obj else "(missing)"
            
            box = layout.box()
            box.label(text=f"Pattern on: {obj.name}", icon='MOD_ARRAY')
            box.label(text=f"Source: {src_name}")
        else:
            layout.label(text="Select faces in Edit Mode, then Generate", icon='INFO')

        layout.prop(settings, "pattern")

        if settings.pattern == 'CUSTOM':
            layout.prop(settings, "custom_tile_object")
            if settings.custom_tile_object is None:
                layout.label(text="Pick an object above", icon='ERROR')

        col = layout.column(align=True)
        if settings.pattern == 'CUSTOM':
            col.label(text="Scale multipliers (1.0 = object's own size):")
        col.prop(settings, "width")
        col.prop(settings, "length")
        col.prop(settings, "depth")

        layout.prop(settings, "rotation_angle")

        if settings.pattern == 'STRETCHER':
            col = layout.column(align=True)
            col.prop(settings, "row_offset")
            col.prop(settings, "max_random_offset")
            col.prop(settings, "random_offset_seed")

        if settings.pattern == 'CUSTOM':
            layout.prop(settings, "max_random_offset")
            layout.prop(settings, "row_offset")
            layout.prop(settings, "random_offset_seed")

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

        # Attribute Name Sub-panel
        box = layout.box()
        box.label(text="Edge Attribute Names:", icon='FILE_TEXT')
        box.prop(settings, "attr_bevel_top")
        box.prop(settings, "attr_bevel_bottom")
        box.prop(settings, "attr_bevel_side")
        box.prop(settings, "attr_width_edge")
        box.prop(settings, "attr_length_edge")
        
        layout.separator()
        layout.operator("mesh.sync_tile_settings", icon='COPY_ID')

        layout.separator()
        if is_result:
            layout.operator("object.smart_tile_update", text="Force Refresh", icon='FILE_REFRESH')
            layout.label(text="Settings above update live", icon='CHECKMARK')
        else:
            layout.operator("object.smart_tile_generate", icon='MESH_GRID')


# ---------------------------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------------------------

classes = (
    SmartTileSettings,
    MESH_OT_sync_tile_settings,
    SMARTTILE_OT_generate,
    SMARTTILE_OT_update,
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
