"""
Blender script: low-poly, roughly polished stone cube (fits a 1x1x1 bounding box).

Run from Blender's Scripting tab, or headless:
    blender --background --python stone_cube.py
"""

import os
import random

import bmesh
import bpy
from mathutils import Vector, noise

# --- Parameters -------------------------------------------------------------
SEED = 7              # change for a different stone
SIZE = 1.0            # final bounding-box edge length
RESOLUTION = 8        # quads along each cube edge (triangles = 12 * RESOLUTION^2)
BEVEL = 0.15          # radius of worn, rounded edges (relative to SIZE)
ROUNDNESS = 0.12      # 0 = box, 1 = sphere; bulges faces, softens corners
NOISE_AMP = 0.03      # smooth, large-scale surface undulation
NOISE_SCALE = 2.5     # frequency of that undulation
JITTER = 0.008        # small per-vertex randomness for a chipped look
TRIANGULATE = True    # crisp low-poly facets
SMOOTHING = 1.0       # shading across polygon edges (0 = faceted, 1 = fully smooth)
NAME = "StoneCube"
# Texture split in two halves: top half -> front side (-Y), bottom half -> the
# other five sides. Relative paths resolve next to the .blend file, this
# script, or the current working directory.
TEXTURE_PATH = "block-1-gray.png"
UV_MARGIN = 0.01      # inset per half, avoids colour bleeding across the split
# GLB output path; relative paths resolve next to the saved .blend file, or
# the current working directory if unsaved. Set to None to skip export.
EXPORT_PATH = "stone_cube.glb"


def resolve_path(path, must_exist=False):
    if os.path.isabs(path):
        return path
    roots = []
    if bpy.data.filepath:
        roots.append(os.path.dirname(bpy.data.filepath))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(script_dir):
        roots.append(script_dir)
    roots.append(os.getcwd())
    if must_exist:
        for root in roots:
            if os.path.exists(os.path.join(root, path)):
                return os.path.join(root, path)
        raise FileNotFoundError(f"{path} not found in {roots}")
    return os.path.join(roots[0] if bpy.data.filepath else os.getcwd(), path)


def uv_project(bm):
    """Box-project UVs on an axis-aligned unit cube. Front (-Y) uses the top
    half of the texture, upright; all other sides use the bottom half."""
    uv_layer = bm.loops.layers.uv.verify()
    bm.normal_update()
    lo, hi = UV_MARGIN, 1.0 - UV_MARGIN
    for f in bm.faces:
        n = f.normal
        axis = max(range(3), key=lambda i: abs(n[i]))
        sign = 1 if n[axis] > 0 else -1
        front = axis == 1 and sign < 0
        for loop in f.loops:
            x, y, z = loop.vert.co
            if axis == 0:
                u, v = y * sign, z
            elif axis == 1:
                u, v = -x * sign, z
            else:
                u, v = x, y * sign
            u = lo + (u + 0.5) * (hi - lo)
            v = lo + (v + 0.5) * (hi - lo)
            loop[uv_layer].uv = (u, v * 0.5 + (0.5 if front else 0.0))


def build_mesh(rng):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)

    # Even quad grid on every face so the surface can undulate uniformly.
    if RESOLUTION > 1:
        bmesh.ops.subdivide_edges(
            bm, edges=list(bm.edges), cuts=RESOLUTION - 1, use_grid_fill=True
        )

    # UVs are projected while the cube is still flat-sided.
    uv_project(bm)

    # Round the edges: project each vertex onto a rounded box of radius BEVEL.
    inner = 0.5 - BEVEL
    for v in bm.verts:
        core = Vector((max(-inner, min(inner, c)) for c in v.co))
        v.co = core + (v.co - core).normalized() * BEVEL

    # Pull towards a sphere: faces bulge out, corners soften like worn stone.
    radius = 0.5 * 3 ** 0.5 * 0.8
    for v in bm.verts:
        v.co = v.co.lerp(v.co.normalized() * radius, ROUNDNESS)

    # Smooth undulation along the surface normal plus tiny random chips.
    bm.normal_update()
    offset = Vector((rng.uniform(-100, 100) for _ in range(3)))
    for v in bm.verts:
        v.co += v.normal * noise.noise(v.co * NOISE_SCALE + offset) * NOISE_AMP
        v.co += Vector((rng.uniform(-1, 1) for _ in range(3))) * JITTER

    if TRIANGULATE:
        bmesh.ops.triangulate(bm, faces=list(bm.faces))

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))

    # Normalise to an exact SIZE bounding box centred on the origin.
    lo = Vector((min(v.co[i] for v in bm.verts) for i in range(3)))
    hi = Vector((max(v.co[i] for v in bm.verts) for i in range(3)))
    centre = (lo + hi) / 2
    for v in bm.verts:
        for i in range(3):
            v.co[i] = (v.co[i] - centre[i]) / (hi[i] - lo[i]) * SIZE

    mesh = bpy.data.meshes.new(NAME)
    bm.to_mesh(mesh)
    bm.free()

    set_smooth_normals(mesh)
    return mesh


def set_smooth_normals(mesh):
    """Smooth shading across all polygon edges. For 0 < SMOOTHING < 1, custom
    normals blend each face normal towards the averaged vertex normal."""
    for poly in mesh.polygons:
        poly.use_smooth = SMOOTHING > 0
    if SMOOTHING <= 0 or SMOOTHING >= 1:
        return

    vert_normals = [v.vector.copy() for v in mesh.vertex_normals]
    normals = [None] * len(mesh.loops)
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            normals[li] = poly.normal.lerp(vert_normals[vi], SMOOTHING).normalized()
    mesh.normals_split_custom_set(normals)


def build_material():
    mat = bpy.data.materials.new(NAME + "_Mat")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.38  # semi-glossy, "roughly polished"

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(resolve_path(TEXTURE_PATH, must_exist=True),
                                     check_existing=True)
    tex.extension = 'EXTEND'
    tex.location = (-450, 300)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def export_glb(path):
    path = resolve_path(path)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
    )
    print(f"{NAME}: exported to {path}")


def main():
    rng = random.Random(SEED)

    old = bpy.data.objects.get(NAME)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    mesh = build_mesh(rng)
    mesh.materials.append(build_material())

    obj = bpy.data.objects.new(NAME, mesh)
    bpy.context.collection.objects.link(obj)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    print(f"{NAME}: {len(mesh.vertices)} verts, {len(mesh.polygons)} faces, "
          f"dimensions {tuple(round(d, 4) for d in obj.dimensions)}")

    if EXPORT_PATH:
        export_glb(EXPORT_PATH)


if __name__ == "__main__":
    main()
