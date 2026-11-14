"""
Blender script: low-poly, roughly polished stone cube (fits a 1x1x1 bounding box).

Run from Blender's Scripting tab, or headless:
    blender --background --python stone_cube.py
"""

import os
import random

import bmesh
import bpy
import numpy as np
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
ROUGHNESS = 0.8       # surface shine: 0 = polished/glossy, 1 = fully matte
NAME = "StoneCube"
# Texture split in two halves: top half -> front side (-Y), bottom half -> the
# other five sides. Relative paths resolve next to the .blend file, this
# script, or the current working directory.
TEXTURE_PATH = "block-1-gray.png"
UV_MARGIN = 0.01      # inset per half, avoids colour bleeding across the split
# Procedural stone normal map (surface detail only, geometry is unchanged).
# Same layout and UVs as the colour texture (top half -> front, bottom half ->
# the other five sides), since some engines only read TEXCOORD_0. Saved next
# to the export as a PNG.
NORMAL_PATH = "stone_cube_normal.png"
NORMAL_TILE = 128     # pixels per half (image is NORMAL_TILE x 2*NORMAL_TILE)
NORMAL_STRENGTH = 0.6 # baked into the pixels (some engines ignore glTF normal scale)
# Flip the relief (cracks become ridges in spec-correct viewers such as Blender).
# Compensates for kiss3d 0.47, whose WGSL cotangent frame inverts normal maps.
NORMAL_INVERT = True
DETAIL_SCALE = 4.5    # size of grain, pits and cracks (1 = fine, higher = coarser)
GRAIN_DEPTH = 1.2     # grain / mottled bumps (in pixel heights, times DETAIL_SCALE)
PIT_COUNT = 70        # pores per side
CRACKS = (5, 8)       # min/max main cracks per side (branches added randomly)
CRACK_DEPTH = 4.0     # depth of cracks (in pixel heights, times DETAIL_SCALE)
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


# --- Procedural normal map --------------------------------------------------
def value_noise(rs, n, freq):
    """Smooth random noise in [0, 1] on an n x n grid with `freq` cells."""
    grid = rs.random((freq + 1, freq + 1))
    t = np.linspace(0, freq, n, endpoint=False)
    i = t.astype(int)
    f = t - i
    f = f * f * (3 - 2 * f)
    a = grid[i][:, i] * (1 - f)[None, :] + grid[i][:, i + 1] * f[None, :]
    b = grid[i + 1][:, i] * (1 - f)[None, :] + grid[i + 1][:, i + 1] * f[None, :]
    return a * (1 - f)[:, None] + b * f[:, None]


def stamp(field, x, y, radius, depth):
    """Carve a cone-shaped dent (max-combined so overlaps don't deepen)."""
    n = field.shape[0]
    r = int(np.ceil(radius)) + 1
    x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, n)
    y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, n)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    dent = depth * np.clip(1 - d / radius, 0, 1) ** 0.7
    np.maximum(field[y0:y1, x0:x1], dent, out=field[y0:y1, x0:x1])


def feature_scale():
    """Feature size in pixels per unit of DETAIL_SCALE, independent of NORMAL_TILE."""
    return DETAIL_SCALE * NORMAL_TILE / 256


def carve_crack(rs, field, x, y, angle, length, width, depth):
    """Random-walk crack `length` pixels long; wiggles scale with DETAIL_SCALE."""
    n = field.shape[0]
    step_len = max(1.0, 0.6 * feature_scale())
    steps = max(int(length / step_len), 2)
    for step in range(steps):
        t = step / steps
        taper = np.sin(np.pi * min(0.15 + t, 1.0))  # thin at both ends
        angle += rs.normal(0, 0.22)
        dx, dy = np.cos(angle) * step_len, np.sin(angle) * step_len
        # Stamp densely along the segment so the crack walls stay smooth.
        sub = int(np.ceil(step_len / 0.5))
        for k in range(sub):
            stamp(field, x + dx * k / sub, y + dy * k / sub,
                  max(width * taper, 0.6), depth * taper)
        x += dx
        y += dy
        if not (0 <= x < n and 0 <= y < n):
            return
        if steps > 30 and rs.random() < 0.012:
            carve_crack(rs, field, x, y, angle + rs.choice((-1, 1)) * rs.uniform(0.5, 1.2),
                        length * rs.uniform(0.2, 0.45), width * 0.6, depth * 0.6)


def stone_height(rs, n):
    """Height field (in pixel units) of one side: grain, pores and cracks."""
    s = feature_scale()
    grain = sum(value_noise(rs, n, max(2, int(f / s))) * 0.55 ** k
                for k, f in enumerate((4, 8, 16, 32, 64, 128)))
    grit = value_noise(rs, n, max(2, int(n / (1.5 * s))))
    height = (grain / 1.8 + grit * 0.35) * GRAIN_DEPTH * s

    pits = np.zeros((n, n))
    for _ in range(PIT_COUNT):
        stamp(pits, *rs.uniform(0, n, 2), rs.uniform(0.8, 2.5) * s, rs.uniform(0.6, 1.8) * s)

    cracks = np.zeros((n, n))
    for _ in range(rs.integers(CRACKS[0], CRACKS[1] + 1)):
        carve_crack(rs, cracks, *rs.uniform(0, n, 2), rs.uniform(0, 2 * np.pi),
                    rs.uniform(0.3, 0.9) * n, rs.uniform(1.2, 2.2) * s, CRACK_DEPTH * s)
    return height - pits - cracks


def build_normal_map(seed):
    """Tangent-space normals (OpenGL / glTF convention, +Y up) in the colour
    texture's layout: bottom half for the five shared sides, top half for the
    front."""
    rs = np.random.default_rng(seed)
    n = NORMAL_TILE
    atlas = np.zeros((2 * n, n, 4), dtype=np.float32)
    for row in range(2):  # Blender rows go bottom-up: 0 = other sides, 1 = front
        h = stone_height(rs, n)          # rows = v (bottom-up), cols = u
        # Light 3x3 blur rounds off the creases left by the cone-shaped stamps.
        h = sum(np.roll(np.roll(h, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) / 9
        if NORMAL_INVERT:
            h = -h
        dv, du = np.gradient(h)
        normal = np.dstack((-du * NORMAL_STRENGTH, -dv * NORMAL_STRENGTH, np.ones_like(h)))
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        atlas[row * n:(row + 1) * n, :, :3] = normal * 0.5 + 0.5
    atlas[..., 3] = 1.0

    name = NAME + "_Normal"
    old = bpy.data.images.get(name)
    if old:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(name, n, 2 * n, alpha=False)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(atlas.ravel())  # Blender pixels start bottom-left
    img.filepath_raw = resolve_path(NORMAL_PATH)
    img.file_format = 'PNG'
    img.save()
    return img


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


def build_material(normal_image):
    mat = bpy.data.materials.new(NAME + "_Mat")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = ROUGHNESS

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(resolve_path(TEXTURE_PATH, must_exist=True),
                                     check_existing=True)
    tex.extension = 'EXTEND'
    tex.location = (-450, 300)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    # Stone surface detail: normal map on the same UVs as the colour texture.
    # Strength is baked into the pixels, so the node stays at 1.0.
    ntex = nodes.new("ShaderNodeTexImage")
    ntex.image = normal_image
    ntex.extension = 'EXTEND'
    ntex.location = (-450, -150)
    nmap = nodes.new("ShaderNodeNormalMap")
    nmap.space = 'TANGENT'
    nmap.uv_map = "UVMap"
    nmap.inputs["Strength"].default_value = 1.0
    nmap.location = (-180, -150)
    links.new(ntex.outputs["Color"], nmap.inputs["Color"])
    links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
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
    mesh.materials.append(build_material(build_normal_map(SEED)))

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
