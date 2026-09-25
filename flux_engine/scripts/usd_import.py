"""OpenUSD -> editor mesh interchange. Run only with the trusted app runtime.

No source layer is saved or archive extracted. Resolved texture bytes are copied
into content-addressed personal storage so packaged textures survive project saves.
"""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdShade, UsdSkel, UsdLux
from PIL import Image


class Importer:
    def __init__(self, path, cache, time=None, stage=None, local_geometry=False):
        self.local_geometry = local_geometry
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.stage = stage or Usd.Stage.Open(str(Path(path).resolve()), load=Usd.Stage.LoadAll)
        if not self.stage:
            raise ValueError("OpenUSD could not open the stage")
        self.time = Usd.TimeCode(
            self.stage.GetStartTimeCode() if time is None else time
        )
        self.scale = UsdGeom.GetStageMetersPerUnit(self.stage)
        self.z_up = UsdGeom.GetStageUpAxis(self.stage) == UsdGeom.Tokens.z
        self.transforms = UsdGeom.XformCache(self.time)
        self.materials = []
        self.material_ids = {}
        self.meshes = []
        self.lights = []
        self.cameras = []
        self.warnings = set()
        self.texture_cache = {}
        self.point_prototypes = set()
        self.vertex_count = 0
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("metersPerUnit must be positive and finite")
        for error in self.stage.GetCompositionErrors():
            self.warn(f"Composition: {error}")
        if self.stage.HasAuthoredTimeCodeRange():
            start, end = self.stage.GetStartTimeCode(), self.stage.GetEndTimeCode()
            animated = any(attr.GetNumTimeSamples() > 0
                           for prim in self.stage.Traverse() for attr in prim.GetAttributes())
        else:
            start, end, animated = math.inf, -math.inf, False
            for prim in self.stage.Traverse():
                for attr in prim.GetAttributes():
                    samples = attr.GetTimeSamples()
                    if samples:
                        animated = True
                        start, end = min(start, samples[0]), max(end, samples[-1])
            if not animated:
                start = end = 0
        self.animation = dict(start=start, end=end, rate=self.stage.GetTimeCodesPerSecond(),
                              animated=end > start and animated, name="USD stage")

    def warn(self, text):
        self.warnings.add(str(text))

    def persist(self, data, extension):
        digest = hashlib.sha256(data).hexdigest()
        dest = self.cache / (digest + extension)
        if not dest.exists():
            temporary = dest.with_name(dest.name + f".{os.getpid()}.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, dest)
        return str(dest.resolve())

    def image_asset(self, value):
        if not value:
            raise ValueError("Empty texture asset path")
        resolved = value.resolvedPath
        if not resolved:
            resolved = str(Ar.GetResolver().Resolve(value.path))
        if not resolved:
            raise ValueError(f"Unresolved texture: {value.path}")
        asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(resolved))
        if not asset:
            raise ValueError(f"Cannot read texture: {resolved}")
        image = Image.open(io.BytesIO(bytes(asset.GetBuffer()))).convert("RGBA")
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        return image

    def input_value(self, shader, name, default):
        attr = shader.GetInput(name)
        value = attr.Get(self.time) if attr else None
        return value if value is not None else default

    def texture(self, input_attr, color=False, normal=False):
        """Resolve UVTexture through node-graph output forwarding; bake channel/scale/bias."""
        if not input_attr or not input_attr.HasConnectedSource():
            return None
        connected = input_attr.GetConnectedSource()
        visited = set()
        while connected:
            source, output_name, attribute_type = connected
            key = (str(source.GetPath()), str(output_name))
            if key in visited:
                self.warn("Cyclic material connection ignored")
                return None
            visited.add(key)
            shader = UsdShade.Shader(source.GetPrim())
            if shader and shader.GetIdAttr().Get() == "UsdUVTexture":
                break
            forward = (
                source.GetOutput(output_name)
                if attribute_type == UsdShade.AttributeType.Output
                else source.GetInput(output_name)
            )
            connected = forward.GetConnectedSource() if forward else None
        else:
            self.warn(
                f"Unsupported texture network at {input_attr.GetAttr().GetPath()}"
            )
            return None
        cache_key = (str(shader.GetPath()), str(output_name), color, normal)
        if cache_key in self.texture_cache:
            return self.texture_cache[cache_key]
        try:
            image = self.image_asset(self.input_value(shader, "file", None))
            scale = self.input_value(shader, "scale", (1, 1, 1, 1))
            bias = self.input_value(shader, "bias", (0, 0, 0, 0))
            space = str(self.input_value(shader, "sourceColorSpace", "auto"))
            channel = {"r": 0, "g": 1, "b": 2, "a": 3}.get(str(output_name))
            # USD normal maps commonly author scale=2,bias=-1. The editor
            # already decodes [0,1] to [-1,1], so encode the final vector again.
            srgb = space == "sRGB" or (space == "auto" and color)
            remap_normal = normal and (
                tuple(scale[:3]) != (1, 1, 1) or tuple(bias[:3]) != (0, 0, 0)
            )
            # Every channel operation is independent: evaluate 256 possible
            # byte values once, then let Pillow apply the lookup in native code.
            bands = image.split()
            converted = []
            for output_channel in range(4):
                source_channel = (
                    channel
                    if channel is not None and output_channel < 3
                    else output_channel
                )
                table = []
                for byte in range(256):
                    value = byte / 255
                    if srgb and source_channel < 3:
                        value = (
                            value / 12.92
                            if value <= 0.04045
                            else ((value + 0.055) / 1.055) ** 2.4
                        )
                    value = value * scale[source_channel] + bias[source_channel]
                    if output_channel < 3:
                        if remap_normal:
                            value = value * 0.5 + 0.5
                        elif color and not normal:
                            value = (
                                value * 12.92
                                if value <= 0.0031308
                                else 1.055 * max(value, 0) ** (1 / 2.4) - 0.055
                            )
                    table.append(round(max(0, min(1, value)) * 255))
                converted.append(bands[source_channel].point(table))
            image = Image.merge("RGBA", converted)
            st = shader.GetInput("st")
            uv_name, uv_transform = "st", (1.0, 1.0, 0.0, 0.0, 0.0)
            connection = st.GetConnectedSource() if st else None
            visited = set()
            while connection:
                node, output, kind = connection
                if str(node.GetPath()) in visited:
                    break
                visited.add(str(node.GetPath()))
                uv_shader = UsdShade.Shader(node.GetPrim())
                node_type = uv_shader.GetIdAttr().Get() if uv_shader else None
                if node_type == "UsdPrimvarReader_float2":
                    uv_name = str(self.input_value(uv_shader, "varname", "st"))
                    break
                if node_type == "UsdTransform2d":
                    sc = self.input_value(uv_shader, "scale", (1, 1))
                    tr = self.input_value(uv_shader, "translation", (0, 0))
                    rot = self.input_value(uv_shader, "rotation", 0)
                    uv_transform = (*sc, *tr, rot)
                    connection = uv_shader.GetInput("in").GetConnectedSource()
                else:
                    self.warn(f"Unsupported UV input {node.GetPath()}; using st")
                    break
            for axis in ["wrapS", "wrapT"]:
                wrap = str(self.input_value(shader, axis, "repeat"))
                if wrap not in ("repeat", "useMetadata"):
                    self.warn(
                        f"{shader.GetPath()}: {wrap} wrapping is approximated by repeat"
                    )
            result = (image, uv_name, uv_transform)
            self.texture_cache[cache_key] = result
            return result
        except Exception as error:
            self.warn(f"{shader.GetPath()}: {error}")
            return None

    def save_image(self, image):
        data = io.BytesIO()
        image.save(data, format="PNG")
        return self.persist(data.getvalue(), ".png")

    def material(self, prim):
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        key = str(bound.GetPath()) if bound else "display:" + str(prim.GetPath())
        surface_prim = prim.GetParent() if prim.IsA(UsdGeom.Subset) else prim
        gprim = UsdGeom.Gprim(surface_prim)
        double_sided = bool(gprim and gprim.GetDoubleSidedAttr().Get(self.time))
        # A shared USD material can be bound to single- and double-sided meshes.
        if double_sided:
            key += " · Double sided"
        if key in self.material_ids:
            return self.material_ids[key]
        record = dict(
            name=key,
            double_sided=double_sided,
            alpha_cutoff=None,
            diffuse=[0.8, 0.8, 0.8],
            diffuse_texture="",
            normal_texture="",
            roughness_texture="",
            emissive_texture="",
            pbr=dict(
                metallic=0.0, roughness=0.5, opacity=1.0, emissive=[0.0, 0.0, 0.0]
            ),
            uv_name="st",
            uv_transform=[1.0, 1.0, 0.0, 0.0, 0.0],
        )
        shader = bound.ComputeSurfaceSource()[0] if bound else None
        if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface":
            record["diffuse"] = list(
                self.input_value(shader, "diffuseColor", (0.18, 0.18, 0.18))
            )
            cutoff = float(self.input_value(shader, "opacityThreshold", 0.0))
            if cutoff > 0:
                record["alpha_cutoff"] = cutoff
            record["pbr"].update(
                metallic=float(self.input_value(shader, "metallic", 0)),
                roughness=float(self.input_value(shader, "roughness", 0.5)),
                opacity=float(self.input_value(shader, "opacity", 1)),
                emissive=list(self.input_value(shader, "emissiveColor", (0, 0, 0))),
            )
            uv_settings = set()
            maps = {}
            for slot, name in [
                ("diffuse_texture", "diffuseColor"),
                ("normal_texture", "normal"),
                ("emissive_texture", "emissiveColor"),
                ("metallic", "metallic"),
                ("roughness", "roughness"),
                ("opacity", "opacity"),
            ]:
                texture = self.texture(
                    shader.GetInput(name),
                    color=name in ("diffuseColor", "emissiveColor"),
                    normal=name == "normal",
                )
                if texture:
                    image, uv_name, uv_transform = texture
                    maps[slot] = image
                    uv_settings.add((uv_name, uv_transform))
                    if slot in record:
                        record[slot] = self.save_image(image)
                    if slot == "diffuse_texture":
                        record["diffuse"] = [1.0, 1.0, 1.0]
                    if slot == "emissive_texture":
                        record["pbr"]["emissive"] = [1.0, 1.0, 1.0]
            if uv_settings:
                uv_name, uv_transform = sorted(uv_settings)[0]
                record.update(uv_name=uv_name, uv_transform=list(uv_transform))
                if len(uv_settings) > 1:
                    self.warn(
                        f"{key}: multiple UV sets/transforms; viewport uses {uv_name}"
                    )
            if "metallic" in maps or "roughness" in maps:
                size = max(
                    (
                        im.size
                        for name, im in maps.items()
                        if name in ("metallic", "roughness")
                    ),
                    key=lambda size: size[0] * size[1],
                )
                channels = []
                for slot in ("metallic", "roughness"):
                    if slot in maps:
                        channels.append(
                            maps[slot]
                            .resize(size, Image.Resampling.BILINEAR)
                            .getchannel("R")
                        )
                    else:
                        channels.append(
                            Image.new(
                                "L",
                                size,
                                round(max(0, min(1, record["pbr"][slot])) * 255),
                            )
                        )
                record["roughness_texture"] = self.save_image(
                    Image.merge(
                        "RGBA",
                        (*channels, Image.new("L", size, 0), Image.new("L", size, 255)),
                    )
                )
                record["pbr"]["metallic"] = record["pbr"]["roughness"] = 1.0
            if "opacity" in maps:
                opacity = maps["opacity"].getchannel("R")
                base = maps.get(
                    "diffuse_texture", Image.new("RGBA", opacity.size, "white")
                ).copy()
                base.putalpha(opacity.resize(base.size, Image.Resampling.BILINEAR))
                record["diffuse_texture"] = self.save_image(base)
                record["pbr"]["opacity"] = 1.0
            for name in ["clearcoat", "displacement", "occlusion", "ior"]:
                inp = shader.GetInput(name)
                if inp and inp.HasConnectedSource():
                    self.warn(
                        f"{key}: {name} texture is not represented by the viewport shader"
                    )
        else:
            gprim = UsdGeom.Gprim(prim)
            if gprim:
                color = gprim.GetDisplayColorPrimvar().ComputeFlattened(self.time)
                if color:
                    record["diffuse"] = list(color[0])
                    if len(color) > 1:
                        self.warn(
                            f"{prim.GetPath()}: varying displayColor is approximated by its first value"
                        )
                opacity = gprim.GetDisplayOpacityPrimvar().ComputeFlattened(self.time)
                if opacity:
                    record["pbr"]["opacity"] = float(opacity[0])
            if shader:
                self.warn(
                    f"{key}: shader {shader.GetIdAttr().Get()} uses a displayColor fallback; only UsdPreviewSurface is supported"
                )
        index = len(self.materials)
        self.material_ids[key] = index
        self.materials.append(record)
        return index

    @staticmethod
    def triangles(points, face):
        """Ear clipping for concave polygons; return indices into face corners."""
        if len(face) < 3:
            return []
        if len(face) == 3:
            return [(0, 1, 2)]
        coords = [points[i] for i in face]
        normal = [
            sum(
                (coords[i][(a + 1) % 3] - coords[(i + 1) % len(coords)][(a + 1) % 3])
                * (coords[i][(a + 2) % 3] + coords[(i + 1) % len(coords)][(a + 2) % 3])
                for i in range(len(coords))
            )
            for a in range(3)
        ]
        drop = max(range(3), key=lambda a: abs(normal[a]))
        axes = [a for a in range(3) if a != drop]
        p = [(v[axes[0]], v[axes[1]]) for v in coords]

        def cross(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        area = sum(
            p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
            for i in range(len(p))
        )
        sign = 1 if area >= 0 else -1
        extent = max(max(v[a] for v in p) - min(v[a] for v in p) for a in (0, 1))
        epsilon = max(extent * extent * 1e-12, 1e-30)
        remaining, result = list(range(len(face))), []
        while len(remaining) > 3:
            for pos, b in enumerate(remaining):
                a, c = remaining[pos - 1], remaining[(pos + 1) % len(remaining)]
                if sign * cross(p[a], p[b], p[c]) <= epsilon:
                    continue
                if any(
                    all(
                        sign * cross(p[x], p[y], p[k]) >= -epsilon
                        for x, y in [(a, b), (b, c), (c, a)]
                    )
                    for k in remaining
                    if k not in (a, b, c)
                ):
                    continue
                result.append((a, b, c))
                remaining.pop(pos)
                break
            else:
                raise ValueError(
                    "Degenerate or self-intersecting polygon cannot be triangulated"
                )
        result.append(tuple(remaining))
        return result

    def primvar(self, mesh, name):
        pv = UsdGeom.PrimvarsAPI(mesh).FindPrimvarWithInheritance(name)
        if pv and pv.HasValue():
            return pv.ComputeFlattened(self.time), str(pv.GetInterpolation())
        return [], "constant"

    @staticmethod
    def sample(values, interpolation, point, face, corner):
        index = {
            "constant": 0,
            "uniform": face,
            "vertex": point,
            "varying": point,
            "faceVarying": corner,
        }.get(interpolation)
        if index is None or index >= len(values):
            raise ValueError(f"Invalid {interpolation} primvar index {index}")
        return values[index]

    def emit_mesh(self, prim, transform=None, label=None):
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            return self.emit_primitive(prim, transform, label)
        points = mesh.GetPointsAttr().Get(self.time)
        counts = mesh.GetFaceVertexCountsAttr().Get(self.time)
        indices = mesh.GetFaceVertexIndicesAttr().Get(self.time)
        if not points or not counts:
            return
        if (
            not indices
            or sum(counts) != len(indices)
            or any(c < 3 for c in counts)
            or any(i < 0 or i >= len(points) for i in indices)
        ):
            raise ValueError(f"{prim.GetPath()}: invalid mesh topology")
        if mesh.GetSubdivisionSchemeAttr().Get() not in ("none", None):
            self.warn(
                f"{prim.GetPath()}: subdivision surface imported as its control mesh"
            )
        materials = [self.material(prim)] * len(counts)
        for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
            material = self.material(subset.GetPrim())
            for face in subset.GetIndicesAttr().Get(self.time) or []:
                if face < 0 or face >= len(counts):
                    raise ValueError(
                        f"{subset.GetPath()}: material subset face is out of bounds"
                    )
                materials[face] = material
        holes = set(mesh.GetHoleIndicesAttr().Get(self.time) or [])
        normals = mesh.GetNormalsAttr().Get(self.time)
        normal_interpolation = str(mesh.GetNormalsInterpolation())
        normal_pv, normal_pv_interp = self.primvar(prim, "normals")
        if normal_pv:
            normals, normal_interpolation = normal_pv, normal_pv_interp
        self.emit_faces(
            prim,
            points,
            counts,
            indices,
            materials,
            holes,
            normals,
            normal_interpolation,
            transform,
            label,
        )

    def emit_faces(
        self,
        prim,
        points,
        counts,
        indices,
        materials,
        holes,
        normals,
        normal_interpolation,
        transform,
        label,
        explicit_uv=None,
    ):
        world = (
            transform
            if transform is not None
            else self.transforms.GetLocalToWorldTransform(prim)
        )
        if self.local_geometry:
            world = Gf.Matrix4d(1)
        normal_matrix = world.GetInverse().GetTranspose()
        left = UsdGeom.Gprim(prim).GetOrientationAttr().Get() == "leftHanded"
        reverse = left != (world.GetDeterminant() < 0)
        # Avoid repeated transforms and Gf's expensive Python iterator boundary.
        # Index components explicitly, and transform each source value once.
        positions = []
        for point in points:
            p = world.Transform(Gf.Vec3d(point[0], point[1], point[2]))
            x, y, z = p[0] * self.scale, p[1] * self.scale, p[2] * self.scale
            positions.append((x, z, -y) if self.z_up else (x, y, z))
        transformed_normals = []
        if normals:
            for value in normals:
                n = normal_matrix.TransformDir(Gf.Vec3d(value[0], value[1], value[2]))
                x, y, z = n[0], n[1], n[2]
                length = max(math.sqrt(x * x + y * y + z * z), 1e-20)
                x, y, z = x / length, y / length, z / length
                transformed_normals.append((x, z, -y) if self.z_up else (x, y, z))
        partitions = {}
        uv_cache = {}
        corner_start = 0
        for face_no, count in enumerate(counts):
            face = indices[corner_start : corner_start + count]
            if face_no in holes:
                corner_start += count
                continue
            mat = materials[face_no]
            settings = self.materials[mat]
            if mat not in uv_cache:
                uv_cache[mat] = explicit_uv or self.primvar(prim, settings["uv_name"])
            uv_values, uv_interp = uv_cache[mat]
            output = partitions.setdefault(
                mat,
                dict(
                    name=label or str(prim.GetPath()),
                    positions=[],
                    normals=[],
                    texcoords=[],
                    indices=[],
                    material=mat,
                ),
            )
            for triangle in self.triangles(points, face):
                if reverse:
                    triangle = (triangle[0], triangle[2], triangle[1])
                for corner in triangle:
                    point_index = face[corner]
                    output["positions"].extend(positions[point_index])
                    if transformed_normals:
                        output["normals"].extend(
                            self.sample(
                                transformed_normals,
                                normal_interpolation,
                                point_index,
                                face_no,
                                corner_start + corner,
                            )
                        )
                    if uv_values:
                        uv = self.sample(
                            uv_values,
                            uv_interp,
                            point_index,
                            face_no,
                            corner_start + corner,
                        )
                        sx, sy, tx, ty, rotation = settings["uv_transform"]
                        x, y = uv[0] * sx, uv[1] * sy
                        angle = math.radians(rotation)
                        output["texcoords"].extend(
                            (
                                x * math.cos(angle) - y * math.sin(angle) + tx,
                                x * math.sin(angle) + y * math.cos(angle) + ty,
                            )
                        )
                    output["indices"].append(len(output["indices"]))
                    self.vertex_count += 1
                    if self.vertex_count > 30_000_000:
                        raise ValueError(
                            "USD exceeds the 30 million expanded vertex import limit"
                        )
            corner_start += count
        for mat, output in partitions.items():
            if len(partitions) > 1:
                output["name"] += " · " + self.materials[mat]["name"].rsplit("/", 1)[-1]
            self.meshes.append(output)

    def emit_primitive(self, prim, transform=None, label=None):
        kind = prim.GetTypeName()
        if kind not in ("Cube", "Sphere", "Cylinder", "Cone", "Capsule", "Plane"):
            return

        def attr(name, default):
            value = prim.GetAttribute(name).Get(self.time)
            return value if value is not None else default

        points, faces, uv = [], [], []
        if kind == "Cube":
            r = float(attr("size", 2)) * 0.5
            points = [
                (x * r, y * r, z * r)
                for x, y, z in [
                    (-1, -1, -1),
                    (1, -1, -1),
                    (1, 1, -1),
                    (-1, 1, -1),
                    (-1, -1, 1),
                    (1, -1, 1),
                    (1, 1, 1),
                    (-1, 1, 1),
                ]
            ]
            faces = [
                [0, 3, 2, 1],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ]
        elif kind == "Plane":
            w, h = float(attr("width", 2)) * 0.5, float(attr("length", 2)) * 0.5
            points = [(-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)]
            faces = [[0, 1, 2, 3]]
        else:
            radius, height = float(attr("radius", 1)), float(attr("height", 2))
            segments, rings = 32, 16
            if kind in ("Sphere", "Capsule"):
                for j in range(rings + 1):
                    theta = math.pi * j / rings
                    z = radius * math.cos(theta)
                    if kind == "Capsule":
                        z += height * 0.5 if j <= rings // 2 else -height * 0.5
                    for i in range(segments + 1):
                        phi = 2 * math.pi * i / segments
                        points.append(
                            (
                                radius * math.sin(theta) * math.cos(phi),
                                radius * math.sin(theta) * math.sin(phi),
                                z,
                            )
                        )
                        uv.append((i / segments, j / rings))
                for j in range(rings):
                    for i in range(segments):
                        a = j * (segments + 1) + i
                        b = a + segments + 1
                        if j == 0:
                            faces.append([a, b, b + 1])
                        elif j == rings - 1:
                            faces.append([a, b, a + 1])
                        else:
                            faces.append([a, b, b + 1, a + 1])
            else:
                for j in range(2):
                    r = radius if kind == "Cylinder" or j == 0 else 0
                    for i in range(segments):
                        angle = 2 * math.pi * i / segments
                        points.append(
                            (
                                r * math.cos(angle),
                                r * math.sin(angle),
                                (-0.5 + j) * height,
                            )
                        )
                for i in range(segments):
                    k = (i + 1) % segments
                    faces.append(
                        [i, k, k + segments, i + segments]
                        if kind == "Cylinder"
                        else [i, k, i + segments]
                    )
                faces.append(list(reversed(range(segments))))
                if kind == "Cylinder":
                    faces.append(list(range(segments, 2 * segments)))
        axis = str(attr("axis", "Z"))
        if axis == "X":
            points = [(z, x, y) for x, y, z in points]
        elif axis == "Y":
            points = [(y, z, x) for x, y, z in points]
        counts = [len(f) for f in faces]
        indices = [i for f in faces for i in f]
        self.emit_faces(
            prim,
            points,
            counts,
            indices,
            [self.material(prim)] * len(faces),
            set(),
            [],
            "constant",
            transform,
            label,
            (uv, "vertex") if uv else None,
        )

    def visible(self, prim):
        imageable = UsdGeom.Imageable(prim)
        return not imageable or (
            imageable.ComputeVisibility(self.time) != "invisible"
            and imageable.ComputePurpose() != "guide"
        )

    def emit_instancer(self, prim, prefix=None, parent_transform=None, depth=0):
        if depth > 32:
            raise ValueError("Point instancer nesting exceeds 32 levels")
        instancer = UsdGeom.PointInstancer(prim)
        targets = instancer.GetPrototypesRel().GetTargets()
        proto_indices = instancer.GetProtoIndicesAttr().Get(self.time) or []
        transforms = instancer.ComputeInstanceTransformsAtTime(
            self.time,
            self.time,
            UsdGeom.PointInstancer.IncludeProtoXform,
            UsdGeom.PointInstancer.IgnoreMask,
        )
        mask = instancer.ComputeMaskAtTime(self.time)
        world = (
            parent_transform
            if parent_transform is not None
            else self.transforms.GetLocalToWorldTransform(prim)
        )
        for index, proto_index in enumerate(proto_indices):
            if mask and not mask[index]:
                continue
            if (
                proto_index < 0
                or proto_index >= len(targets)
                or index >= len(transforms)
            ):
                raise ValueError(f"{prim.GetPath()}: invalid point instance prototype")
            root = self.stage.GetPrimAtPath(targets[proto_index])
            inverse = self.transforms.GetLocalToWorldTransform(root).GetInverse()
            for child in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not self.visible(child) or any(
                    child.GetPath().HasPrefix(target)
                    for target in self.point_prototypes
                    if target != root.GetPath() and target.HasPrefix(root.GetPath())
                ):
                    continue
                matrix = (
                    self.transforms.GetLocalToWorldTransform(child)
                    * inverse
                    * transforms[index]
                    * world
                )
                name = f"{prefix or prim.GetPath()}/instance_{index}{str(child.GetPath())[len(str(root.GetPath())):]}"
                if child.IsA(UsdGeom.PointInstancer):
                    self.emit_instancer(child, name, matrix, depth + 1)
                else:
                    self.emit_mesh(child, matrix, name)

    def viewport_vector(self, vector, position=False):
        value = vector if self.z_up else (vector[0], -vector[2], vector[1])
        return [float(v) * (self.scale if position else 1.0) for v in value]

    def emit_camera(self, prim):
        camera = UsdGeom.Camera(prim)
        world = self.transforms.GetLocalToWorldTransform(prim)
        eye = self.viewport_vector(world.Transform(Gf.Vec3d(0)), True)
        direction = self.viewport_vector(
            world.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
        )
        up = self.viewport_vector(world.TransformDir(Gf.Vec3d(0, 1, 0)).GetNormalized())
        aperture = float(camera.GetVerticalApertureAttr().Get(self.time))
        focal = float(camera.GetFocalLengthAttr().Get(self.time))
        self.cameras.append(
            dict(
                name=str(prim.GetPath()),
                eye=eye,
                target=[a + b for a, b in zip(eye, direction)],
                up=up,
                fovy=math.degrees(2 * math.atan(aperture / (2 * max(focal, 1e-6)))),
                orthographic=camera.GetProjectionAttr().Get(self.time)
                == "orthographic",
                ortho_scale=aperture * 0.1 * self.scale,
            )
        )
        if camera.GetHorizontalApertureOffsetAttr().Get(
            self.time
        ) or camera.GetVerticalApertureOffsetAttr().Get(self.time):
            self.warn(
                f"{prim.GetPath()}: camera lens shift is not supported by the viewport"
            )

    def emit_light(self, prim):
        api = UsdLux.LightAPI(prim)
        kind = prim.GetTypeName()
        mapping = {
            "DistantLight": "directional",
            "SphereLight": "sphere",
            "RectLight": "rectangle",
            "DiskLight": "disk",
            "CylinderLight": "tube",
            "DomeLight": "environment",
        }
        if kind not in mapping:
            self.warn(f"{prim.GetPath()}: unsupported light type {kind}")
            return

        def attr(name, default):
            value = prim.GetAttribute("inputs:" + name).Get(self.time)
            return value if value is not None else default

        world = self.transforms.GetLocalToWorldTransform(prim)
        axes = [
            self.viewport_vector(
                world.TransformDir(Gf.Vec3d(*(1 if i == j else 0 for i in range(3))))
            )
            for j in range(3)
        ]
        lengths = [math.sqrt(sum(v * v for v in axis)) for axis in axes]
        x, y, z = [
            [v / max(length, 1e-20) for v in axis]
            for axis, length in zip(axes, lengths)
        ]
        pitch = math.asin(max(-1, min(1, -x[2])))
        roll = math.atan2(y[2], z[2]) if abs(math.cos(pitch)) > 1e-6 else 0
        yaw = (
            math.atan2(x[1], x[0])
            if abs(math.cos(pitch)) > 1e-6
            else math.atan2(-y[0], y[1])
        )
        light = dict(
            name=str(prim.GetPath()),
            kind=mapping[kind],
            position=self.viewport_vector(world.Transform(Gf.Vec3d(0)), True),
            rotation_degrees=list(map(math.degrees, (roll, pitch, yaw))),
            color=list(api.GetColorAttr().Get(self.time)),
            intensity=float(api.GetIntensityAttr().Get(self.time)),
            exposure=float(api.GetExposureAttr().Get(self.time)),
            temperature=float(api.GetColorTemperatureAttr().Get(self.time)),
            use_temperature=bool(api.GetEnableColorTemperatureAttr().Get(self.time)),
            radius=float(attr("radius", 0.5)) * self.scale * max(lengths),
            size=[
                float(attr("width", attr("length", 1))) * self.scale * lengths[0],
                float(attr("height", 1)) * self.scale * lengths[1],
            ],
            cone_angle=float(attr("shaping:cone:angle", 180)),
            cone_softness=float(attr("shaping:cone:softness", 0)),
            diffuse=float(api.GetDiffuseAttr().Get(self.time)),
            specular=float(api.GetSpecularAttr().Get(self.time)),
            environment_path="",
        )
        if kind == "SphereLight" and attr("treatAsPoint", False):
            light["kind"] = "point"
        if light["cone_angle"] < 180:
            light["kind"] = "spot"
        if kind == "DomeLight":
            value = attr("texture:file", None)
            if value and value.path:
                try:
                    resolved = value.resolvedPath or str(
                        Ar.GetResolver().Resolve(value.path)
                    )
                    asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(resolved))
                    # Keep the original HDR/EXR bytes for the existing environment importer.
                    extension = Path(value.path).suffix.lower()
                    if extension not in (".hdr", ".exr"):
                        self.warn(
                            f"{prim.GetPath()}: dome texture must be HDR or EXR in this viewport"
                        )
                    else:
                        light["environment_path"] = self.persist(
                            bytes(asset.GetBuffer()), extension
                        )
                except Exception as error:
                    self.warn(f"{prim.GetPath()}: {error}")
        if api.GetNormalizeAttr().Get(self.time):
            self.warn(
                f"{prim.GetPath()}: normalized light energy is approximated by viewport intensity"
            )
        self.lights.append(light)

    def run(self):
        if any(prim.IsA(UsdSkel.Root) for prim in self.stage.Traverse()):
            # Bake into an anonymous flattened stage; never modify source files.
            layer = self.stage.Flatten()
            self.stage = Usd.Stage.Open(layer)
            try:
                if not UsdSkel.BakeSkinning(
                    self.stage.Traverse(),
                    Gf.Interval(self.time.GetValue(), self.time.GetValue()),
                ):
                    self.warn("Some skeleton deformation could not be baked")
            except Exception as error:
                self.warn(f"Skeleton deformation: {error}")
            self.transforms = UsdGeom.XformCache(self.time)
        prims = list(Usd.PrimRange.Stage(self.stage, Usd.TraverseInstanceProxies()))
        for prim in prims:
            if prim.IsA(UsdGeom.PointInstancer):
                self.point_prototypes.update(
                    UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets()
                )
        with Ar.ResolverContextBinder(self.stage.GetPathResolverContext()):
            for prim in prims:
                if any(
                    prim.GetPath().HasPrefix(root) for root in self.point_prototypes
                ) or not self.visible(prim):
                    continue
                if prim.IsA(UsdGeom.PointInstancer):
                    self.emit_instancer(prim)
                elif prim.IsA(UsdGeom.Mesh) or prim.GetTypeName() in (
                    "Cube",
                    "Sphere",
                    "Cylinder",
                    "Cone",
                    "Capsule",
                    "Plane",
                ):
                    self.emit_mesh(prim)
                elif prim.IsA(UsdGeom.Camera):
                    self.emit_camera(prim)
                elif prim.HasAPI(UsdLux.LightAPI):
                    self.emit_light(prim)
                elif prim.GetTypeName() in (
                    "BasisCurves",
                    "NurbsCurves",
                    "NurbsPatch",
                    "Points",
                    "Volume",
                ):
                    self.warn(
                        f"{prim.GetPath()}: {prim.GetTypeName()} is not imported into the static mesh viewport"
                    )
        if not self.meshes and not self.lights and not self.cameras:
            raise ValueError(
                "USD stage has no visible renderable mesh geometry. "
                + " ".join(sorted(self.warnings))
            )
        return dict(
            meshes=self.meshes,
            materials=self.materials,
            warnings=sorted(self.warnings),
            time_code=self.time.GetValue(),
            scene=dict(lights=self.lights, cameras=self.cameras, animation=self.animation),
        )


class TransformSampler:
    """Keep static mesh buffers intact when only scene transforms animate."""
    def __init__(self, stage):
        self.stage = stage
        self.prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
        self.supported = not any(
            prim.IsA(UsdSkel.Root) or prim.IsA(UsdGeom.PointInstancer)
            or any(attr.GetNumTimeSamples() > 1 and (
                not attr.GetName().startswith("xformOp"))
                for attr in prim.GetAttributes())
            for prim in self.prims)
        self.prims = [prim for prim in self.prims if prim.IsA(UsdGeom.Mesh)
                      or prim.GetTypeName() in ("Cube", "Sphere", "Cylinder", "Cone", "Capsule", "Plane")]
        scale = UsdGeom.GetStageMetersPerUnit(stage)
        if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
            self.axes = Gf.Matrix4d(scale,0,0,0, 0,scale,0,0, 0,0,scale,0, 0,0,0,1)
        else:
            self.axes = Gf.Matrix4d(scale,0,0,0, 0,0,scale,0, 0,-scale,0,0, 0,0,0,1)
        self.inverse_axes = self.axes.GetInverse()

    def sample(self, time):
        if not self.supported:
            return None
        cache = UsdGeom.XformCache(Usd.TimeCode(time))
        transforms = {}
        for prim in self.prims:
            path = str(prim.GetPath())
            delta = self.inverse_axes * cache.GetLocalToWorldTransform(prim) * self.axes
            transforms[path] = [[delta[r][c] for c in range(4)] for r in range(4)]
        return {"transforms": transforms, "time_code": time}


def main():
    try:
        if len(sys.argv) > 3 and sys.argv[3] == "--serve":
            stage = Usd.Stage.Open(str(Path(sys.argv[1]).resolve()), load=Usd.Stage.LoadAll)
            if not stage:
                raise ValueError("OpenUSD could not open the stage")
            sampler = TransformSampler(stage)
            local_geometry_sent = False
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    time = float(request["time"] if isinstance(request, dict) else request)
                    if not math.isfinite(time):
                        raise ValueError("Time must be finite")
                    result = sampler.sample(time) if isinstance(request, dict) else None
                    if result is None:
                        result = Importer(sys.argv[1], sys.argv[2], time, stage=stage).run()
                    elif not local_geometry_sent:
                        geometry = Importer(sys.argv[1], sys.argv[2], time, stage=stage, local_geometry=True).run()
                        geometry.update(result)
                        result = geometry
                        local_geometry_sent = True
                except Exception as error:
                    result = {"error": str(error)}
                print(json.dumps(result, allow_nan=False, separators=(",", ":")), flush=True)
            return 0
        time = (
            None
            if len(sys.argv) < 4 or sys.argv[3] == "default"
            else float(sys.argv[3])
        )
        if time is not None and not math.isfinite(time):
            raise ValueError("Time code must be finite")
        result = Importer(sys.argv[1], sys.argv[2], time).run()
        sys.stdout.write(json.dumps(result, allow_nan=False, separators=(",", ":")))
    except Exception as error:
        print(f"USD import failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
