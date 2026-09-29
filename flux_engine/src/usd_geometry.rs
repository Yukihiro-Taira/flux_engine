use super::*;

struct Primvar {
    values: Vec<Vec<f32>>,
    indices: Vec<i32>,
    interpolation: String,
}
impl Primvar {
    fn read(p: &Prim, name: &str, time: f64, inherited: bool, default: &str) -> Result<Self> {
        let mut current = Some(p.clone());
        while let Some(p) = current {
            if let Some(v) = value(&p, name, time)? {
                let values = match v {
                    sdf::Value::Vec2fVec(v) => v.into_iter().map(|v| vec![v.x, v.y]).collect(),
                    sdf::Value::Vec2dVec(v) => v
                        .into_iter()
                        .map(|v| vec![v.x as f32, v.y as f32])
                        .collect(),
                    sdf::Value::Vec3fVec(v) => v.into_iter().map(|v| vec![v.x, v.y, v.z]).collect(),
                    sdf::Value::FloatVec(v) => v.into_iter().map(|v| vec![v]).collect(),
                    _ => bail!("{}: unsupported primvar {name}", p.path()),
                };
                let interpolation = p
                    .attribute(name)
                    .get_metadata::<openusd::tf::Token>("interpolation")?
                    .map(|v| v.to_string())
                    .unwrap_or(default.into());
                return Ok(Self {
                    values,
                    indices: ints(&p, &format!("{name}:indices"), time)?,
                    interpolation,
                });
            }
            current = if inherited { parent(&p)? } else { None };
        }
        Ok(Self {
            values: vec![],
            indices: vec![],
            interpolation: default.into(),
        })
    }
    fn sample(&self, point: usize, face: usize, corner: usize) -> Result<&[f32]> {
        let index = match self.interpolation.as_str() {
            "constant" => 0,
            "uniform" => face,
            "vertex" | "varying" => point,
            "faceVarying" => corner,
            other => bail!("Invalid primvar interpolation {other}"),
        };
        let index = if self.indices.is_empty() {
            index
        } else {
            usize::try_from(*self.indices.get(index).context("Invalid primvar index")?)
                .context("Negative primvar index")?
        };
        Ok(self
            .values
            .get(index)
            .context("Primvar index out of bounds")?)
    }
}

impl Importer<'_> {
    pub(super) fn mesh(
        &mut self,
        p: &Prim,
        kind: &str,
        mut matrix: Matrix,
        label: Option<String>,
    ) -> Result<()> {
        let (mut points, counts, indices, explicit_uv) = if kind == "Mesh" {
            (
                vectors(p, "points", self.time)?,
                ints(p, "faceVertexCounts", self.time)?,
                ints(p, "faceVertexIndices", self.time)?,
                vec![],
            )
        } else {
            primitive(p, kind, self.time)?
        };
        if points.is_empty() || counts.is_empty() {
            return Ok(());
        }
        ensure!(
            counts.iter().all(|&c| c >= 3)
                && counts.iter().map(|&c| c as u64).sum::<u64>() == indices.len() as u64
                && indices
                    .iter()
                    .all(|&i| i >= 0 && (i as usize) < points.len()),
            "{}: invalid mesh topology",
            p.path()
        );
        ensure!(
            points.iter().flatten().all(|v| v.is_finite()),
            "{}: non-finite mesh points",
            p.path()
        );
        let mut normals = Primvar::read(p, "primvars:normals", self.time, true, "constant")?;
        if normals.values.is_empty() {
            normals = Primvar::read(p, "normals", self.time, false, "vertex")?;
        }
        if kind == "Mesh" {
            self.skin(p, &mut points, &mut normals)?;
            if token(p, "subdivisionScheme", self.time, "catmullClark")? != "none" {
                self.warn(format!(
                    "{}: subdivision surface imported as its control mesh",
                    p.path()
                ));
            }
        }
        if self.local {
            matrix = Matrix::IDENTITY;
        }
        let normal_matrix = matrix
            .inverse()
            .map(|m| gf::Matrix4d(std::array::from_fn(|i| m.0[(i % 4) * 4 + i / 4])));
        let determinant = matrix.0[0] * (matrix.0[5] * matrix.0[10] - matrix.0[6] * matrix.0[9])
            - matrix.0[1] * (matrix.0[4] * matrix.0[10] - matrix.0[6] * matrix.0[8])
            + matrix.0[2] * (matrix.0[4] * matrix.0[9] - matrix.0[5] * matrix.0[8]);
        let reverse = (token(p, "orientation", self.time, "rightHanded")? == "leftHanded")
            != (determinant < 0.);
        let transformed: Vec<_> = points
            .iter()
            .map(|&v| self.mesh_vector(matrix.transform_point(v.into()).into(), true))
            .collect();
        let default_material = self.material(p)?;
        let mut materials = vec![default_material; counts.len()];
        for subset in p.children()? {
            if subset
                .type_name()?
                .is_some_and(|t| t.as_str() == "GeomSubset")
                && token(&subset, "elementType", self.time, "face")? == "face"
            {
                let mat = self.material(&subset)?;
                for face in ints(&subset, "indices", self.time)? {
                    ensure!(
                        face >= 0 && (face as usize) < materials.len(),
                        "Material subset face out of bounds"
                    );
                    materials[face as usize] = mat;
                }
            }
        }
        let holes: HashSet<_> = ints(p, "holeIndices", self.time)?.into_iter().collect();
        let mut partitions: BTreeMap<usize, UsdMesh> = BTreeMap::new();
        let mut uvs = HashMap::new();
        let mut offset = 0;
        for (face_no, &count) in counts.iter().enumerate() {
            let count = count as usize;
            let face = &indices[offset..offset + count];
            if holes.contains(&(face_no as i32)) {
                offset += count;
                continue;
            }
            let mat = materials[face_no];
            if !uvs.contains_key(&mat) {
                let uv = if explicit_uv.is_empty() {
                    Primvar::read(
                        p,
                        &format!("primvars:{}", self.uv_settings[mat].0),
                        self.time,
                        true,
                        "constant",
                    )?
                } else {
                    Primvar {
                        values: explicit_uv.iter().map(|v| v.to_vec()).collect(),
                        indices: vec![],
                        interpolation: "vertex".into(),
                    }
                };
                uvs.insert(mat, uv);
            }
            let uv = &uvs[&mat];
            let mesh = partitions.entry(mat).or_insert_with(|| UsdMesh {
                name: label.clone().unwrap_or_else(|| p.path().to_string()),
                positions: vec![],
                normals: vec![],
                texcoords: vec![],
                indices: vec![],
                material: mat,
            });
            let triangles = triangulate(&points, face)?;
            for triangle in triangles.chunks_exact(3) {
                let triangle = if reverse {
                    [triangle[0], triangle[2], triangle[1]]
                } else {
                    [triangle[0], triangle[1], triangle[2]]
                };
                for corner in triangle {
                    let point = face[corner] as usize;
                    mesh.positions.extend(transformed[point]);
                    if !normals.values.is_empty() {
                        if let Some(m) = normal_matrix {
                            let n = normals.sample(point, face_no, offset + corner)?;
                            ensure!(n.len() == 3, "Invalid normal vector");
                            mesh.normals.extend(normalize(self.mesh_vector(
                                m.transform_vec([n[0], n[1], n[2]].into()).into(),
                                false,
                            )));
                        }
                    }
                    if !uv.values.is_empty() {
                        let v = uv.sample(point, face_no, offset + corner)?;
                        ensure!(v.len() == 2, "Invalid texture coordinate");
                        let [sx, sy, tx, ty, r] = self.uv_settings[mat].1;
                        let (s, c) = r.to_radians().sin_cos();
                        let (x, y) = (v[0] * sx, v[1] * sy);
                        mesh.texcoords
                            .extend([x * c - y * s + tx, x * s + y * c + ty]);
                    }
                    mesh.indices.push(mesh.indices.len() as u32);
                    self.vertices += 1;
                    ensure!(
                        self.vertices <= 30_000_000,
                        "USD exceeds the 30 million expanded vertex import limit"
                    );
                }
            }
            offset += count;
        }
        let split = partitions.len() > 1;
        for (mat, mut mesh) in partitions {
            if split {
                mesh.name.push_str(" · ");
                mesh.name.push_str(
                    self.scene.materials[mat]
                        .name
                        .rsplit('/')
                        .next()
                        .unwrap_or("material"),
                );
            }
            self.scene.meshes.push(mesh);
        }
        Ok(())
    }
    fn skin(&mut self, p: &Prim, points: &mut Vec<[f32; 3]>, normals: &mut Primvar) -> Result<()> {
        use openusd_schemas::skel::{
            SkelAnimQuery, SkelBindingAPI, Skeleton, SkeletonResolver, SkinningResolver,
        };
        let Some(binding) = SkelBindingAPI::get(self.stage, p.path().clone())? else {
            return Ok(());
        };
        let Some(skeleton_path) = binding.inherited_skeleton()? else {
            return Ok(());
        };
        let skeleton =
            Skeleton::get(self.stage, skeleton_path.clone())?.context("Missing bound skeleton")?;
        let resolver = SkeletonResolver::from_skeleton(&skeleton)?;
        ensure!(
            resolver.rest_pose_local().len() == resolver.joint_order().len()
                && resolver.inverse_bind_transforms().len() == resolver.joint_order().len(),
            "Skeleton joint and transform counts do not match"
        );
        if !binding.blend_shape_targets()?.is_empty() {
            self.warn(format!(
                "{}: blend-shape deformation is not supported by this importer",
                p.path()
            ));
        }

        let skeleton_binding = SkelBindingAPI::get(self.stage, skeleton_path.clone())?;
        let animation = skeleton_binding
            .map(|b| b.inherited_animation_source())
            .transpose()?
            .flatten();
        let mut local = resolver.rest_pose_local().to_vec();
        if let Some(animation) = animation {
            if let Some(query) = SkelAnimQuery::new(self.stage, animation)? {
                let animated = query.compute_joint_local_transforms(self.stage, self.time)?;
                for (i, name) in resolver.joint_order().iter().enumerate() {
                    if let Some(j) = query.joint_order().iter().position(|n| n == name) {
                        if let Some(m) = animated.get(j) {
                            local[i] = *m;
                        }
                    }
                }
            }
        }
        let transforms = resolver.compute_skinning_transforms_from_local(&local, Matrix::IDENTITY);
        let skin = SkinningResolver::from_binding(&binding, resolver.joint_order())?;
        if skin.skinning_method().as_token() != "classicLinear" {
            self.warn(format!(
                "{}: dual-quaternion skinning is approximated by linear blend skinning",
                p.path()
            ));
        }

        let indices = binding.joint_indices()?;
        let weights = binding.joint_weights()?;
        let influence_count = skin.num_influences_per_component();
        let components = if skin.is_rigidly_deformed() {
            1
        } else {
            points.len()
        };
        ensure!(
            influence_count > 0
                && components.checked_mul(influence_count) == Some(indices.len())
                && weights.len() == indices.len(),
            "Invalid skinning influence arrays"
        );
        ensure!(
            indices
                .iter()
                .all(|&i| i >= 0 && (i as usize) < skin.joint_order_len())
                && weights.iter().all(|w| w.is_finite() && *w >= 0.),
            "Invalid skinning weights or joint indices"
        );

        let skel_to_mesh = world(&self.stage.prim(skeleton_path)?, self.time)?
            * world(p, self.time)?
                .inverse()
                .context("Singular skinned mesh transform")?;
        let source: Vec<gf::Vec3f> = points.iter().copied().map(Into::into).collect();
        let result = if skin.is_rigidly_deformed() {
            let m = skin.compute_rigid_transform(&transforms);
            source.iter().map(|&v| m.transform_point(v)).collect()
        } else {
            skin.compute_skinned_points(&source, &transforms)
        };
        *points = result
            .into_iter()
            .map(|v| skel_to_mesh.transform_point(v).into())
            .collect();
        // Regenerate geometric normals after deformation instead of retaining the bind-pose normals.
        normals.values.clear();
        Ok(())
    }
    pub(super) fn instancer(
        &mut self,
        p: &Prim,
        matrix: Matrix,
        label: Option<String>,
        depth: usize,
    ) -> Result<()> {
        ensure!(depth < 32, "Point instancer nesting exceeds 32 levels");
        let targets = p.relationship("prototypes").targets()?;
        let indices = ints(p, "protoIndices", self.time)?;
        let positions = vectors(p, "positions", self.time)?;
        let scales = vectors(p, "scales", self.time)?;
        let orientations =
            match value(p, "orientationsf", self.time)?.or(value(p, "orientations", self.time)?) {
                Some(sdf::Value::QuatfVec(v)) => v,
                Some(sdf::Value::QuathVec(v)) => v
                    .into_iter()
                    .map(|q| {
                        gf::Quatf::from([q.w.to_f32(), q.x.to_f32(), q.y.to_f32(), q.z.to_f32()])
                    })
                    .collect(),
                _ => vec![],
            };
        let ids = match value(p, "ids", self.time)? {
            Some(sdf::Value::Int64Vec(v)) => v,
            _ => (0..indices.len() as i64).collect(),
        };
        let mut hidden = match value(p, "invisibleIds", self.time)? {
            Some(sdf::Value::Int64Vec(v)) => v.into_iter().collect::<HashSet<_>>(),
            _ => HashSet::new(),
        };
        if let Some(sdf::Value::Int64ListOp(op)) = p.get_metadata::<sdf::Value>("inactiveIds")? {
            hidden.extend(op.flatten());
        }
        ensure!(
            positions.len() == indices.len()
                && ids.len() == indices.len()
                && (scales.is_empty() || scales.len() == indices.len())
                && (orientations.is_empty() || orientations.len() == indices.len()),
            "Invalid point instance arrays"
        );
        let all = paths(self.stage)?;
        for (i, &index) in indices.iter().enumerate() {
            if hidden.contains(&ids[i]) {
                continue;
            }
            ensure!(
                index >= 0 && (index as usize) < targets.len(),
                "Invalid point instance prototype"
            );
            let root = &targets[index as usize];
            let local = Matrix::from_trs(
                positions[i].into(),
                orientations
                    .get(i)
                    .copied()
                    .unwrap_or([1., 0., 0., 0.].into()),
                scales.get(i).copied().unwrap_or([1.; 3]).into(),
            );
            let root_prim = self.stage.prim(root.clone())?;
            let root_parent = parent(&root_prim)?
                .map(|p| world(&p, self.time))
                .transpose()?
                .unwrap_or(Matrix::IDENTITY)
                .inverse()
                .context("Singular prototype parent transform")?;
            for path in &all {
                if path != root && !path.as_str().starts_with(&format!("{root}/")) {
                    continue;
                }
                if self.prototypes.iter().any(|other| {
                    other != root.as_str()
                        && other.starts_with(&format!("{root}/"))
                        && (path.as_str() == other
                            || path.as_str().starts_with(&format!("{other}/")))
                }) {
                    continue;
                }
                let child = self.stage.prim(path.clone())?;
                if !visible(&child, self.time)? {
                    continue;
                }
                let name = format!(
                    "{}/instance_{i}{}",
                    label.as_deref().unwrap_or(p.path().as_str()),
                    &path.as_str()[root.as_str().len()..]
                );
                self.emit(
                    &child,
                    Some(world(&child, self.time)? * root_parent * local * matrix),
                    Some(name),
                    depth + 1,
                )?;
            }
        }
        Ok(())
    }
}
fn triangulate(points: &[[f32; 3]], face: &[i32]) -> Result<Vec<usize>> {
    if face.len() == 3 {
        return Ok(vec![0, 1, 2]);
    }
    let mut n = [0f64; 3];
    for i in 0..face.len() {
        let a = points[face[i] as usize];
        let b = points[face[(i + 1) % face.len()] as usize];
        for k in 0..3 {
            n[k] += (a[(k + 1) % 3] as f64 - b[(k + 1) % 3] as f64)
                * (a[(k + 2) % 3] as f64 + b[(k + 2) % 3] as f64);
        }
    }
    let drop = (0..3)
        .max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs()))
        .unwrap();
    let axes: Vec<_> = (0..3).filter(|&a| a != drop).collect();
    let coords: Vec<f64> = face
        .iter()
        .flat_map(|&i| axes.iter().map(move |&a| points[i as usize][a] as f64))
        .collect();
    let mut triangles =
        earcutr::earcut(&coords, &[], 2).map_err(|e| anyhow::anyhow!("Invalid polygon: {e:?}"))?;
    ensure!(
        triangles.len() == (face.len() - 2) * 3,
        "Degenerate or self-intersecting polygon cannot be triangulated"
    );
    // earcut emits a fixed orientation; retain the authored polygon winding.
    let area = (0..face.len())
        .map(|i| {
            coords[2 * i] * coords[2 * ((i + 1) % face.len()) + 1]
                - coords[2 * ((i + 1) % face.len())] * coords[2 * i + 1]
        })
        .sum::<f64>();
    for t in triangles.chunks_exact_mut(3) {
        let [a, b, c] = [t[0] * 2, t[1] * 2, t[2] * 2];
        let cross = (coords[b] - coords[a]) * (coords[c + 1] - coords[a + 1])
            - (coords[b + 1] - coords[a + 1]) * (coords[c] - coords[a]);
        if cross * area < 0. {
            t.swap(1, 2);
        }
    }
    Ok(triangles)
}
type Primitive = (Vec<[f32; 3]>, Vec<i32>, Vec<i32>, Vec<[f32; 2]>);
fn primitive(p: &Prim, kind: &str, time: f64) -> Result<Primitive> {
    let mut points = vec![];
    let mut faces: Vec<Vec<i32>> = vec![];
    let mut uv = vec![];
    let n = |name, default| scalar(p, name, time, default);
    if kind == "Cube" {
        let r = n("size", 2.)? * 0.5;
        points = vec![
            [-1., -1., -1.],
            [1., -1., -1.],
            [1., 1., -1.],
            [-1., 1., -1.],
            [-1., -1., 1.],
            [1., -1., 1.],
            [1., 1., 1.],
            [-1., 1., 1.],
        ]
        .into_iter()
        .map(|v| v.map(|v| v * r))
        .collect();
        faces = vec![
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ];
    } else if kind == "Plane" {
        let w = n("width", 2.)? * 0.5;
        let h = n("length", 2.)? * 0.5;
        points = vec![[-w, -h, 0.], [w, -h, 0.], [w, h, 0.], [-w, h, 0.]];
        faces.push(vec![0, 1, 2, 3]);
    } else {
        let radius = n("radius", 1.)?;
        let height = n("height", 2.)?;
        let segments = 32;
        let rings = 16;
        if matches!(kind, "Sphere" | "Capsule") {
            for j in 0..=rings {
                let theta = std::f32::consts::PI * j as f32 / rings as f32;
                let z = radius * theta.cos()
                    + if kind == "Capsule" {
                        if j <= rings / 2 {
                            height * 0.5
                        } else {
                            -height * 0.5
                        }
                    } else {
                        0.
                    };
                for i in 0..=segments {
                    let phi = std::f32::consts::TAU * i as f32 / segments as f32;
                    points.push([
                        radius * theta.sin() * phi.cos(),
                        radius * theta.sin() * phi.sin(),
                        z,
                    ]);
                    uv.push([i as f32 / segments as f32, j as f32 / rings as f32]);
                }
            }
            for j in 0..rings {
                for i in 0..segments {
                    let a = j * (segments + 1) + i;
                    let b = a + segments + 1;
                    faces.push(if j == 0 {
                        vec![a, b, b + 1]
                    } else if j == rings - 1 {
                        vec![a, b, a + 1]
                    } else {
                        vec![a, b, b + 1, a + 1]
                    });
                }
            }
        } else {
            for j in 0..2 {
                let r = if kind == "Cylinder" || j == 0 {
                    radius
                } else {
                    0.
                };
                for i in 0..segments {
                    let a = std::f32::consts::TAU * i as f32 / segments as f32;
                    points.push([r * a.cos(), r * a.sin(), (j as f32 - 0.5) * height]);
                }
            }
            for i in 0..segments {
                let k = (i + 1) % segments;
                faces.push(if kind == "Cylinder" {
                    vec![i, k, k + segments, i + segments]
                } else {
                    vec![i, k, i + segments]
                });
            }
            faces.push((0..segments).rev().collect());
            if kind == "Cylinder" {
                faces.push((segments..2 * segments).collect());
            }
        }
    }
    let axis = token(p, "axis", time, "Z")?;
    if axis == "X" {
        for v in &mut points {
            *v = [v[2], v[0], v[1]];
        }
    } else if axis == "Y" {
        for v in &mut points {
            *v = [v[1], v[2], v[0]];
        }
    }
    Ok((
        points,
        faces.iter().map(|f| f.len() as i32).collect(),
        faces.into_iter().flatten().collect(),
        uv,
    ))
}
