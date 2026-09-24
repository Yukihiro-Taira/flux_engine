use std::collections::{HashMap, HashSet};

use anyhow::{Context, bail};
use wgpu::util::DeviceExt;

use crate::model;

#[cfg(not(target_arch = "wasm32"))]
const MODEL_CACHE_VERSION: u32 = 5;

#[cfg(not(target_arch = "wasm32"))]
#[derive(serde::Serialize, serde::Deserialize)]
struct ModelCache {
    version: u32,
    source_len: u64,
    source_modified_ns: u128,
    model: PreparedModel,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedModel {
    meshes: Vec<PreparedMesh>,
    materials: Vec<model::MaterialSource>,
    import_warnings: Vec<String>,
    imported_scene: Option<crate::usd_import::ImportedScene>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PreparedMesh {
    name: String,
    vertices: Vec<model::ModelVertex>,
    indices: Vec<u32>,
    uv_tile: (i32, i32),
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    geometry_stats: model::GeometryStats,
    material: usize,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn prepare_model_from_path(
    path: &std::path::Path, time_code: Option<f64>,
) -> anyhow::Result<PreparedModel> {
    // A USD stage depends on layers, references, variants, and external textures.
    // Always recompose it; a root-file-only OBJ/FBX cache cannot validate those dependencies.
    if crate::usd_import::is_usd(path) {
        return load_usd_from_path(path, time_code);
    }
    let metadata = std::fs::metadata(path)?;
    let source_modified_ns = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    let cache_path = model_cache_path(path);
    if let Ok(file) = std::fs::File::open(&cache_path)
        && let Ok(cache) = read_model_cache(std::io::BufReader::new(file))
        && cache.version == MODEL_CACHE_VERSION
        && cache.source_len == metadata.len()
        && cache.source_modified_ns == source_modified_ns
        && !cache.model.meshes.is_empty()
    {
        return Ok(cache.model);
    }

    let prepared = load_prepared_model(path)?;
    let cache = ModelCache {
        version: MODEL_CACHE_VERSION,
        source_len: metadata.len(),
        source_modified_ns,
        model: prepared,
    };
    // Stream the cache to disk so large production meshes do not require a
    // second full-size allocation during serialization.
    let temporary = cache_path.with_extension("fxcache.tmp");
    if let Ok(file) = std::fs::File::create(&temporary) {
        let mut writer = std::io::BufWriter::new(file);
        if bincode::serialize_into(&mut writer, &cache).is_ok()
            && std::io::Write::flush(&mut writer).is_ok()
        {
            let _ = std::fs::remove_file(&cache_path);
            let _ = std::fs::rename(&temporary, &cache_path);
        } else {
            let _ = std::fs::remove_file(&temporary);
        }
    }
    Ok(cache.model)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_model_cache(mut reader: impl std::io::Read + std::io::Seek) -> anyhow::Result<ModelCache> {
    use bincode::Options;
    use std::io::SeekFrom;
    // Check the format before decoding schema-dependent vectors and strings.
    // Old material layouts must never be interpreted as lengths in the new layout.
    let length = reader.seek(SeekFrom::End(0))?;
    reader.rewind()?;
    let version: u32 = bincode::deserialize_from(&mut reader)?;
    anyhow::ensure!(version == MODEL_CACHE_VERSION, "Outdated model cache");
    reader.rewind()?;
    Ok(bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(length)
        .deserialize_from(reader)?)
}

#[cfg(not(target_arch = "wasm32"))]
fn model_cache_path(path: &std::path::Path) -> std::path::PathBuf {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map_or_else(|| "fxcache".to_owned(), |value| format!("{value}.fxcache"));
    path.with_extension(extension)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_prepared_model(path: &std::path::Path) -> anyhow::Result<PreparedModel> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("fbx"))
    {
        return load_fbx_from_path(path);
    }
    let (models, materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )?;
    let label = path.to_string_lossy();
    build_model(models, materials.unwrap_or_default(), &label)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_usd_from_path(path: &std::path::Path, time_code: Option<f64>) -> anyhow::Result<PreparedModel> {
    let scene = crate::usd_import::load(path, time_code)?;
    log::info!("Composed USD at time code {}", scene.time_code);
    let models: Vec<_> = scene.meshes.into_iter().map(|mesh| tobj::Model {
        name: mesh.name,
        mesh: tobj::Mesh {
            positions: mesh.positions, normals: mesh.normals, texcoords: mesh.texcoords,
            indices: mesh.indices, material_id: Some(mesh.material), ..Default::default()
        },
    }).collect();
    let mut prepared = if models.is_empty() {
        PreparedModel { meshes: Vec::new(), materials: Vec::new(), import_warnings: Vec::new(), imported_scene: None }
    } else {
        build_model_with_normals(models, Vec::new(), &path.to_string_lossy(), true)?
    };
    prepared.materials = scene.materials;
    prepared.import_warnings = scene.warnings;
    prepared.imported_scene = Some(scene.scene);
    Ok(prepared)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_fbx_from_path(path: &std::path::Path) -> anyhow::Result<PreparedModel> {
    let filename = path
        .to_str()
        .context("FBX path contains unsupported non-UTF-8 characters")?;
    let mut options = ufbx::LoadOpts::default();
    options.target_axes = ufbx::CoordinateAxes::right_handed_y_up();
    options.target_unit_meters = 1.0;
    options.generate_missing_normals = true;
    let scene = ufbx::load_file(filename, options)
        .map_err(|error| anyhow::anyhow!("FBX parse failed: {error:?}"))?;
    let mut imported_models = Vec::new();

    for node in &scene.nodes {
        let Some(mesh) = node.mesh.as_ref() else {
            continue;
        };
        if mesh.num_faces == 0 {
            continue;
        }
        let mut partitions = HashMap::<u32, tobj::Mesh>::new();
        let mut triangle_indices = Vec::new();
        for (face_index, &face) in mesh.faces.iter().enumerate() {
            triangle_indices.clear();
            ufbx::triangulate_face_vec(&mut triangle_indices, mesh, face);
            let material_index = mesh.face_material.get(face_index).copied().unwrap_or(0);
            let output = partitions.entry(material_index).or_default();
            for &corner_index in &triangle_indices {
                let corner_index = corner_index as usize;
                let source_position = mesh.vertex_position[corner_index];
                let position = ufbx::transform_position(&node.geometry_to_world, source_position);
                output.positions.extend_from_slice(&[
                    position.x as f32,
                    position.y as f32,
                    position.z as f32,
                ]);
                if mesh.vertex_uv.exists {
                    let uv = mesh.vertex_uv[corner_index];
                    output
                        .texcoords
                        .extend_from_slice(&[uv.x as f32, uv.y as f32]);
                }
                output.indices.push(output.indices.len() as u32);
            }
        }
        let partition_count = partitions.len();
        for (material_index, mut mesh) in partitions {
            if mesh.indices.is_empty() {
                continue;
            }
            let node_name: &str = node.element.name.as_ref();
            mesh.material_id = Some(material_index as usize);
            let name = if partition_count > 1 {
                format!("{node_name} · Material {}", material_index + 1)
            } else if node_name.is_empty() {
                "FBX Mesh".to_owned()
            } else {
                node_name.to_owned()
            };
            imported_models.push(tobj::Model { name, mesh });
        }
    }

    if imported_models.is_empty() {
        bail!("FBX contains no polygon mesh geometry");
    }
    let label = path.to_string_lossy();
    build_model(imported_models, Vec::new(), &label)
}

fn build_model(
    models: Vec<tobj::Model>,
    imported_materials: Vec<tobj::Material>,
    file_name: &str,
) -> anyhow::Result<PreparedModel> {
    build_model_with_normals(models, imported_materials, file_name, false)
}

fn build_model_with_normals(
    models: Vec<tobj::Model>, imported_materials: Vec<tobj::Material>, file_name: &str,
    preserve_normals: bool,
) -> anyhow::Result<PreparedModel> {
    let materials = imported_materials
        .into_iter()
        .map(|material| model::MaterialSource {
            name: material.name,
            diffuse: material.diffuse,
            pbr: None,
            diffuse_texture: material.diffuse_texture,
            normal_texture: material.normal_texture,
            emissive_texture: material
                .unknown_param
                .get("map_Ke")
                .cloned()
                .unwrap_or_default(),
            roughness_texture: material
                .unknown_param
                .get("map_Pr")
                .or_else(|| material.unknown_param.get("map_roughness"))
                .cloned()
                .unwrap_or_default(),
        })
        .collect();
    let repack_overlapping_uvs = authored_uv_spaces_overlap(&models);
    let mut occupied_tiles = if repack_overlapping_uvs {
        HashSet::new()
    } else {
        models
            .iter()
            .flat_map(authored_uv_tiles)
            .collect::<HashSet<_>>()
    };

    let mut meshes = Vec::with_capacity(models.len());
    for (model_index, imported) in models.into_iter().enumerate() {
        let object_name = if imported.name.trim().is_empty() {
            format!("Group {}", model_index + 1)
        } else {
            imported.name.clone()
        };
        let source = imported.mesh;
        if source.positions.is_empty() && source.indices.is_empty() {
            // OBJ files commonly contain named groups with no faces. They are
            // metadata, not renderable geometry, and must not create zero-size
            // GPU buffers or invalid infinite bounds.
            continue;
        }
        if source.positions.len() % 3 != 0 {
            bail!(
                "OBJ object `{object_name}` has {} position values; positions must contain complete xyz triples",
                source.positions.len()
            );
        }
        let vertex_count = source.positions.len() / 3;
        if vertex_count == 0 {
            bail!("OBJ object `{object_name}` contains faces but no vertices");
        }
        if source.indices.is_empty() {
            // This viewport renders polygonal surfaces. Ignore point/line-only
            // groups while retaining every group that contains faces.
            continue;
        }
        validate_triangle_indices(&source.indices, vertex_count, &object_name)?;
        if let Some((value_index, _)) = source
            .positions
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            bail!(
                "OBJ object `{object_name}` has a non-finite position at component {value_index}"
            );
        }

        let expected_normal_values = vertex_count * 3;
        let expected_texcoord_values = vertex_count * 2;
        let has_authored_normals = source.normals.len() == expected_normal_values
            && source.normals.iter().all(|value| value.is_finite());
        let has_texture_coordinates = source.texcoords.len() == expected_texcoord_values
            && source.texcoords.iter().all(|value| value.is_finite());

        // OBJ assets are conventionally authored Y-up, while this viewport
        // uses Z-up. A +90 degree rotation around X maps (x, y, z) to
        // (x, -z, y). Apply it to geometry before tangent generation so the
        // complete shading basis uses the same coordinate system.
        let vertices = (0..vertex_count)
            .map(|i| {
                let tex_coords = if has_texture_coordinates {
                    [
                        source.texcoords[i * 2],
                        flip_uv_v_preserving_tile(source.texcoords[i * 2 + 1]),
                    ]
                } else {
                    [0.0, 0.0]
                };
                if !has_authored_normals {
                    model::ModelVertex {
                        position: [
                            source.positions[i * 3],
                            -source.positions[i * 3 + 2],
                            source.positions[i * 3 + 1],
                        ],
                        tex_coords,
                        normal: [0.0, 0.0, 0.0],
                        tangent: [1.0, 0.0, 0.0, 1.0],
                    }
                } else {
                    model::ModelVertex {
                        position: [
                            source.positions[i * 3],
                            -source.positions[i * 3 + 2],
                            source.positions[i * 3 + 1],
                        ],
                        tex_coords,
                        normal: [
                            source.normals[i * 3],
                            -source.normals[i * 3 + 2],
                            source.normals[i * 3 + 1],
                        ],
                        tangent: [1.0, 0.0, 0.0, 1.0],
                    }
                }
            })
            .collect::<Vec<_>>();
        let mut vertices = vertices;
        let uv_group_index = meshes.len();
        let sequential_tile = ((uv_group_index % 10) as i32, (uv_group_index / 10) as i32);
        let uv_tile = if repack_overlapping_uvs {
            remap_group_uvs(&mut vertices, has_texture_coordinates, sequential_tile);
            sequential_tile
        } else if has_texture_coordinates {
            primary_authored_uv_tile(&source).unwrap_or(sequential_tile)
        } else {
            let mut candidate_index = 0_usize;
            let generated_tile = loop {
                let candidate = ((candidate_index % 10) as i32, (candidate_index / 10) as i32);
                if occupied_tiles.insert(candidate) {
                    break candidate;
                }
                candidate_index += 1;
            };
            remap_group_uvs(&mut vertices, false, generated_tile);
            generated_tile
        };
        // Rebuild a continuous viewport normal field even when the OBJ carries
        // split/flat export normals. This is Phong-style smooth shading: only
        // the shading basis changes; polygon positions and UVs remain exact.
        if !preserve_normals || !has_authored_normals {
            calculate_normals(&mut vertices, &source.indices);
        }
        calculate_tangents(&mut vertices, &source.indices);

        let mut bounds_min = [f32::INFINITY; 3];
        let mut bounds_max = [f32::NEG_INFINITY; 3];
        for vertex in &vertices {
            for axis in 0..3 {
                bounds_min[axis] = bounds_min[axis].min(vertex.position[axis]);
                bounds_max[axis] = bounds_max[axis].max(vertex.position[axis]);
            }
        }
        let geometry_stats = geometry_stats(&vertices, &source.indices, true, has_authored_normals);
        meshes.push(PreparedMesh {
            name: object_name,
            vertices,
            indices: source.indices,
            uv_tile,
            bounds_min,
            bounds_max,
            geometry_stats,
            material: source.material_id.unwrap_or(0),
        });
    }

    if meshes.is_empty() {
        bail!("OBJ `{file_name}` contains no triangulated surface geometry");
    }

    Ok(PreparedModel { meshes, materials, import_warnings: Vec::new(), imported_scene: None })
}

pub(crate) fn upload_prepared_model(
    prepared: PreparedModel,
    file_name: &str,
    device: &wgpu::Device,
) -> anyhow::Result<model::Model> {
    let mut meshes = Vec::with_capacity(prepared.meshes.len());
    for mesh in prepared.meshes {
        let create_vertex_buffer = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::VERTEX,
            })
        };
        let vertex_buffer = create_vertex_buffer(
            &format!("{file_name:?} vertex buffer"),
            bytemuck::cast_slice(&mesh.vertices),
        );
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{file_name:?} index buffer")),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        meshes.push(model::Mesh {
            name: mesh.name,
            vertex_buffer,
            index_buffer,
            num_elements: mesh.indices.len() as u32,
            point_marker_buffer: None,
            point_marker_count: 0,
            vertex_normal_buffer: None,
            vertex_normal_count: 0,
            point_normal_buffer: None,
            point_normal_count: 0,
            face_normal_buffer: None,
            face_normal_count: 0,
            uv_edges: Vec::new(),
            source_vertices: mesh.vertices,
            source_indices: mesh.indices,
            uv_tile: mesh.uv_tile,
            bounds_min: mesh.bounds_min,
            bounds_max: mesh.bounds_max,
            geometry_stats: mesh.geometry_stats,
            material: mesh.material,
        });
    }
    Ok(model::Model {
        meshes,
        materials: prepared.materials,
        import_warnings: prepared.import_warnings,
        imported_scene: prepared.imported_scene,
    })
}

fn validate_triangle_indices(
    indices: &[u32],
    vertex_count: usize,
    object_name: &str,
) -> anyhow::Result<()> {
    if indices.len() % 3 != 0 {
        bail!(
            "OBJ object `{object_name}` produced {} indices after triangulation; expected complete triangles",
            indices.len()
        );
    }
    if let Some((index_position, index)) = indices
        .iter()
        .copied()
        .enumerate()
        .find(|(_, index)| *index as usize >= vertex_count)
    {
        bail!(
            "OBJ object `{object_name}` references vertex {index} at index {index_position}, but only {vertex_count} vertices exist"
        );
    }
    Ok(())
}

fn authored_uv_tiles(imported: &tobj::Model) -> HashSet<(i32, i32)> {
    authored_uv_tiles_from_mesh(&imported.mesh)
}

fn authored_uv_tiles_from_mesh(mesh: &tobj::Mesh) -> HashSet<(i32, i32)> {
    let vertex_count = mesh.positions.len() / 3;
    if mesh.texcoords.len() != vertex_count * 2 {
        return HashSet::new();
    }
    mesh.indices
        .chunks_exact(3)
        .filter_map(|triangle| {
            let indices = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];
            if indices.iter().any(|&index| index >= vertex_count) {
                return None;
            }
            let center = [
                indices
                    .iter()
                    .map(|&index| mesh.texcoords[index * 2])
                    .sum::<f32>()
                    / 3.0,
                indices
                    .iter()
                    .map(|&index| mesh.texcoords[index * 2 + 1])
                    .sum::<f32>()
                    / 3.0,
            ];
            center
                .iter()
                .all(|value| value.is_finite())
                .then_some((center[0].floor() as i32, center[1].floor() as i32))
        })
        .collect()
}

fn authored_uv_spaces_overlap(models: &[tobj::Model]) -> bool {
    let mut owners = HashMap::<(i32, i32), usize>::new();
    for (group_index, imported) in models.iter().enumerate() {
        for tile in authored_uv_tiles(imported) {
            if owners
                .insert(tile, group_index)
                .is_some_and(|owner| owner != group_index)
            {
                return true;
            }
        }
    }
    false
}

fn primary_authored_uv_tile(mesh: &tobj::Mesh) -> Option<(i32, i32)> {
    authored_uv_tiles_from_mesh(mesh).into_iter().min()
}

fn flip_uv_v_preserving_tile(value: f32) -> f32 {
    let tile = value.floor();
    tile + 1.0 - (value - tile)
}

fn remap_group_uvs(vertices: &mut [model::ModelVertex], has_authored_uvs: bool, tile: (i32, i32)) {
    const MARGIN: f32 = 0.01;
    if vertices.is_empty() {
        return;
    }

    let source_uvs = if has_authored_uvs {
        vertices
            .iter()
            .map(|vertex| vertex.tex_coords)
            .collect::<Vec<_>>()
    } else {
        // Project along the thinnest bounds axis, exposing the two dimensions
        // with the greatest geometric extent. This gives UV-less OBJ groups a
        // stable, useful inspection map without changing their geometry.
        let (bounds_min, bounds_max) = vertex_bounds(vertices);
        let extents = [
            bounds_max[0] - bounds_min[0],
            bounds_max[1] - bounds_min[1],
            bounds_max[2] - bounds_min[2],
        ];
        let dropped_axis = (0..3)
            .min_by(|&left, &right| extents[left].total_cmp(&extents[right]))
            .unwrap_or(2);
        let axes = match dropped_axis {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        vertices
            .iter()
            .map(|vertex| [vertex.position[axes[0]], vertex.position[axes[1]]])
            .collect::<Vec<_>>()
    };

    let mut minimum = [f32::INFINITY; 2];
    let mut maximum = [f32::NEG_INFINITY; 2];
    for uv in &source_uvs {
        for axis in 0..2 {
            minimum[axis] = minimum[axis].min(uv[axis]);
            maximum[axis] = maximum[axis].max(uv[axis]);
        }
    }
    let extent = [maximum[0] - minimum[0], maximum[1] - minimum[1]];
    let largest_extent = extent[0].max(extent[1]);
    let scale = if largest_extent > 1.0e-8 {
        (1.0 - MARGIN * 2.0) / largest_extent
    } else {
        0.0
    };
    let used_size = [extent[0] * scale, extent[1] * scale];
    let centering = [
        MARGIN + (1.0 - MARGIN * 2.0 - used_size[0]) * 0.5,
        MARGIN + (1.0 - MARGIN * 2.0 - used_size[1]) * 0.5,
    ];

    for (vertex, source_uv) in vertices.iter_mut().zip(source_uvs) {
        vertex.tex_coords = [
            tile.0 as f32 + centering[0] + (source_uv[0] - minimum[0]) * scale,
            tile.1 as f32 + centering[1] + (source_uv[1] - minimum[1]) * scale,
        ];
    }
}

fn vertex_bounds(vertices: &[model::ModelVertex]) -> ([f32; 3], [f32; 3]) {
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(vertex.position[axis]);
            maximum[axis] = maximum[axis].max(vertex.position[axis]);
        }
    }
    (minimum, maximum)
}

fn normal_arrow(origin: [f32; 3], normal: [f32; 3]) -> [model::NormalMarkerVertex; 6] {
    let marker = |along, width| model::NormalMarkerVertex {
        origin,
        normal,
        shape: [along, width],
    };
    // Two triangles form a ribbon that tapers to 12% of its base width.
    [
        marker(0.0, -1.0),
        marker(0.0, 1.0),
        marker(1.0, -0.12),
        marker(0.0, 1.0),
        marker(1.0, 0.12),
        marker(1.0, -0.12),
    ]
}

pub(crate) fn normal_markers(
    vertices: &[model::ModelVertex],
    indices: &[u32],
) -> (
    Vec<model::NormalMarkerVertex>,
    Vec<model::NormalMarkerVertex>,
    Vec<model::NormalMarkerVertex>,
) {
    let vertex_markers = vertices
        .iter()
        .flat_map(|vertex| normal_arrow(vertex.position, vertex.normal))
        .collect();

    let mut point_accum = HashMap::<[u32; 3], ([f32; 3], [f32; 3])>::new();
    for vertex in vertices {
        let entry = point_accum
            .entry(position_key(vertex.position))
            .or_insert((vertex.position, [0.0; 3]));
        entry.1 = add3(entry.1, vertex.normal);
    }
    let point_markers = point_accum
        .into_values()
        .flat_map(|(position, normal)| {
            normal_arrow(position, normalize3_or(normal, [0.0, 0.0, 1.0]))
        })
        .collect();

    let face_markers = indices
        .chunks_exact(3)
        .flat_map(|triangle| {
            let p0 = vertices[triangle[0] as usize].position;
            let p1 = vertices[triangle[1] as usize].position;
            let p2 = vertices[triangle[2] as usize].position;
            let center = [
                (p0[0] + p1[0] + p2[0]) / 3.0,
                (p0[1] + p1[1] + p2[1]) / 3.0,
                (p0[2] + p1[2] + p2[2]) / 3.0,
            ];
            let normal = normalize3_or(
                cross3(subtract3(p1, p0), subtract3(p2, p0)),
                [0.0, 0.0, 1.0],
            );
            normal_arrow(center, normal)
        })
        .collect();
    (vertex_markers, point_markers, face_markers)
}

fn geometry_stats(
    vertices: &[model::ModelVertex],
    indices: &[u32],
    has_texture_coordinates: bool,
    has_authored_normals: bool,
) -> model::GeometryStats {
    let mut point_ids = HashMap::<[u32; 3], u32>::new();
    let mut vertex_points = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        let key = position_key(vertex.position);
        let next_id = point_ids.len() as u32;
        vertex_points.push(*point_ids.entry(key).or_insert(next_id));
    }

    let mut edges = HashMap::<(u32, u32), u32>::new();
    let mut surface_area = 0.0_f64;
    let mut signed_volume = 0.0_f64;
    let mut degenerate_triangle_count = 0;
    for triangle in indices.chunks_exact(3) {
        let vertex_indices = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let points = vertex_indices.map(|index| vertex_points[index]);
        for (a, b) in [
            (points[0], points[1]),
            (points[1], points[2]),
            (points[2], points[0]),
        ] {
            let edge = if a <= b { (a, b) } else { (b, a) };
            *edges.entry(edge).or_default() += 1;
        }

        let p0 = vertices[vertex_indices[0]].position;
        let p1 = vertices[vertex_indices[1]].position;
        let p2 = vertices[vertex_indices[2]].position;
        let cross = cross3(subtract3(p1, p0), subtract3(p2, p0));
        let double_area = f64::from(dot3(cross, cross)).sqrt();
        if double_area <= 1.0e-10 {
            degenerate_triangle_count += 1;
        }
        surface_area += double_area * 0.5;
        signed_volume += f64::from(dot3(p0, cross3(p1, p2))) / 6.0;
    }

    let boundary_edge_count = edges.values().filter(|&&uses| uses == 1).count();
    let non_manifold_edge_count = edges.values().filter(|&&uses| uses > 2).count();
    let is_closed = boundary_edge_count == 0 && non_manifold_edge_count == 0;
    model::GeometryStats {
        point_count: point_ids.len(),
        vertex_count: vertices.len(),
        index_count: indices.len(),
        triangle_count: indices.len() / 3,
        edge_count: edges.len(),
        boundary_edge_count,
        non_manifold_edge_count,
        degenerate_triangle_count,
        uv_vertex_count: usize::from(has_texture_coordinates) * vertices.len(),
        uv_seam_vertex_count: vertices.len().saturating_sub(point_ids.len()),
        authored_normals: has_authored_normals,
        surface_area,
        enclosed_volume: is_closed.then_some(signed_volume.abs()),
    }
}

fn calculate_normals(vertices: &mut [model::ModelVertex], indices: &[u32]) {
    // `single_index` duplicates positions at UV seams. Accumulate by position
    // so an otherwise smooth surface does not acquire a lighting seam merely
    // because its texture coordinates wrap there. The unnormalized face cross
    // normals are angle-weighted, preventing large or skinny triangulation
    // faces from dominating the interpolated surface direction.
    let mut accumulated = HashMap::<[u32; 3], [f32; 3]>::new();
    for triangle in indices.chunks_exact(3) {
        let [i0, i1, i2] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let [p0, p1, p2] = [
            vertices[i0].position,
            vertices[i1].position,
            vertices[i2].position,
        ];
        let edge1 = subtract3(p1, p0);
        let edge2 = subtract3(p2, p0);
        let face_cross = cross3(edge1, edge2);
        if dot3(face_cross, face_cross) <= 1.0e-20 {
            continue;
        }
        let face_normal = normalize3_or(face_cross, [0.0, 0.0, 1.0]);
        for (index, left, right) in [(i0, p1, p2), (i1, p2, p0), (i2, p0, p1)] {
            let corner = vertices[index].position;
            let left_edge = normalize3_or(subtract3(left, corner), [1.0, 0.0, 0.0]);
            let right_edge = normalize3_or(subtract3(right, corner), [0.0, 1.0, 0.0]);
            let angle = dot3(left_edge, right_edge).clamp(-1.0, 1.0).acos();
            let normal = accumulated
                .entry(position_key(vertices[index].position))
                .or_insert([0.0; 3]);
            *normal = add3(*normal, scale3(face_normal, angle));
        }
    }

    for vertex in vertices {
        let normal = accumulated
            .get(&position_key(vertex.position))
            .copied()
            .unwrap_or([0.0, 0.0, 1.0]);
        vertex.normal = normalize3_or(normal, [0.0, 0.0, 1.0]);
    }
}

fn calculate_tangents(vertices: &mut [model::ModelVertex], indices: &[u32]) {
    let mut tangents = vec![[0.0_f32; 3]; vertices.len()];
    let mut bitangents = vec![[0.0_f32; 3]; vertices.len()];
    for triangle in indices.chunks_exact(3) {
        let [i0, i1, i2] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let [v0, v1, v2] = [vertices[i0], vertices[i1], vertices[i2]];
        let edge1 = [
            v1.position[0] - v0.position[0],
            v1.position[1] - v0.position[1],
            v1.position[2] - v0.position[2],
        ];
        let edge2 = [
            v2.position[0] - v0.position[0],
            v2.position[1] - v0.position[1],
            v2.position[2] - v0.position[2],
        ];
        let duv1 = [
            v1.tex_coords[0] - v0.tex_coords[0],
            v1.tex_coords[1] - v0.tex_coords[1],
        ];
        let duv2 = [
            v2.tex_coords[0] - v0.tex_coords[0],
            v2.tex_coords[1] - v0.tex_coords[1],
        ];
        let determinant = duv1[0] * duv2[1] - duv1[1] * duv2[0];
        if determinant.abs() < 1.0e-8 {
            continue;
        }
        let reciprocal = determinant.recip();
        let tangent = [
            (edge1[0] * duv2[1] - edge2[0] * duv1[1]) * reciprocal,
            (edge1[1] * duv2[1] - edge2[1] * duv1[1]) * reciprocal,
            (edge1[2] * duv2[1] - edge2[2] * duv1[1]) * reciprocal,
        ];
        let bitangent = [
            (edge2[0] * duv1[0] - edge1[0] * duv2[0]) * reciprocal,
            (edge2[1] * duv1[0] - edge1[1] * duv2[0]) * reciprocal,
            (edge2[2] * duv1[0] - edge1[2] * duv2[0]) * reciprocal,
        ];
        for &index in triangle {
            let index = index as usize;
            tangents[index] = add3(tangents[index], tangent);
            bitangents[index] = add3(bitangents[index], bitangent);
        }
    }

    for ((vertex, tangent), bitangent) in vertices.iter_mut().zip(tangents).zip(bitangents) {
        let normal = vertex.normal;
        // Gram-Schmidt keeps the tangent basis orthonormal even on meshes with
        // distorted UVs. Use a deterministic axis for missing/degenerate UVs.
        let orthogonal_tangent = subtract3(tangent, scale3(normal, dot3(normal, tangent)));
        let fallback = tangent_from_normal(normal);
        let tangent = normalize3_or(orthogonal_tangent, fallback);
        let handedness = if dot3(cross3(normal, tangent), bitangent) < 0.0 {
            -1.0
        } else {
            1.0
        };
        vertex.tangent = [tangent[0], tangent[1], tangent[2], handedness];
    }
}

fn position_key(position: [f32; 3]) -> [u32; 3] {
    position.map(f32::to_bits)
}

fn add3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn subtract3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale3(vector: [f32; 3], scalar: f32) -> [f32; 3] {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize3_or(vector: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = dot3(vector, vector);
    if length_squared <= 1.0e-20 {
        return fallback;
    }
    scale3(vector, length_squared.sqrt().recip())
}

fn tangent_from_normal(normal: [f32; 3]) -> [f32; 3] {
    let reference = if normal[2].abs() < 0.999 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize3_or(cross3(reference, normal), [1.0, 0.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    #[ignore = "local USD benchmark; set LEARN_WGPU_USD_BENCHMARK_ASSET"]
    fn benchmark_usd_preparation() {
        let Some(path) = std::env::var_os("LEARN_WGPU_USD_BENCHMARK_ASSET") else {
            return;
        };
        let started = std::time::Instant::now();
        let prepared = prepare_model_from_path(std::path::Path::new(&path), None).unwrap();
        let elapsed = started.elapsed();
        eprintln!(
            "USD preparation: {:.3}s, {} groups, {} triangles, {} materials",
            elapsed.as_secs_f64(), prepared.meshes.len(),
            prepared.meshes.iter().map(|mesh| mesh.indices.len() / 3).sum::<usize>(),
            prepared.materials.len(),
        );
        assert!(elapsed.as_secs_f64() < 20.0, "USD preparation exceeded 20 seconds: {elapsed:?}");
    }

    #[test]
    fn generates_a_valid_basis_for_meshes_without_normals() {
        let mut vertices = [
            vertex([-1.0, -1.0, 0.0], [0.0, 0.0]),
            vertex([1.0, -1.0, 0.0], [1.0, 0.0]),
            vertex([1.0, 1.0, 0.0], [1.0, 1.0]),
            vertex([-1.0, 1.0, 0.0], [0.0, 1.0]),
        ];
        let indices = [0, 1, 2, 0, 2, 3];

        calculate_normals(&mut vertices, &indices);
        calculate_tangents(&mut vertices, &indices);

        for vertex in vertices {
            assert!(vertex.normal[2] > 0.999);
            assert!((dot3(vertex.normal, vertex.normal) - 1.0).abs() < 1.0e-5);
            let tangent = [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]];
            assert!((dot3(tangent, tangent) - 1.0).abs() < 1.0e-5);
            assert!(dot3(vertex.normal, tangent).abs() < 1.0e-5);
        }
    }

    #[test]
    fn geometry_stats_distinguish_points_edges_and_triangles() {
        let vertices = vec![
            vertex([-1.0, -1.0, 0.0], [0.0, 0.0]),
            vertex([1.0, -1.0, 0.0], [1.0, 0.0]),
            vertex([1.0, 1.0, 0.0], [1.0, 1.0]),
            vertex([-1.0, 1.0, 0.0], [0.0, 1.0]),
        ];
        let stats = geometry_stats(&vertices, &[0, 1, 2, 0, 2, 3], true, false);
        assert_eq!(stats.point_count, 4);
        assert_eq!(stats.vertex_count, 4);
        assert_eq!(stats.triangle_count, 2);
        assert_eq!(stats.edge_count, 5);
        assert_eq!(stats.boundary_edge_count, 4);
        assert_eq!(stats.non_manifold_edge_count, 0);
        assert!((stats.surface_area - 4.0).abs() < 1.0e-6);
        assert!(stats.enclosed_volume.is_none());
    }

    #[test]
    fn validates_triangle_topology_before_cpu_or_gpu_indexing() {
        assert!(validate_triangle_indices(&[0, 1, 2, 2, 3, 0], 4, "valid").is_ok());
        assert!(validate_triangle_indices(&[0, 1], 4, "incomplete").is_err());
        assert!(validate_triangle_indices(&[0, 1, 4], 4, "out-of-range").is_err());
    }

    #[test]
    fn parser_preserves_multiple_obj_objects() {
        let source = b"o first\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n\
                       o second\nv 0 0 1\nv 1 0 1\nv 0 1 1\nf 4 5 6\n";
        let (models, _) = tobj::load_obj_buf(
            &mut BufReader::new(source.as_slice()),
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |_| Err(tobj::LoadError::OpenFileFailed),
        )
        .expect("multi-object OBJ must parse");

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "first");
        assert_eq!(models[1].name, "second");
        assert_eq!(
            models
                .iter()
                .map(|model| model.mesh.indices.len())
                .sum::<usize>(),
            6
        );
    }

    #[test]
    fn outdated_model_cache_is_rejected_before_decoding_materials() {
        let mut bytes = (MODEL_CACHE_VERSION - 1).to_le_bytes().to_vec();
        bytes.extend_from_slice(&[255; 64]);
        let error = read_model_cache(std::io::Cursor::new(bytes)).err().unwrap();
        assert!(error.to_string().contains("Outdated"));
    }

    #[test]
    fn processed_model_cache_round_trips_without_losing_render_data() {
        let source =
            b"o cached\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n";
        let (models, materials) = tobj::load_obj_buf(
            &mut BufReader::new(source.as_slice()),
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |_| Err(tobj::LoadError::OpenFileFailed),
        )
        .unwrap();
        let prepared = build_model(models, materials.unwrap_or_default(), "cached.obj").unwrap();
        let cache = ModelCache {
            version: MODEL_CACHE_VERSION,
            source_len: source.len() as u64,
            source_modified_ns: 123,
            model: prepared,
        };
        let bytes = bincode::serialize(&cache).unwrap();
        let restored = read_model_cache(std::io::Cursor::new(&bytes)).unwrap();

        assert_eq!(restored.version, MODEL_CACHE_VERSION);
        assert_eq!(restored.model.meshes.len(), 1);
        let mesh = &restored.model.meshes[0];
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices, [0, 1, 2]);
        assert_eq!(mesh.geometry_stats.triangle_count, 1);
        // Inspection overlays are intentionally absent from the persistent
        // cache; they are derived lazily only when their viewport feature is on.
        assert!(bytes.len() < 1_024);
    }

    #[test]
    fn obj_groups_are_fitted_into_distinct_udim_tiles() {
        let mut first = [
            vertex([-1.0, -1.0, 0.0], [-2.0, 4.0]),
            vertex([1.0, -1.0, 0.0], [3.0, 4.0]),
            vertex([0.0, 1.0, 0.0], [0.0, 8.0]),
        ];
        let mut second = first;

        remap_group_uvs(&mut first, true, (0, 0));
        remap_group_uvs(&mut second, true, (1, 0));

        assert!(first.iter().all(|vertex| {
            vertex.tex_coords[0] > 0.0
                && vertex.tex_coords[0] < 1.0
                && vertex.tex_coords[1] > 0.0
                && vertex.tex_coords[1] < 1.0
        }));
        assert!(second.iter().all(|vertex| {
            vertex.tex_coords[0] > 1.0
                && vertex.tex_coords[0] < 2.0
                && vertex.tex_coords[1] > 0.0
                && vertex.tex_coords[1] < 1.0
        }));
    }

    #[test]
    fn uvless_group_receives_planar_uvs_in_its_assigned_tile() {
        let mut vertices = [
            vertex([-2.0, -1.0, 0.0], [0.0, 0.0]),
            vertex([2.0, -1.0, 0.0], [0.0, 0.0]),
            vertex([0.0, 1.0, 0.0], [0.0, 0.0]),
        ];
        remap_group_uvs(&mut vertices, false, (2, 1));

        assert!(vertices.iter().all(|vertex| {
            vertex.tex_coords[0] > 2.0
                && vertex.tex_coords[0] < 3.0
                && vertex.tex_coords[1] > 1.0
                && vertex.tex_coords[1] < 2.0
        }));
        assert_ne!(vertices[0].tex_coords, vertices[1].tex_coords);
    }

    #[test]
    fn preserves_distinct_authored_uv_spaces_and_detects_collisions() {
        let group = |name: &str, tile_u: f32| {
            let mut mesh = tobj::Mesh::default();
            mesh.positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            mesh.texcoords = vec![tile_u + 0.1, 0.1, tile_u + 0.9, 0.1, tile_u + 0.1, 0.9];
            mesh.indices = vec![0, 1, 2];
            tobj::Model {
                name: name.to_owned(),
                mesh,
            }
        };

        let distinct = vec![group("body", 0.0), group("eyes", 1.0)];
        assert!(!authored_uv_spaces_overlap(&distinct));
        assert_eq!(primary_authored_uv_tile(&distinct[0].mesh), Some((0, 0)));
        assert_eq!(primary_authored_uv_tile(&distinct[1].mesh), Some((1, 0)));

        let overlapping = vec![group("body", 0.0), group("eyes", 0.0)];
        assert!(authored_uv_spaces_overlap(&overlapping));
    }

    #[test]
    fn parses_standard_and_pbr_mtl_texture_entries() {
        let source = b"newmtl skin\nKd 0.8 0.6 0.5\nmap_Kd skin_color.png\nmap_Bump skin_normal.png\nmap_Pr skin_roughness.png\n";
        let (materials, names) =
            tobj::load_mtl_buf(&mut BufReader::new(source.as_slice())).expect("MTL must parse");
        let material = &materials[names["skin"]];
        assert_eq!(material.diffuse_texture, "skin_color.png");
        assert_eq!(material.normal_texture, "skin_normal.png");
        assert_eq!(
            material.unknown_param.get("map_Pr").map(String::as_str),
            Some("skin_roughness.png")
        );
    }

    fn vertex(position: [f32; 3], tex_coords: [f32; 2]) -> model::ModelVertex {
        model::ModelVertex {
            position,
            tex_coords,
            normal: [0.0; 3],
            tangent: [1.0, 0.0, 0.0, 1.0],
        }
    }
}
