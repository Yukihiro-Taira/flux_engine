use std::{
    collections::{HashMap, HashSet},
    io::{BufReader, Cursor},
};

use anyhow::{Context, Ok, bail};
use wgpu::util::DeviceExt;

use crate::{model, texture};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    if !origin.ends_with("learn-wgpu") {
        origin = format!("{}/learn-wgpu", origin);
    }
    let base = reqwest::Url::parse(&format!("{}/", origin,)).unwrap();
    base.join(file_name).unwrap()
}
//----------------------------------------//
//LOAD functions

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    #[cfg(target_arch = "wasm32")]
    let txt = {
        let url = format_url(file_name);
        reqwest::get(url).await?.text().await?
    };
    #[cfg(not(target_arch = "wasm32"))]
    let txt = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read_to_string(path)?
    };

    Ok(txt)
}

#[allow(dead_code)]
pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    let data = {
        let url = format_url(file_name);
        reqwest::get(url).await?.bytes().await?.to_vec()
    };
    #[cfg(not(target_arch = "wasm32"))]
    let data = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read(path)?
    };

    Ok(data)
}

#[allow(dead_code)]
pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    let image = image::load_from_memory(&data)?;
    texture::Texture::from_bytes(device, queue, &image, Some(file_name))
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    _layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |material_path| async move {
            let material_text = load_string(&material_path)
                .await
                .map_err(|_| tobj::LoadError::OpenFileFailed)?;
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(material_text)))
        },
    )
    .await?;

    let materials = obj_materials?;
    build_model(models, materials, file_name, device)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_model_from_path(
    path: &std::path::Path,
    device: &wgpu::Device,
) -> anyhow::Result<model::Model> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("fbx"))
    {
        return load_fbx_from_path(path, device);
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
    build_model(models, materials.unwrap_or_default(), &label, device)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_fbx_from_path(
    path: &std::path::Path,
    device: &wgpu::Device,
) -> anyhow::Result<model::Model> {
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
    build_model(imported_models, Vec::new(), &label, device)
}

fn build_model(
    models: Vec<tobj::Model>,
    imported_materials: Vec<tobj::Material>,
    file_name: &str,
    device: &wgpu::Device,
) -> anyhow::Result<model::Model> {
    let materials = imported_materials
        .into_iter()
        .map(|material| model::MaterialSource {
            name: material.name,
            diffuse: material.diffuse,
            diffuse_texture: material.diffuse_texture,
            normal_texture: material.normal_texture,
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
        calculate_normals(&mut vertices, &source.indices);
        calculate_tangents(&mut vertices, &source.indices);

        let marker_corners = [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ];
        let point_markers = vertices
            .iter()
            .flat_map(|vertex| {
                marker_corners.map(|corner| model::PointMarkerVertex {
                    center: vertex.position,
                    corner,
                })
            })
            .collect::<Vec<_>>();
        let (vertex_normals, point_normals, face_normals) =
            normal_markers(&vertices, &source.indices);
        let uv_edges = source
            .indices
            .chunks_exact(3)
            .flat_map(|triangle| {
                let uv = [
                    vertices[triangle[0] as usize].tex_coords,
                    vertices[triangle[1] as usize].tex_coords,
                    vertices[triangle[2] as usize].tex_coords,
                ];
                [(uv[0], uv[1]), (uv[1], uv[2]), (uv[2], uv[0])]
            })
            .collect();
        let mut bounds_min = [f32::INFINITY; 3];
        let mut bounds_max = [f32::NEG_INFINITY; 3];
        for vertex in &vertices {
            for axis in 0..3 {
                bounds_min[axis] = bounds_min[axis].min(vertex.position[axis]);
                bounds_max[axis] = bounds_max[axis].max(vertex.position[axis]);
            }
        }
        let geometry_stats = geometry_stats(&vertices, &source.indices, true, has_authored_normals);
        let point_marker_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point marker buffer"),
            contents: bytemuck::cast_slice(&point_markers),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let create_normal_buffer = |label, markers: &[model::NormalMarkerVertex]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(markers),
                usage: wgpu::BufferUsages::VERTEX,
            })
        };
        let vertex_normal_buffer =
            create_normal_buffer("Vertex normal marker buffer", &vertex_normals);
        let point_normal_buffer =
            create_normal_buffer("Point normal marker buffer", &point_normals);
        let face_normal_buffer = create_normal_buffer("Face normal marker buffer", &face_normals);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Index Buffer", file_name)),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Index Buffer", file_name)),
            contents: bytemuck::cast_slice(&source.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        meshes.push(model::Mesh {
            name: object_name,
            vertex_buffer,
            index_buffer,
            num_elements: source.indices.len() as u32,
            point_marker_buffer,
            point_marker_count: point_markers.len() as u32,
            vertex_normal_buffer,
            vertex_normal_count: vertex_normals.len() as u32,
            point_normal_buffer,
            point_normal_count: point_normals.len() as u32,
            face_normal_buffer,
            face_normal_count: face_normals.len() as u32,
            uv_edges,
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

    Ok(model::Model { meshes, materials })
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

fn normal_markers(
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
    use std::{fs::File, path::PathBuf};

    use super::*;

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

    #[test]
    fn t_pose_can_generate_finite_unit_normals() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("res/t-pose.obj");
        let mut reader = BufReader::new(File::open(path).expect("t-pose.obj must be available"));
        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |_| Err(tobj::LoadError::OpenFileFailed),
        )
        .expect("t-pose.obj must parse");
        assert!(!models.is_empty());

        for imported in models {
            assert!(imported.mesh.normals.is_empty());
            let mut vertices = (0..imported.mesh.positions.len() / 3)
                .map(|index| {
                    vertex(
                        [
                            imported.mesh.positions[index * 3],
                            -imported.mesh.positions[index * 3 + 2],
                            imported.mesh.positions[index * 3 + 1],
                        ],
                        [
                            imported.mesh.texcoords[index * 2],
                            1.0 - imported.mesh.texcoords[index * 2 + 1],
                        ],
                    )
                })
                .collect::<Vec<_>>();

            calculate_normals(&mut vertices, &imported.mesh.indices);
            calculate_tangents(&mut vertices, &imported.mesh.indices);

            for vertex in vertices {
                assert!(vertex.normal.iter().all(|component| component.is_finite()));
                assert!((dot3(vertex.normal, vertex.normal) - 1.0).abs() < 1.0e-4);
                let tangent = [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]];
                assert!(tangent.iter().all(|component| component.is_finite()));
                assert!((dot3(tangent, tangent) - 1.0).abs() < 1.0e-4);
                assert!(dot3(vertex.normal, tangent).abs() < 1.0e-4);
            }
        }
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
