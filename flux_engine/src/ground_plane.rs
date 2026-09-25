use cgmath::InnerSpace;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use wgpu::util::DeviceExt;

use crate::{InstanceRaw, material, model};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum BasicShape {
    Plane,
    Cube,
    Sphere,
    Cylinder,
    Cone,
    Torus,
}

impl BasicShape {
    pub const ALL: [Self; 6] = [
        Self::Plane,
        Self::Cube,
        Self::Sphere,
        Self::Cylinder,
        Self::Cone,
        Self::Torus,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Plane => "Plane",
            Self::Cube => "Cube",
            Self::Sphere => "UV Sphere",
            Self::Cylinder => "Cylinder",
            Self::Cone => "Cone",
            Self::Torus => "Torus",
        }
    }
}

pub struct GroundPlane {
    pub scene_id: u64,
    pub generated: bool,
    pub selected: bool,
    pub snap_bottom_to_grid: bool,
    pub kind: BasicShape,
    pub visible: bool,
    pub wireframe: bool,
    pub hide_surface: bool,
    pub size: f32,
    pub height: f32,
    pub position_x: f32,
    pub position_y: f32,
    pub rotation_degrees: [f32; 3],
    pub shape_height: f32,
    pub thickness: f32,
    pub subdivisions: u32,
    pub triangulate_subdivision: bool,
    pub uv_scale: f32,
    pub material: material::PbrMaterial,
    wire_material: material::PbrMaterial,
    pub base_color_path: String,
    pub normal_path: String,
    pub roughness_path: String,
    pub emissive_path: String,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    quad_edge_buffer: wgpu::Buffer,
    quad_edge_count: u32,
    instance_buffer: wgpu::Buffer,
    local_positions: Vec<[f32; 3]>,
    local_indices: Vec<u32>,
}

#[derive(Clone)]
pub struct SharedPrimitiveMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    quad_edge_buffer: wgpu::Buffer,
    quad_edge_count: u32,
    local_positions: Vec<[f32; 3]>,
    local_indices: Vec<u32>,
}

pub struct GeneratedGeometryStats {
    pub points: usize,
    pub vertices: usize,
    pub indices: usize,
    pub triangles: usize,
    pub edges: usize,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

impl GroundPlane {
    pub fn new_instance_from(device: &wgpu::Device, template: &Self) -> Self {
        let instance = ground_instance(BasicShape::Plane, 1.0, 1.0, 0.0);
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Generated object transform"),
            contents: bytemuck::bytes_of(&instance),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            scene_id: 0,
            generated: false,
            selected: false,
            snap_bottom_to_grid: false,
            kind: BasicShape::Plane,
            visible: true,
            wireframe: false,
            hide_surface: false,
            size: 1.0,
            height: 0.0,
            position_x: 0.0,
            position_y: 0.0,
            rotation_degrees: [0.0; 3],
            shape_height: 1.0,
            thickness: 0.0,
            subdivisions: 1,
            triangulate_subdivision: false,
            uv_scale: 1.0,
            material: material::PbrMaterial::new_instance_from(device, &template.material),
            wire_material: material::PbrMaterial::new_instance_from(
                device,
                &template.wire_material,
            ),
            base_color_path: String::new(),
            normal_path: String::new(),
            roughness_path: String::new(),
            emissive_path: String::new(),
            vertex_buffer: template.vertex_buffer.clone(),
            index_buffer: template.index_buffer.clone(),
            index_count: template.index_count,
            quad_edge_buffer: template.quad_edge_buffer.clone(),
            quad_edge_count: template.quad_edge_count,
            instance_buffer,
            local_positions: template.local_positions.clone(),
            local_indices: template.local_indices.clone(),
        }
    }

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<Self> {
        let (vertices, indices) = primitive_mesh(BasicShape::Plane, 1);
        let local_positions = vertices.iter().map(|vertex| vertex.position).collect();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ground plane vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ground plane indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let quad_edges = quad_edge_indices(&indices);
        let quad_edge_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Generated quad edges"),
            contents: bytemuck::cast_slice(&quad_edges),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instance = ground_instance(BasicShape::Plane, 1.0, 1.0, 0.0);
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ground plane transform"),
            contents: bytemuck::bytes_of(&instance),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let mut material = material::PbrMaterial::new_untextured(device, queue)?;
        material.uniform.base_color = [0.18, 0.18, 0.18, 1.0];
        material.uniform.properties[0] = 0.0;
        material.uniform.properties[1] = 0.65;
        material.uniform.options[2] = 1.0;
        material.upload(queue);
        let mut wire_material = material::PbrMaterial::new_untextured(device, queue)?;
        wire_material.uniform.base_color = [1.0, 0.32, 0.04, 1.0];
        wire_material.uniform.properties[0] = 0.0;
        wire_material.uniform.properties[1] = 1.0;
        wire_material.uniform.options[1] = 1.5;
        wire_material.uniform.options[2] = 0.0;
        wire_material.upload(queue);
        Ok(Self {
            scene_id: 0,
            generated: false,
            selected: false,
            snap_bottom_to_grid: false,
            kind: BasicShape::Plane,
            visible: true,
            wireframe: false,
            hide_surface: false,
            size: 1.0,
            height: 0.0,
            position_x: 0.0,
            position_y: 0.0,
            rotation_degrees: [0.0; 3],
            shape_height: 1.0,
            thickness: 0.0,
            subdivisions: 1,
            triangulate_subdivision: false,
            uv_scale: 1.0,
            material,
            wire_material,
            base_color_path: String::new(),
            normal_path: String::new(),
            roughness_path: String::new(),
            emissive_path: String::new(),
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            quad_edge_buffer,
            quad_edge_count: quad_edges.len() as u32,
            instance_buffer,
            local_positions,
            local_indices: indices,
        })
    }

    pub fn rebuild_shape(&mut self, device: &wgpu::Device) {
        let (vertices, mut indices) = primitive_mesh(self.kind, self.subdivisions);
        if self.triangulate_subdivision && matches!(self.kind, BasicShape::Plane | BasicShape::Cube)
        {
            for (cell, triangles) in indices.chunks_exact_mut(6).enumerate() {
                if cell % 2 == 1 {
                    let a = triangles[0];
                    let b = triangles[1];
                    let c = triangles[4];
                    let d = triangles[5];
                    triangles.copy_from_slice(&[a, b, d, b, c, d]);
                }
            }
        }
        self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Generated primitive vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Generated primitive indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.index_count = indices.len() as u32;
        let quad_edges = quad_edge_indices(&indices);
        self.quad_edge_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Generated quad edges"),
            contents: bytemuck::cast_slice(&quad_edges),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.quad_edge_count = quad_edges.len() as u32;
        self.local_positions = vertices.iter().map(|vertex| vertex.position).collect();
        self.local_indices = indices;
    }

    pub fn shared_mesh(&self) -> SharedPrimitiveMesh {
        SharedPrimitiveMesh {
            vertex_buffer: self.vertex_buffer.clone(),
            index_buffer: self.index_buffer.clone(),
            index_count: self.index_count,
            quad_edge_buffer: self.quad_edge_buffer.clone(),
            quad_edge_count: self.quad_edge_count,
            local_positions: self.local_positions.clone(),
            local_indices: self.local_indices.clone(),
        }
    }

    pub fn use_shared_mesh(&mut self, mesh: &SharedPrimitiveMesh) {
        self.vertex_buffer = mesh.vertex_buffer.clone();
        self.index_buffer = mesh.index_buffer.clone();
        self.index_count = mesh.index_count;
        self.quad_edge_buffer = mesh.quad_edge_buffer.clone();
        self.quad_edge_count = mesh.quad_edge_count;
        self.local_positions.clone_from(&mesh.local_positions);
        self.local_indices.clone_from(&mesh.local_indices);
    }

    pub fn geometry_stats(&self) -> GeneratedGeometryStats {
        let model = self.model_matrix();
        let mut point_keys = HashSet::new();
        let mut bounds_min = [f32::INFINITY; 3];
        let mut bounds_max = [f32::NEG_INFINITY; 3];
        for position in &self.local_positions {
            point_keys.insert(position.map(f32::to_bits));
            let world = model * cgmath::Vector4::new(position[0], position[1], position[2], 1.0);
            let world = [world.x / world.w, world.y / world.w, world.z / world.w];
            for axis in 0..3 {
                bounds_min[axis] = bounds_min[axis].min(world[axis]);
                bounds_max[axis] = bounds_max[axis].max(world[axis]);
            }
        }
        let mut edges = HashSet::new();
        for triangle in self.local_indices.chunks_exact(3) {
            for (a, b) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                edges.insert(if a <= b { (a, b) } else { (b, a) });
            }
        }
        GeneratedGeometryStats {
            points: point_keys.len(),
            vertices: self.local_positions.len(),
            indices: self.local_indices.len(),
            triangles: self.local_indices.len() / 3,
            edges: edges.len(),
            bounds_min,
            bounds_max,
        }
    }

    pub fn upload(&mut self, queue: &wgpu::Queue) {
        self.apply_grid_snap();
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::bytes_of(&InstanceRaw {
                model: self.model_matrix().into(),
            }),
        );
        self.material.uniform.udim = [0.0, 0.0, self.uv_scale.max(0.001), self.uv_scale.max(0.001)];
        self.material.upload(queue);
        self.wire_material.upload(queue);
    }

    fn apply_grid_snap(&mut self) {
        if !self.generated || !self.snap_bottom_to_grid || self.kind == BasicShape::Plane {
            return;
        }
        let model = self.model_matrix();
        let minimum_z = self
            .local_positions
            .iter()
            .map(|position| {
                let world =
                    model * cgmath::Vector4::new(position[0], position[1], position[2], 1.0);
                world.z / world.w
            })
            .fold(f32::INFINITY, f32::min);
        if minimum_z.is_finite() {
            self.height -= minimum_z;
        }
    }

    pub fn model_matrix(&self) -> cgmath::Matrix4<f32> {
        let z_scale = match self.kind {
            BasicShape::Plane => self.thickness.max(0.001),
            BasicShape::Sphere | BasicShape::Cube | BasicShape::Torus => self.size,
            BasicShape::Cylinder | BasicShape::Cone => self.shape_height,
        };
        let translation_z = if self.kind == BasicShape::Plane && self.thickness > 0.0 {
            self.height - self.thickness * 0.5
        } else {
            self.height
        };
        cgmath::Matrix4::from_translation(cgmath::Vector3::new(
            self.position_x,
            self.position_y,
            translation_z,
        )) * cgmath::Matrix4::from_angle_z(cgmath::Deg(self.rotation_degrees[2]))
            * cgmath::Matrix4::from_angle_y(cgmath::Deg(self.rotation_degrees[1]))
            * cgmath::Matrix4::from_angle_x(cgmath::Deg(self.rotation_degrees[0]))
            * cgmath::Matrix4::from_nonuniform_scale(
                self.size.max(0.001),
                self.size.max(0.001),
                z_scale.max(0.001),
            )
    }

    pub fn surface_draw<'a>(&'a self, material: &'a wgpu::BindGroup, depth: f32, blend: bool) -> Option<crate::model::SurfaceDraw<'a>> {
        if !self.generated || !self.visible || (self.wireframe && self.hide_surface) { return None; }
        Some(crate::model::SurfaceDraw {
            vertices: &self.vertex_buffer, indices: &self.index_buffer,
            instances: &self.instance_buffer, material,
            index_range: 0..self.index_count, instance_range: 0..1, depth, blend,
        })
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        solid_pipeline: &'a wgpu::RenderPipeline,
        wireframe_pipeline: Option<&'a wgpu::RenderPipeline>,
        quad_line_pipeline: &'a wgpu::RenderPipeline,
        camera: &'a wgpu::BindGroup,
        environment: &'a wgpu::BindGroup,
        lighting: &'a wgpu::BindGroup,
        material_override: Option<&'a wgpu::BindGroup>,
        draw_surface: bool,
    ) {
        if !self.generated || !self.visible {
            return;
        }
        render_pass.set_bind_group(
            0,
            material_override.unwrap_or(&self.material.bind_group),
            &[],
        );
        render_pass.set_bind_group(1, camera, &[]);
        render_pass.set_bind_group(2, environment, &[]);
        render_pass.set_bind_group(3, lighting, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        if draw_surface && (!self.wireframe || !self.hide_surface) {
            render_pass.set_pipeline(solid_pipeline);
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        if !self.wireframe {
            return;
        }
        render_pass.set_bind_group(0, &self.wire_material.bind_group, &[]);
        if !self.triangulate_subdivision {
            render_pass.set_pipeline(quad_line_pipeline);
            render_pass
                .set_index_buffer(self.quad_edge_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.quad_edge_count, 0, 0..1);
        } else if let Some(wireframe_pipeline) = wireframe_pipeline {
            render_pass.set_pipeline(wireframe_pipeline);
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
    }


}

fn quad_edge_indices(triangle_indices: &[u32]) -> Vec<u32> {
    let mut edges = Vec::new();
    for pair in triangle_indices.chunks(6) {
        if pair.len() < 6 {
            for triangle in pair.chunks_exact(3) {
                edges.extend_from_slice(&[
                    triangle[0],
                    triangle[1],
                    triangle[1],
                    triangle[2],
                    triangle[2],
                    triangle[0],
                ]);
            }
            continue;
        }
        let triangle_edges = [
            normalized_edge(pair[0], pair[1]),
            normalized_edge(pair[1], pair[2]),
            normalized_edge(pair[2], pair[0]),
            normalized_edge(pair[3], pair[4]),
            normalized_edge(pair[4], pair[5]),
            normalized_edge(pair[5], pair[3]),
        ];
        for edge in triangle_edges {
            if triangle_edges
                .iter()
                .filter(|candidate| **candidate == edge)
                .count()
                == 1
            {
                edges.extend_from_slice(&[edge.0, edge.1]);
            }
        }
    }
    edges
}

fn normalized_edge(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn ground_instance(kind: BasicShape, size: f32, shape_height: f32, height: f32) -> InstanceRaw {
    let z_scale = match kind {
        BasicShape::Plane | BasicShape::Sphere | BasicShape::Cube | BasicShape::Torus => size,
        BasicShape::Cylinder | BasicShape::Cone => shape_height,
    };
    let model = cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, height))
        * cgmath::Matrix4::from_nonuniform_scale(
            size.max(0.001),
            size.max(0.001),
            z_scale.max(0.001),
        );
    InstanceRaw {
        model: model.into(),
    }
}

fn primitive_mesh(kind: BasicShape, subdivisions: u32) -> (Vec<model::ModelVertex>, Vec<u32>) {
    let subdivisions = subdivisions.clamp(1, 8);
    match kind {
        BasicShape::Plane => cube_mesh(subdivisions),
        BasicShape::Cube => cube_mesh(subdivisions),
        BasicShape::Sphere => sphere_mesh(8 * subdivisions, 4 * subdivisions),
        BasicShape::Cylinder => radial_mesh(8 * subdivisions, false),
        BasicShape::Cone => radial_mesh(8 * subdivisions, true),
        BasicShape::Torus => torus_mesh(8 * subdivisions, 4 * subdivisions),
    }
}

fn cube_mesh(subdivisions: u32) -> (Vec<model::ModelVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (normal, axis_u, axis_v) in [
        ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
        ([0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]),
        ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]),
        ([-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]),
        ([0., 1., 0.], [0., 0., 1.], [1., 0., 0.]),
        ([0., -1., 0.], [0., 0., -1.], [1., 0., 0.]),
    ] {
        let base = vertices.len() as u32;
        for y in 0..=subdivisions {
            for x in 0..=subdivisions {
                let u = x as f32 / subdivisions as f32;
                let v = y as f32 / subdivisions as f32;
                let position = [
                    normal[0] * 0.5 + axis_u[0] * (u - 0.5) + axis_v[0] * (v - 0.5),
                    normal[1] * 0.5 + axis_u[1] * (u - 0.5) + axis_v[1] * (v - 0.5),
                    normal[2] * 0.5 + axis_u[2] * (u - 0.5) + axis_v[2] * (v - 0.5),
                ];
                vertices.push(model::ModelVertex {
                    position,
                    tex_coords: [u, v],
                    normal,
                    tangent: [axis_u[0], axis_u[1], axis_u[2], 1.0],
                });
            }
        }
        let stride = subdivisions + 1;
        for y in 0..subdivisions {
            for x in 0..subdivisions {
                let a = base + y * stride + x;
                let b = a + stride;
                indices.extend_from_slice(&[a, a + 1, b + 1, a, b + 1, b]);
            }
        }
    }
    (vertices, indices)
}

fn sphere_mesh(segments: u32, rings: u32) -> (Vec<model::ModelVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let normal = [phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()];
            vertices.push(model::ModelVertex {
                position: [normal[0] * 0.5, normal[1] * 0.5, normal[2] * 0.5],
                tex_coords: [u, v],
                normal,
                tangent: [-theta.sin(), theta.cos(), 0., 1.],
            });
        }
    }
    let mut indices = Vec::new();
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * (segments + 1) + segment;
            let b = a + segments + 1;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (vertices, indices)
}

fn radial_mesh(segments: u32, cone: bool) -> (Vec<model::ModelVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for segment in 0..=segments {
        let u = segment as f32 / segments as f32;
        let a = u * std::f32::consts::TAU;
        let (c, s) = (a.cos(), a.sin());
        let side_normal = if cone {
            let n = cgmath::Vector3::new(c, s, 0.5).normalize();
            [n.x, n.y, n.z]
        } else {
            [c, s, 0.]
        };
        vertices.push(model::ModelVertex {
            position: [c * 0.5, s * 0.5, -0.5],
            tex_coords: [u, 0.],
            normal: side_normal,
            tangent: [-s, c, 0., 1.],
        });
        vertices.push(model::ModelVertex {
            position: if cone {
                [0., 0., 0.5]
            } else {
                [c * 0.5, s * 0.5, 0.5]
            },
            tex_coords: [u, 1.],
            normal: side_normal,
            tangent: [-s, c, 0., 1.],
        });
    }
    for segment in 0..segments {
        let a = segment * 2;
        indices.extend_from_slice(&[a, a + 2, a + 1, a + 1, a + 2, a + 3]);
    }
    let bottom = vertices.len() as u32;
    vertices.push(model::ModelVertex {
        position: [0., 0., -0.5],
        tex_coords: [0.5, 0.5],
        normal: [0., 0., -1.],
        tangent: [1., 0., 0., 1.],
    });
    for segment in 0..segments {
        let a = segment * 2;
        let next = (segment + 1) * 2;
        indices.extend_from_slice(&[bottom, next, a]);
    }
    if !cone {
        let top = vertices.len() as u32;
        vertices.push(model::ModelVertex {
            position: [0., 0., 0.5],
            tex_coords: [0.5, 0.5],
            normal: [0., 0., 1.],
            tangent: [1., 0., 0., 1.],
        });
        for segment in 0..segments {
            let a = segment * 2 + 1;
            let next = (segment + 1) * 2 + 1;
            indices.extend_from_slice(&[top, a, next]);
        }
    }
    (vertices, indices)
}

fn torus_mesh(major_segments: u32, minor_segments: u32) -> (Vec<model::ModelVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for major in 0..=major_segments {
        let u = major as f32 / major_segments as f32;
        let a = u * std::f32::consts::TAU;
        for minor in 0..=minor_segments {
            let v = minor as f32 / minor_segments as f32;
            let b = v * std::f32::consts::TAU;
            let r = 0.35 + 0.15 * b.cos();
            let normal = [a.cos() * b.cos(), a.sin() * b.cos(), b.sin()];
            vertices.push(model::ModelVertex {
                position: [r * a.cos(), r * a.sin(), 0.15 * b.sin()],
                tex_coords: [u, v],
                normal,
                tangent: [-a.sin(), a.cos(), 0., 1.],
            });
        }
    }
    let mut indices = Vec::new();
    for major in 0..major_segments {
        for minor in 0..minor_segments {
            let a = major * (minor_segments + 1) + minor;
            let b = a + minor_segments + 1;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_basic_shape_has_valid_finite_triangle_geometry() {
        for kind in BasicShape::ALL {
            let (vertices, indices) = primitive_mesh(kind, 1);
            assert!(!vertices.is_empty(), "{} has no vertices", kind.name());
            assert!(!indices.is_empty(), "{} has no indices", kind.name());
            assert_eq!(indices.len() % 3, 0, "{} is not triangles", kind.name());
            assert!(indices.iter().all(|index| *index < vertices.len() as u32));
            assert!(vertices.iter().all(|vertex| {
                vertex
                    .position
                    .iter()
                    .chain(vertex.normal.iter())
                    .chain(vertex.tangent.iter())
                    .all(|value| value.is_finite())
            }));
        }
    }
}
