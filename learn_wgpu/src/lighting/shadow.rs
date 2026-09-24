use std::collections::HashMap;

use cgmath::{Deg, InnerSpace, Matrix4, Point3, SquareMatrix, Vector3, ortho, perspective};
use wgpu::util::DeviceExt;

use crate::{Camera, InstanceRaw, model, model::Vertex};

use super::{LightId, LightKind, LightingManager};

pub const SHADOW_SIZE: u32 = 1024;
pub const MAX_SHADOW_LAYERS: u32 = 64;

const OPENGL_TO_WGPU: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowIndex {
    index: [u32; 4],
    padding: [u32; 60],
}

pub struct ShadowRenderer {
    pub texture_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub matrix_buffer: wgpu::Buffer,
    _texture: wgpu::Texture,
    layer_views: Vec<wgpu::TextureView>,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    assignments: HashMap<LightId, (u32, u32)>,
    pass_count: u32,
}

impl ShadowRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow depth array"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: MAX_SHADOW_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow depth array view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer_views = (0..MAX_SHADOW_LAYERS)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Shadow depth layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow comparison sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let identity: [[f32; 4]; 4] = Matrix4::<f32>::identity().into();
        let matrices = vec![identity; MAX_SHADOW_LAYERS as usize];
        let matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shadow matrices"),
            contents: bytemuck::cast_slice(&matrices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let indices = (0..MAX_SHADOW_LAYERS)
            .map(|index| ShadowIndex {
                index: [index, 0, 0, 0],
                padding: [0; 60],
            })
            .collect::<Vec<_>>();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shadow pass indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow pass layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(256),
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow pass bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &index_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(256),
                    }),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow depth shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow depth pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            texture_view,
            sampler,
            matrix_buffer,
            _texture: texture,
            layer_views,
            pipeline,
            bind_group,
            assignments: HashMap::new(),
            pass_count: 0,
        }
    }

    pub fn prepare(&mut self, queue: &wgpu::Queue, manager: &LightingManager, camera: &Camera) {
        self.assignments.clear();
        let mut matrices = Vec::<[[f32; 4]; 4]>::new();
        for light in manager
            .active_direct_lights()
            .filter(|light| light.shadows_enabled)
        {
            let needed = if matches!(light.kind, LightKind::Point) {
                6
            } else {
                1
            };
            if matrices.len() + needed > MAX_SHADOW_LAYERS as usize {
                break;
            }
            let base = matrices.len() as u32;
            let position = Point3::new(light.position[0], light.position[1], light.position[2]);
            if matches!(light.kind, LightKind::Point) {
                for (direction, up) in [
                    (Vector3::unit_x(), Vector3::unit_z()),
                    (-Vector3::unit_x(), Vector3::unit_z()),
                    (Vector3::unit_y(), Vector3::unit_z()),
                    (-Vector3::unit_y(), Vector3::unit_z()),
                    (Vector3::unit_z(), Vector3::unit_y()),
                    (-Vector3::unit_z(), Vector3::unit_y()),
                ] {
                    matrices.push(
                        (OPENGL_TO_WGPU
                            * perspective(Deg(90.0), 1.0, 0.02, 1000.0)
                            * Matrix4::look_at_rh(position, position + direction, up))
                        .into(),
                    );
                }
            } else {
                let direction = light_direction(light.rotation_degrees);
                let matrix = if matches!(light.kind, LightKind::Directional) {
                    let center = Point3::new(camera.target.x, camera.target.y, camera.target.z);
                    OPENGL_TO_WGPU
                        * ortho(-50.0, 50.0, -50.0, 50.0, 0.1, 500.0)
                        * Matrix4::look_at_rh(
                            center - direction * 100.0,
                            center,
                            safe_up(direction),
                        )
                } else {
                    let angle = if matches!(light.kind, LightKind::Spot) {
                        (light.outer_angle_degrees * 2.0).clamp(1.0, 175.0)
                    } else {
                        120.0
                    };
                    OPENGL_TO_WGPU
                        * perspective(Deg(angle), 1.0, 0.02, 1000.0)
                        * Matrix4::look_at_rh(position, position + direction, safe_up(direction))
                };
                matrices.push(matrix.into());
            }
            self.assignments.insert(light.id, (base, needed as u32));
        }
        self.pass_count = matrices.len() as u32;
        if !matrices.is_empty() {
            queue.write_buffer(&self.matrix_buffer, 0, bytemuck::cast_slice(&matrices));
        }
    }

    pub fn assignment(&self, id: LightId) -> Option<(u32, u32)> {
        self.assignments.get(&id).copied()
    }

    pub fn render<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        model: &'a model::Model,
        group_enabled: &[bool],
        instance_buffer: &'a wgpu::Buffer,
        instance_count: u32,
        generated: impl Iterator<Item = &'a crate::ground_plane::GroundPlane>,
    ) {
        let generated = generated.collect::<Vec<_>>();
        for layer in 0..self.pass_count {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow depth pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.layer_views[layer as usize],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[layer * 256]);
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            for (index, mesh) in model.meshes.iter().enumerate() {
                if !group_enabled.get(index).copied().unwrap_or(true) {
                    continue;
                }
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.num_elements, 0, 0..instance_count);
            }
            for object in &generated {
                object.draw_shadow(&mut pass);
            }
        }
    }
}

fn light_direction(rotation: [f32; 3]) -> Vector3<f32> {
    let rotation = cgmath::Matrix3::from_angle_z(Deg(rotation[2]))
        * cgmath::Matrix3::from_angle_y(Deg(rotation[1]))
        * cgmath::Matrix3::from_angle_x(Deg(rotation[0]));
    (rotation * Vector3::new(0.0, 0.0, -1.0)).normalize()
}

fn safe_up(direction: Vector3<f32>) -> Vector3<f32> {
    if direction.z.abs() > 0.98 {
        Vector3::unit_y()
    } else {
        Vector3::unit_z()
    }
}
