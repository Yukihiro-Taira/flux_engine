use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridVertex {
    position: [f32; 3],
    color: [f32; 4],
}

impl GridVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x4
        ];

        fn desc() -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GridVertex>()
                    as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &Self::ATTRIBUTES,
            }
        }
}

pub struct Grid {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

impl Grid {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
        size: f32,
        spacing: f32,
        color: [f32; 4],
    ) -> Self {
        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Grid Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("grid.wgsl").into(),
                ),
            });
            
        let pipeline_layout =
            device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("Grid Pipline Layout"),
                    bind_group_layouts: &[Some(camera_layout)],
                    immediate_size: 0,
                },
            );
        let pipeline =
            device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("Grid Pipeline"),
                    layout: Some(&pipeline_layout),

                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[GridVertex::desc()],
                        compilation_options:
                            wgpu::PipelineCompilationOptions::default(),
                    },

                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(
                            wgpu::ColorTargetState {
                                format: surface_format,
                                blend: Some(
                                    wgpu::BlendState::ALPHA_BLENDING,
                                ),
                                write_mask: wgpu::ColorWrites::ALL,
                            },
                        )],
                        compilation_options:
                            wgpu::PipelineCompilationOptions::default(),
                    }),

                    primitive: wgpu::PrimitiveState {
                        topology:
                            wgpu::PrimitiveTopology::LineList,
                            ..Default::default()
                    },

                    depth_stencil: Some(
                        wgpu::DepthStencilState {
                            format:
                                crate::texture::Texture::DEPTH_FORMAT,
                            
                            // The grid should not block models
                            // rendered after it.
                            depth_write_enabled: Some(false),

                            depth_compare: Some(
                                wgpu::CompareFunction::LessEqual,
                            ),

                            stencil:
                                wgpu::StencilState::default(),

                            bias:
                                wgpu::DepthBiasState::default(),
                        },
                    ),

                    multisample:
                        wgpu::MultisampleState::default(),

                    multiview_mask: None,
                    cache: None,
                },
            );
        
        let (vertex_buffer, vertex_count) =
            Self::create_vertices(
                device,
                size,
                spacing,
                color,
            );
        
        Self {
            pipeline,
            vertex_buffer,
            vertex_count,
        }
    }

    pub fn rebuild(
        &mut self,
        device: &wgpu::Device,
        size: f32,
        spacing: f32,
        color: [f32; 4],
        ) {
        let (vertex_buffer, vertex_count) =
            Self::create_vertices(
                device,
                size,
                spacing,
                color,
            );
        self.vertex_buffer = vertex_buffer;
        self.vertex_count = vertex_count;
    }

    fn create_vertices(
        device: &wgpu::Device,
        size: f32,
        spacing: f32,
        color: [f32; 4],
    ) -> (wgpu::Buffer, u32) {
        let size = size.max(1.0);
        let spacing = spacing.max(0.01);
        let line_count = (size / spacing).floor() as i32;

        let mut vertices = Vec::new();

        for line in -line_count..=line_count {
            let offset = line as f32 * spacing;

            // Line parallel to the X axis.
            vertices.push(GridVertex {
                position: [-size, offset, 0.0],
                color,
            });

            vertices.push(GridVertex {
                position: [size, offset, 0.0],
                color,
            });

            // Line parallel to the Z axis
            vertices.push(GridVertex {
                position: [offset, -size, 0.0],
                color,
            });

            vertices.push(GridVertex {
                position: [offset, size, 0.0],
                color,
            });
        }

        let vertex_count = vertices.len() as u32;

        let vertex_buffer =
            device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Grid Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            );
        
        (vertex_buffer, vertex_count)
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);

        render_pass.set_bind_group(
            0,
            camera_bind_group,
            &[],
        );

        render_pass.set_vertex_buffer(
            0,
            self.vertex_buffer.slice(..),
        );

        render_pass.draw(
            0..self.vertex_count,
            0..1,
        );
    }
}