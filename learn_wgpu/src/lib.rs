use crate::texture::Texture;
use cgmath::prelude::*;
use model::Vertex;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc, time::Instant};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[cfg(not(target_arch = "wasm32"))]
use winit::window::Fullscreen;

use houdini_navigation::{HoudiniNavigation, NavigationAction};
use navigation_gizmo::ViewDirection;
use wgpu::util::DeviceExt;

mod editor_ui;
mod egui_renderer;
mod environment;
mod grid;
mod houdini_navigation;
mod lighting;
mod material;
mod material_graph;
mod material_preview;
mod model;
mod navigation_gizmo;
mod resources;
mod texture;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum CompareFunction {
    Undefined = 0,
    Never = 1,
    Less = 2,
    Equal = 3,
    LessEqual = 4,
    Greater = 5,
    NotEqual = 6,
    GreaterEqual = 7,
    Always = 8,
}

//----------------------------------------//

//CAMERA

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionMode {
    Perspective,
    Orthographic,
}

#[derive(Clone)]
struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    projection_mode: ProjectionMode,
    ortho_scale: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let projection = match self.projection_mode {
            ProjectionMode::Perspective => {
                cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar)
            }

            ProjectionMode::Orthographic => {
                let half_height = self.ortho_scale * 0.5;

                let half_width = half_height * self.aspect;

                cgmath::ortho(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    self.znear,
                    self.zfar,
                )
            }
        };
        OPENGL_TO_WGPU_MATRIX * projection * view
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
);

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    position: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PointUniform {
    color: [f32; 4],
    // size in pixels, viewport width, viewport height, unused
    settings: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NormalMarkerUniform {
    color: [f32; 4],
    // world-space length, base width in pixels, viewport width, viewport height
    settings: [f32; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
            position: [0.0; 4],
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
        self.position = [camera.eye.x, camera.eye.y, camera.eye.z, 1.0];
    }
}
//----------------------------------------//
//Instace Buffer

#[derive(Clone)]
struct Instance {
    position: cgmath::Vector3<f32>,
    rotation_degrees: cgmath::Vector3<f32>,
    scale: cgmath::Vector3<f32>,
    global_scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let translation = cgmath::Matrix4::from_translation(self.position);
        let rx = cgmath::Matrix4::from_angle_x(cgmath::Deg(self.rotation_degrees.x));
        let ry = cgmath::Matrix4::from_angle_y(cgmath::Deg(self.rotation_degrees.y));
        let rz = cgmath::Matrix4::from_angle_z(cgmath::Deg(self.rotation_degrees.z));
        let scale = cgmath::Matrix4::from_nonuniform_scale(
            self.scale.x * self.global_scale,
            self.scale.y * self.global_scale,
            self.scale.z * self.global_scale,
        );

        InstanceRaw {
            model: (translation * rz * ry * rx * scale).into(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelViewMode {
    Solid,
    Wireframe,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NormalDisplayMode {
    Vertex,
    Point,
    Face,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UvInspectorMode {
    Wireframe,
    UvMap,
}

struct UvSpaceTexture {
    preview: egui::TextureHandle,
    dimensions: [u32; 2],
    _viewport_texture: texture::Texture,
}

#[derive(Clone, Copy)]
enum GroupTextureKind {
    BaseColor,
    Normal,
    Roughness,
}

#[cfg(not(target_arch = "wasm32"))]
struct DecodedUvTexture {
    rgba: image::RgbaImage,
    preview: Option<image::RgbaImage>,
}

#[cfg(not(target_arch = "wasm32"))]
struct PendingUvTexture {
    tile: (i32, i32),
    path: std::path::PathBuf,
    kind: GroupTextureKind,
    receiver: std::sync::mpsc::Receiver<Result<DecodedUvTexture, String>>,
}

//----------------------------------------//
//store state of game
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,
    render_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    point_uniform_buffer: wgpu::Buffer,
    point_bind_group: wgpu::BindGroup,
    show_points: bool,
    point_size: f32,
    point_color: [f32; 4],
    show_normals: bool,
    normal_mode: NormalDisplayMode,
    normal_length: f32,
    normal_color: [f32; 4],
    normal_pipeline: wgpu::RenderPipeline,
    normal_uniform_buffer: wgpu::Buffer,
    normal_bind_group: wgpu::BindGroup,
    show_uv_map: bool,
    show_uv_overlay: bool,
    selected_uv_space: usize,
    show_all_uv_spaces: bool,
    uv_space_offsets: BTreeMap<(i32, i32), egui::Vec2>,
    uv_inspector_mode: UvInspectorMode,
    uv_space_textures: BTreeMap<(i32, i32), UvSpaceTexture>,
    geo_group_enabled: Vec<bool>,
    geo_texture_enabled: Vec<bool>,
    geo_texture_paths: BTreeMap<(i32, i32), String>,
    geo_normal_paths: BTreeMap<(i32, i32), String>,
    geo_roughness_paths: BTreeMap<(i32, i32), String>,
    geo_normal_textures: BTreeMap<(i32, i32), texture::Texture>,
    geo_roughness_textures: BTreeMap<(i32, i32), texture::Texture>,
    geo_material_bind_groups: BTreeMap<(i32, i32), wgpu::BindGroup>,
    #[cfg(not(target_arch = "wasm32"))]
    pending_uv_textures: Vec<PendingUvTexture>,
    #[cfg(not(target_arch = "wasm32"))]
    pending_model_material_root: Option<std::path::PathBuf>,
    show_uv_texture: bool,
    show_uv_lines: bool,
    viewport_settings_open: bool,
    viewport_settings_just_opened: bool,
    wireframe_pipeline: Option<wgpu::RenderPipeline>,
    model_view_mode: ModelViewMode,
    pbr_material: material::PbrMaterial,
    material_graph: material_graph::MaterialGraphEditor,
    material_preview: material_preview::MaterialPreview,
    lighting: lighting::LightingManager,
    lighting_gpu: lighting::LightingGpu,
    active_side_panel: Option<usize>,
    environment_lighting_layout: wgpu::BindGroupLayout,
    environment_lighting_bind_group: wgpu::BindGroup,
    _fallback_environment_texture: wgpu::Texture,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    houdini_navigation: HoudiniNavigation,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    obj_model: model::Model,
    depth_texture: Texture,

    egui: egui_renderer::EguiRenderer,
    editor: editor_ui::EditorUi,
    instance_buffer_dirty: bool,
    background_color: wgpu::Color,
    environment: Option<environment::Environment>,

    grid: grid::Grid,
    displayed_grid_spacing: f32,
    displayed_grid_extent: f32,
    initial_instances: Vec<Instance>,

    current_view: Option<ViewDirection>,
    fps_sample_started: Instant,
    fps_frames: u32,
    displayed_fps: f64,
}

impl State {
    async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();

        const NUM_INSTANCES_PER_ROW: u32 = 10;

        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let position = cgmath::Vector3 {
                        x: x as f32,
                        y: z as f32,
                        z: 0.0,
                    };

                    Instance {
                        position,
                        rotation_degrees: cgmath::Vector3::new(0.0, 0.0, 0.0),
                        scale: cgmath::Vector3::new(0.01, 0.01, 0.01),
                        global_scale: 1.0,
                    }
                })
            })
            .collect::<Vec<_>>();

        //handle GPU
        //Backend::PRIMARY=> Vulkan + Metal +DX12
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let wireframe_supported = adapter
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE);
        let required_features = if wireframe_supported {
            wgpu::Features::POLYGON_MODE_LINE
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        //----------------------------------------//
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let mut egui = egui_renderer::EguiRenderer::new(&window, &device, config.format);
        let editor = editor_ui::EditorUi::default();
        let background_color = wgpu::Color {
            r: editor.background[0] as f64,
            g: editor.background[1] as f64,
            b: editor.background[2] as f64,
            a: editor.background[3] as f64,
        };

        let pbr_material = material::PbrMaterial::new_untextured(&device, &queue)?;
        let environment_lighting_layout = environment::lighting_bind_group_layout(&device);
        let (_fallback_environment_texture, environment_lighting_bind_group) =
            environment::fallback_lighting_bind_group(
                &device,
                &queue,
                &environment_lighting_layout,
            );
        let material_preview = material_preview::MaterialPreview::new(
            &device,
            &mut egui.renderer,
            &pbr_material.layout,
            &environment_lighting_layout,
        );
        let lighting = lighting::LightingManager::default();
        let lighting_gpu = lighting::LightingGpu::new(&device);

        //----------------------------------------//
        //Depth Texture
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        //----------------------------------------//

        let camera = Camera {
            eye: (8.0, -8.0, 6.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_z(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            // Wide editor clipping range for close material inspection and
            // full-scale imported scenes.
            znear: 0.001,
            zfar: 1_000_000.0,

            projection_mode: ProjectionMode::Perspective,
            ortho_scale: 10.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bund_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&pbr_material.layout),
                    Some(&camera_bind_group_layout),
                    Some(&environment_lighting_layout),
                    Some(&lighting_gpu.layout),
                ],
                immediate_size: 0,
            });

        //----------------------------------------//
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let create_model_pipeline = |label, topology, polygon_mode, cull_mode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&render_pipeline_layout),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    polygon_mode,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: texture::Texture::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),

                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },

                multiview_mask: None,
                cache: None,
            })
        };
        let render_pipeline = create_model_pipeline(
            "Solid model pipeline",
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::PolygonMode::Fill,
            Some(wgpu::Face::Back),
        );
        let wireframe_pipeline = wireframe_supported.then(|| {
            create_model_pipeline(
                "Wireframe model pipeline",
                wgpu::PrimitiveTopology::TriangleList,
                wgpu::PolygonMode::Line,
                None,
            )
        });

        let point_uniform = PointUniform {
            color: [0.0, 0.8, 0.72, 1.0],
            settings: [7.0, config.width as f32, config.height as f32, 0.0],
        };
        let point_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point overlay uniform"),
            contents: bytemuck::bytes_of(&point_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let point_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Point overlay layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let point_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Point overlay bind group"),
            layout: &point_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: point_uniform_buffer.as_entire_binding(),
            }],
        });
        let point_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Point overlay pipeline layout"),
                bind_group_layouts: &[Some(&camera_bind_group_layout), Some(&point_layout)],
                immediate_size: 0,
            });
        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point overlay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("point_overlay.wgsl").into()),
        });
        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Point overlay pipeline"),
            layout: Some(&point_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &point_shader,
                entry_point: Some("vs_main"),
                buffers: &[model::PointMarkerVertex::desc(), InstanceRaw::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &point_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let normal_uniform = NormalMarkerUniform {
            color: [0.0, 0.8, 0.72, 1.0],
            settings: [0.01, 3.0, config.width as f32, config.height as f32],
        };
        let normal_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Normal marker uniform"),
            contents: bytemuck::bytes_of(&normal_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let normal_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Normal marker layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let normal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Normal marker bind group"),
            layout: &normal_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: normal_uniform_buffer.as_entire_binding(),
            }],
        });
        let normal_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Normal marker pipeline layout"),
                bind_group_layouts: &[Some(&camera_bind_group_layout), Some(&normal_layout)],
                immediate_size: 0,
            });
        let normal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Normal marker shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("normal_marker.wgsl").into()),
        });
        let normal_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Normal marker pipeline"),
            layout: Some(&normal_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &normal_shader,
                entry_point: Some("vs_main"),
                buffers: &[model::NormalMarkerVertex::desc(), InstanceRaw::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &normal_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let obj_model =
            resources::load_model("t-pose.obj", &device, &queue, &pbr_material.layout).await?;

        //----------------------------------------//

        let grid = grid::Grid::new(
            &device,
            config.format,
            &camera_bind_group_layout,
            editor.grid_size,
            editor.grid_spacing,
            editor.grid_color,
        );

        //----------------------------------------//
        let initial_instances = instances.clone();
        let displayed_grid_spacing = editor.grid_spacing;
        let displayed_grid_extent = editor.grid_size;
        let group_count = obj_model.meshes.len();
        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            window,
            pbr_material,
            point_pipeline,
            point_uniform_buffer,
            point_bind_group,
            show_points: false,
            point_size: 7.0,
            point_color: [0.0, 0.8, 0.72, 1.0],
            show_normals: false,
            normal_mode: NormalDisplayMode::Vertex,
            normal_length: 0.01,
            normal_color: [0.0, 0.8, 0.72, 1.0],
            normal_pipeline,
            normal_uniform_buffer,
            normal_bind_group,
            show_uv_map: false,
            show_uv_overlay: false,
            selected_uv_space: 0,
            show_all_uv_spaces: false,
            uv_space_offsets: BTreeMap::new(),
            uv_inspector_mode: UvInspectorMode::Wireframe,
            uv_space_textures: BTreeMap::new(),
            geo_group_enabled: vec![true; group_count],
            geo_texture_enabled: vec![true; group_count],
            geo_texture_paths: BTreeMap::new(),
            geo_normal_paths: BTreeMap::new(),
            geo_roughness_paths: BTreeMap::new(),
            geo_normal_textures: BTreeMap::new(),
            geo_roughness_textures: BTreeMap::new(),
            geo_material_bind_groups: BTreeMap::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pending_uv_textures: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pending_model_material_root: None,
            show_uv_texture: true,
            show_uv_lines: true,
            viewport_settings_open: false,
            viewport_settings_just_opened: false,
            wireframe_pipeline,
            model_view_mode: ModelViewMode::Solid,
            material_graph: material_graph::MaterialGraphEditor::default(),
            material_preview,
            lighting,
            lighting_gpu,
            active_side_panel: Some(0),
            environment_lighting_layout,
            environment_lighting_bind_group,
            _fallback_environment_texture,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            houdini_navigation: HoudiniNavigation::default(),
            instances,
            instance_buffer,
            obj_model,
            depth_texture,

            egui,
            editor,
            instance_buffer_dirty: false,
            background_color,
            environment: None,
            grid,
            displayed_grid_spacing,
            displayed_grid_extent,
            initial_instances,

            current_view: None,
            fps_sample_started: Instant::now(),
            fps_frames: 0,
            displayed_fps: 0.0,
        })
    }

    //----------------------------------------//
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;

            self.camera.aspect = width as f32 / height as f32;

            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");

            self.is_surface_configured = true;
            self.window.request_redraw();
        }
    }

    fn update(&mut self) {
        self.update_adaptive_grid();
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        // Evaluate the graph before every GPU material upload. This keeps
        // connected values such as Metallic independent of UI/gizmo events.
        self.material_graph
            .apply_to_uniform(&mut self.pbr_material.uniform);
        self.sync_environment_light();
        self.pbr_material.uniform.properties[3] =
            self.editor.hdri_intensity * self.editor.hdri_exposure.exp2();
        self.pbr_material.uniform.options[0] = self.editor.hdri_rotation.to_radians();
        self.pbr_material.uniform.options[2] = if self.editor.texture_enabled {
            1.0
        } else {
            0.0
        };
        self.pbr_material.uniform.inspection[0] = if self.show_uv_overlay { 1.0 } else { 0.0 };
        self.pbr_material.upload(&self.queue);
        self.lighting_gpu
            .upload(&self.queue, &self.lighting, &self.camera);
        if let Some(environment) = &mut self.environment {
            environment.update(
                &self.queue,
                &self.camera,
                self.editor.hdri_intensity,
                self.editor.hdri_exposure,
                self.editor.hdri_rotation,
            );
        }
        if self.instance_buffer_dirty {
            self.upload_instances();
            self.instance_buffer_dirty = false;
        }
    }

    fn sync_environment_light(&mut self) {
        if let Some(light) = self.lighting.lights.iter().find(|light| {
            matches!(
                light.kind,
                lighting::LightKind::Environment | lighting::LightKind::PhysicalSky
            )
        }) {
            self.editor.hdri_intensity = light.intensity;
            self.editor.hdri_exposure = light.exposure;
            self.editor.hdri_rotation = light.rotation_degrees[2];
        }
    }

    fn update_adaptive_grid(&mut self) {
        let distance = (self.camera.eye - self.camera.target)
            .magnitude()
            .max(0.001);
        // Keep the authored 10 cm grid nearby. Each decade of camera distance
        // promotes the visible grid by one decimal unit: 10 cm, 1 m, 10 m…
        let decade = (distance / 10.0).log10().floor().max(0.0);
        let spacing = (self.editor.grid_spacing * 10.0_f32.powf(decade)).min(10_000.0);
        // Grow around both the origin and the current navigation target. This
        // keeps the grid covering the complete view at any zoom or pan and
        // gives it the behavior of an infinite DCC viewport grid.
        let target_radius = (self.camera.target.x * self.camera.target.x
            + self.camera.target.y * self.camera.target.y)
            .sqrt();
        let extent = self
            .editor
            .grid_size
            .max(target_radius + distance * 6.0)
            .max(spacing * 20.0);
        if (spacing - self.displayed_grid_spacing).abs() > f32::EPSILON
            || (extent - self.displayed_grid_extent).abs() > f32::EPSILON
        {
            self.displayed_grid_spacing = spacing;
            self.displayed_grid_extent = extent;
            self.grid.rebuild(
                &self.device,
                self.displayed_grid_extent,
                self.displayed_grid_spacing,
                self.editor.grid_color,
            );
        }
    }

    fn grid_measurement_labels_ui(&self, context: &egui::Context) {
        if !self.editor.show_grid {
            return;
        }
        let major = self.displayed_grid_spacing * 10.0;
        if major <= 0.0 {
            return;
        }
        let extent = self.displayed_grid_extent;
        let count = (extent / major).floor() as i32;
        let view_projection = self.camera.build_view_projection_matrix();
        let screen = context.content_rect();
        let painter = context.layer_painter(egui::LayerId::new(
            // Keep labels below all editor windows and popups.
            egui::Order::Background,
            egui::Id::new("grid_measurement_labels"),
        ));
        let project = |point: cgmath::Vector3<f32>| -> Option<(egui::Pos2, f32)> {
            let clip = view_projection * cgmath::Vector4::new(point.x, point.y, point.z, 1.0);
            if clip.w <= 0.0 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            if ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 || !(0.0..=1.0).contains(&ndc.z) {
                return None;
            }
            Some((
                egui::pos2(
                    screen.left() + (ndc.x + 1.0) * 0.5 * screen.width(),
                    screen.top() + (1.0 - ndc.y) * 0.5 * screen.height(),
                ),
                ndc.z,
            ))
        };

        // Build conservative screen-space depth bounds for visible model
        // instances. A grid label behind one of these bounds is omitted so it
        // cannot bleed through the model as a 2D overlay.
        let mut model_occluders = Vec::new();
        for instance in self
            .instances
            .iter()
            .take(self.editor.visible_instance_count)
        {
            let model: cgmath::Matrix4<f32> = instance.to_raw().model.into();
            let mut bounds = egui::Rect::NOTHING;
            let mut nearest_depth = 1.0_f32;
            let mut valid = false;
            for mesh in &self.obj_model.meshes {
                for x in [mesh.bounds_min[0], mesh.bounds_max[0]] {
                    for y in [mesh.bounds_min[1], mesh.bounds_max[1]] {
                        for z in [mesh.bounds_min[2], mesh.bounds_max[2]] {
                            let world = model * cgmath::Vector4::new(x, y, z, 1.0);
                            if let Some((position, depth)) = project(world.truncate()) {
                                bounds.extend_with(position);
                                nearest_depth = nearest_depth.min(depth);
                                valid = true;
                            }
                        }
                    }
                }
            }
            if valid {
                model_occluders.push((bounds, nearest_depth));
            }
        }
        let hidden_by_model = |position: egui::Pos2, depth: f32| {
            model_occluders
                .iter()
                .any(|(bounds, nearest)| bounds.expand(3.0).contains(position) && depth >= *nearest)
        };
        for index in -count..=count {
            if index == 0 {
                continue;
            }
            let meters = index as f32 * major;
            let text = if meters.fract().abs() < 0.001 {
                format!("{meters:.0} m")
            } else {
                format!("{meters:.1} m")
            };
            if let Some((position, depth)) = project(cgmath::Vector3::new(meters, 0.0, 0.0))
                && !hidden_by_model(position, depth)
            {
                painter.text(
                    position + egui::vec2(3.0, 3.0),
                    egui::Align2::LEFT_TOP,
                    &text,
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(255, 150, 150),
                );
            }
            if let Some((position, depth)) = project(cgmath::Vector3::new(0.0, meters, 0.0))
                && !hidden_by_model(position, depth)
            {
                painter.text(
                    position + egui::vec2(3.0, 3.0),
                    egui::Align2::LEFT_TOP,
                    &text,
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(255, 150, 150),
                );
            }
        }
    }

    fn upload_instances(&self) {
        let data = self
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();

        self.queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&data));
    }

    fn reset_model_transforms(&mut self) {
        self.instances.clone_from(&self.initial_instances);
        self.instance_buffer_dirty = true;
        self.editor.status = "Model transforms restored to defaults".to_owned();
        self.window.request_redraw();
    }

    fn snap_camera_to_view(&mut self, view: ViewDirection) {
        let distance = (self.camera.eye - self.camera.target).magnitude().max(0.01);

        let direction = match view {
            ViewDirection::Right => cgmath::Vector3::new(1.0, 0.0, 0.0),

            ViewDirection::Left => cgmath::Vector3::new(-1.0, 0.0, 0.0),

            ViewDirection::Back => cgmath::Vector3::new(0.0, 1.0, 0.0),

            ViewDirection::Front => cgmath::Vector3::new(0.0, -1.0, 0.0),

            ViewDirection::Top => cgmath::Vector3::new(0.0, 0.0, 1.0),

            ViewDirection::Bottom => cgmath::Vector3::new(0.0, 0.0, -1.0),
        };

        self.camera.eye = self.camera.target + direction * distance;

        self.camera.up = match view {
            ViewDirection::Right
            | ViewDirection::Left
            | ViewDirection::Back
            | ViewDirection::Front => cgmath::Vector3::unit_z(),

            ViewDirection::Top => cgmath::Vector3::unit_y(),

            ViewDirection::Bottom => -cgmath::Vector3::unit_y(),
        };

        self.camera.projection_mode = ProjectionMode::Orthographic;

        self.current_view = Some(view);

        self.houdini_navigation.end_mouse_action();
    }

    fn home_grid_view(&mut self) {
        let distance = self.editor.grid_size.max(1.0) * 1.5;
        self.camera.target = cgmath::Point3::new(0.0, 0.0, 0.0);
        self.camera.eye = cgmath::Point3::new(distance, -distance, distance);
        self.camera.up = cgmath::Vector3::unit_z();
        self.current_view = None;
    }

    fn frame_all_instances(&mut self) {
        let visible_count = self.editor.visible_instance_count.min(self.instances.len());
        let Some(first) = self.instances.first() else {
            self.home_grid_view();
            return;
        };
        if visible_count == 0 {
            self.home_grid_view();
            return;
        }

        let mut minimum = first.position;
        let mut maximum = first.position;
        let mut max_scale =
            first.scale.x.max(first.scale.y).max(first.scale.z) * first.global_scale;
        for instance in self.instances.iter().take(visible_count) {
            minimum.x = minimum.x.min(instance.position.x);
            minimum.y = minimum.y.min(instance.position.y);
            minimum.z = minimum.z.min(instance.position.z);
            maximum.x = maximum.x.max(instance.position.x);
            maximum.y = maximum.y.max(instance.position.y);
            maximum.z = maximum.z.max(instance.position.z);
            max_scale = max_scale.max(
                instance.scale.x.max(instance.scale.y).max(instance.scale.z)
                    * instance.global_scale,
            );
        }

        let center = (minimum + maximum) * 0.5;
        let radius = ((maximum - minimum).magnitude() * 0.5 + max_scale).max(1.0);
        self.frame_point(cgmath::Point3::new(center.x, center.y, center.z), radius);
    }

    fn frame_selected_instance(&mut self) {
        let Some(instance) = self.instances.get(self.editor.selected_instance) else {
            return;
        };
        let center = cgmath::Point3::new(
            instance.position.x,
            instance.position.y,
            instance.position.z,
        );
        let radius =
            instance.scale.x.max(instance.scale.y).max(instance.scale.z) * instance.global_scale;
        let radius = radius.max(0.5);
        self.frame_point(center, radius);
    }

    fn frame_point(&mut self, center: cgmath::Point3<f32>, radius: f32) {
        let view_direction = (self.camera.eye - self.camera.target).normalize();
        self.camera.target = center;
        match self.camera.projection_mode {
            ProjectionMode::Perspective => {
                let half_fov = (self.camera.fovy.to_radians() * 0.5).max(0.01);
                let distance = (radius / half_fov.sin()).max(radius * 2.0);
                self.camera.eye = center + view_direction * distance;
            }
            ProjectionMode::Orthographic => {
                self.camera.ortho_scale = (radius * 2.4).max(0.01);
                let distance = (self.camera.eye - center).magnitude().max(radius * 2.0);
                self.camera.eye = center + view_direction * distance;
            }
        }
        self.current_view = None;
    }

    fn tumble_camera(&mut self, delta_x: f32, delta_y: f32) {
        use cgmath::{Matrix3, Rad};

        let scale = self.houdini_navigation.movement_scale();
        let yaw_amount = -delta_x * self.houdini_navigation.tumble_speed * scale;
        let pitch_amount = -delta_y * self.houdini_navigation.tumble_speed * scale;
        let offset = self.camera.eye - self.camera.target;
        let distance = offset.magnitude();
        if distance <= f32::EPSILON {
            return;
        }

        let world_up = cgmath::Vector3::unit_z();
        let yaw_rotation = Matrix3::from_axis_angle(world_up, Rad(yaw_amount));
        let yawed_offset = yaw_rotation * offset;
        let yawed_up = yaw_rotation * self.camera.up;
        let forward = (-yawed_offset).normalize();
        let right = forward.cross(yawed_up).normalize();
        let pitch_rotation = Matrix3::from_axis_angle(right, Rad(pitch_amount));
        let pitched_offset = pitch_rotation * yawed_offset;
        let pitched_up = (pitch_rotation * yawed_up).normalize();
        let new_forward = (-pitched_offset).normalize();

        if new_forward.dot(world_up).abs() < 0.995 {
            self.camera.eye = self.camera.target + pitched_offset.normalize() * distance;
            self.camera.up = pitched_up;
            self.current_view = None;
        }
    }

    fn track_camera(&mut self, delta_x: f32, delta_y: f32) {
        let forward = (self.camera.target - self.camera.eye).normalize();
        let right = forward.cross(self.camera.up).normalize();
        let up = right.cross(forward).normalize();
        let scale = self.houdini_navigation.movement_scale() * self.houdini_navigation.track_speed;
        let units_per_pixel = match self.camera.projection_mode {
            ProjectionMode::Perspective => {
                let distance = (self.camera.target - self.camera.eye).magnitude();
                2.0 * distance * (self.camera.fovy.to_radians() * 0.5).tan()
                    / self.config.height.max(1) as f32
            }
            ProjectionMode::Orthographic => {
                self.camera.ortho_scale / self.config.height.max(1) as f32
            }
        };
        let movement =
            right * (-delta_x * units_per_pixel * scale) + up * (delta_y * units_per_pixel * scale);
        self.camera.eye += movement;
        self.camera.target += movement;
        self.current_view = None;
    }

    fn dolly_camera(&mut self, delta_y: f32) {
        let scale = self.houdini_navigation.movement_scale();
        let exponent = delta_y * self.houdini_navigation.dolly_speed * scale;
        if self.camera.projection_mode == ProjectionMode::Orthographic {
            self.camera.ortho_scale =
                (self.camera.ortho_scale * exponent.exp()).clamp(0.01, 100_000.0);
            return;
        }

        let offset = self.camera.eye - self.camera.target;
        let current_distance = offset.magnitude();
        if current_distance <= f32::EPSILON {
            return;
        }
        let new_distance = (current_distance * exponent.exp()).clamp(0.01, 100_000.0);
        self.camera.eye = self.camera.target + offset.normalize() * new_distance;
        self.current_view = None;
    }

    fn tilt_camera(&mut self, delta_x: f32) {
        use cgmath::{Matrix3, Rad};

        let forward = (self.camera.target - self.camera.eye).normalize();
        let angle = -delta_x
            * self.houdini_navigation.tumble_speed
            * self.houdini_navigation.movement_scale();
        let rotation = Matrix3::from_axis_angle(forward, Rad(angle));
        self.camera.up = (rotation * self.camera.up).normalize();
        self.current_view = None;
    }

    fn zoom_camera_lens(&mut self, delta_y: f32) {
        let scale = self.houdini_navigation.movement_scale();
        self.camera.fovy = (self.camera.fovy
            + delta_y * self.houdini_navigation.lens_speed * scale)
            .clamp(5.0, 150.0);
        self.current_view = None;
    }

    fn set_tumble_pivot_from_cursor(&mut self) {
        let Some((cursor_x, cursor_y)) = self.houdini_navigation.cursor_position else {
            return;
        };
        let width = self.config.width.max(1) as f32;
        let height = self.config.height.max(1) as f32;
        let ndc_x = cursor_x as f32 / width * 2.0 - 1.0;
        let ndc_y = 1.0 - cursor_y as f32 / height * 2.0;
        let Some(inverse) = self.camera.build_view_projection_matrix().invert() else {
            return;
        };
        let near_clip = inverse * cgmath::Vector4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_clip = inverse * cgmath::Vector4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near_clip.truncate() / near_clip.w;
        let far = far_clip.truncate() / far_clip.w;
        let ray = (far - near).normalize();
        if ray.z.abs() <= f32::EPSILON {
            return;
        }
        let distance_along_ray = -near.z / ray.z;
        if distance_along_ray < 0.0 {
            return;
        }
        let hit = near + ray * distance_along_ray;
        self.camera.target = cgmath::Point3::new(hit.x, hit.y, 0.0);
        self.current_view = None;
    }

    fn current_view_name(&self) -> &'static str {
        match self.current_view {
            Some(ViewDirection::Right) => "Right Orthographic",
            Some(ViewDirection::Left) => "Left Orthographic",
            Some(ViewDirection::Back) => "Back Orthographic",
            Some(ViewDirection::Front) => "Front Orthographic",
            Some(ViewDirection::Top) => "Top Orthographic",
            Some(ViewDirection::Bottom) => "Bottom Orthographic",
            None => match self.camera.projection_mode {
                ProjectionMode::Perspective => "User Perspective",
                ProjectionMode::Orthographic => "User Orthographic",
            },
        }
    }

    fn camera_information_ui(&self, context: &egui::Context) {
        egui::Area::new(egui::Id::new("camera_information"))
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 48.0))
            .show(context, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(160))
                    .corner_radius(5.0)
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.label(self.current_view_name());

                        ui.label(format!(
                            "Eye: {:.2}, {:.2}, {:.2}",
                            self.camera.eye.x, self.camera.eye.y, self.camera.eye.z,
                        ));

                        ui.label(format!(
                            "Target: {:.2}, {:.2}, {:.2}",
                            self.camera.target.x, self.camera.target.y, self.camera.target.z,
                        ));

                        let distance = (self.camera.eye - self.camera.target).magnitude();

                        ui.label(format!("Distance: {:.2}", distance,));
                    });
            });
    }

    fn fps_counter_ui(&self, context: &egui::Context) {
        egui::Area::new(egui::Id::new("fps_counter"))
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
            .order(egui::Order::Middle)
            .interactable(false)
            .show(context, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(160))
                    .corner_radius(5.0)
                    .inner_margin(egui::Margin::symmetric(8, 5))
                    .show(ui, |ui| {
                        ui.monospace(format!("{:07.3} FPS", self.displayed_fps));
                    });
            });
    }

    fn geometry_information_ui(&self, context: &egui::Context) {
        if self.obj_model.meshes.is_empty() {
            return;
        }

        let mut points = 0_usize;
        let mut vertices = 0_usize;
        let mut indices = 0_usize;
        let mut triangles = 0_usize;
        let mut edges = 0_usize;
        let mut boundary_edges = 0_usize;
        let mut non_manifold_edges = 0_usize;
        let mut degenerate_triangles = 0_usize;
        let mut uv_vertices = 0_usize;
        let mut seam_vertices = 0_usize;
        let mut surface_area = 0.0_f64;
        let mut enclosed_volume = Some(0.0_f64);
        let mut all_normals_authored = true;
        let mut bounds_min = [f32::INFINITY; 3];
        let mut bounds_max = [f32::NEG_INFINITY; 3];

        for mesh in &self.obj_model.meshes {
            let stats = &mesh.geometry_stats;
            points += stats.point_count;
            vertices += stats.vertex_count;
            indices += stats.index_count;
            triangles += stats.triangle_count;
            edges += stats.edge_count;
            boundary_edges += stats.boundary_edge_count;
            non_manifold_edges += stats.non_manifold_edge_count;
            degenerate_triangles += stats.degenerate_triangle_count;
            uv_vertices += stats.uv_vertex_count;
            seam_vertices += stats.uv_seam_vertex_count;
            surface_area += stats.surface_area;
            enclosed_volume = enclosed_volume
                .zip(stats.enclosed_volume)
                .map(|(total, volume)| total + volume);
            all_normals_authored &= stats.authored_normals;
            for axis in 0..3 {
                bounds_min[axis] = bounds_min[axis].min(mesh.bounds_min[axis]);
                bounds_max[axis] = bounds_max[axis].max(mesh.bounds_max[axis]);
            }
        }

        let dimensions = [
            bounds_max[0] - bounds_min[0],
            bounds_max[1] - bounds_min[1],
            bounds_max[2] - bounds_min[2],
        ];
        let visible_instances = self.editor.visible_instance_count;
        let vertex_bytes = vertices * std::mem::size_of::<model::ModelVertex>();
        let index_bytes = indices * std::mem::size_of::<u32>();
        let topology = if non_manifold_edges > 0 {
            "Non-manifold"
        } else if boundary_edges > 0 {
            "Open manifold"
        } else {
            "Closed manifold"
        };

        egui::Area::new(egui::Id::new("geometry_information"))
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 168.0))
            .order(egui::Order::Middle)
            .interactable(false)
            .show(context, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(160))
                    .corner_radius(5.0)
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.label("Geometry Information");
                        ui.label(format!("Asset: {}", self.editor.asset_name));
                        ui.label(format!("Meshes: {}", self.obj_model.meshes.len()));
                        ui.label(format!("Points: {points}"));
                        ui.label(format!("Vertices: {vertices}"));
                        ui.label(format!("Lines / edges: {edges}"));
                        ui.label(format!("Triangles: {triangles}"));
                        ui.label(format!("Indices: {indices}"));
                        ui.label(format!("Topology: {topology}"));
                        ui.label(format!("Boundary edges: {boundary_edges}"));
                        ui.label(format!("Non-manifold edges: {non_manifold_edges}"));
                        ui.label(format!("Degenerate triangles: {degenerate_triangles}"));
                        ui.label(format!("UV vertices: {uv_vertices}"));
                        ui.label(format!("Seam duplicates: {seam_vertices}"));
                        ui.label(format!(
                            "Normals: {}",
                            if all_normals_authored {
                                "Authored"
                            } else {
                                "Generated / mixed"
                            }
                        ));
                        ui.label(format!(
                            "Bounds min: {:.3}, {:.3}, {:.3}",
                            bounds_min[0], bounds_min[1], bounds_min[2]
                        ));
                        ui.label(format!(
                            "Bounds max: {:.3}, {:.3}, {:.3}",
                            bounds_max[0], bounds_max[1], bounds_max[2]
                        ));
                        ui.label(format!(
                            "Dimensions: {:.3} × {:.3} × {:.3}",
                            dimensions[0], dimensions[1], dimensions[2]
                        ));
                        ui.label(format!("Surface area: {surface_area:.4}"));
                        ui.label(match enclosed_volume {
                            Some(volume) => format!("Enclosed volume: {volume:.4}"),
                            None => "Enclosed volume: Open mesh".to_owned(),
                        });
                        ui.label(format!("Visible instances: {visible_instances}"));
                        ui.label(format!(
                            "Rendered vertices: {}",
                            vertices * visible_instances
                        ));
                        ui.label(format!(
                            "Rendered triangles: {}",
                            triangles * visible_instances
                        ));
                        ui.label(format!(
                            "Geometry buffers: {:.2} KiB",
                            (vertex_bytes + index_bytes) as f64 / 1024.0
                        ));
                    });
            });
    }

    //--------------------------------------------------------------------//
    // UI STUFF//
    fn build_editor_ui(&mut self, ui: &mut egui::Ui) {
        let context = ui.ctx().clone();
        self.side_panel_tabs(&context);
        let panel_max_height = (context.content_rect().height() - 24.0).max(120.0);

        if self.active_side_panel == Some(0) {
            egui::Window::new("Transform")
                // Version the id when the sizing policy changes so an old forced
                // height cannot override the new content-driven initial layout.
                .id(egui::Id::new("viewport_editor_window_content_sized_v2"))
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
                .default_width(240.0)
                .min_width(240.0)
                .max_height(panel_max_height)
                .resizable(true)
                .collapsible(true)
                .constrain(true)
                .vscroll(true)
                .show(ui, |ui| {
                    if ui.button("Reset Model Transform to default").clicked() {
                        self.reset_model_transforms();
                    }

                    ui.separator();
                    self.asset_drop_ui(ui);
                    ui.separator();
                    self.transform_ui(ui);
                    ui.separator();
                    ui.label(format!("Status: {}", self.editor.status));
                });
        }

        let clicked_view = navigation_gizmo::show(
            &context,
            self.camera.eye,
            self.camera.target,
            self.camera.up,
        );

        if let Some(view) = clicked_view {
            self.snap_camera_to_view(view);
        }

        self.fps_counter_ui(&context);
        self.camera_information_ui(&context);
        self.geometry_information_ui(&context);
        lighting::gizmo::show(&context, &mut self.lighting, &self.camera);
        self.grid_measurement_labels_ui(&context);
        self.viewport_settings_window(&context);
        self.geometry_inspection_window(&context);
        self.lighting_window(&context);
        #[cfg(not(target_arch = "wasm32"))]
        self.apply_pending_model_materials(&context);
        self.textures_window(&context);
        self.geo_tree_window(&context);
        self.uv_map_window(&context);
        self.material_graph.show(
            ui,
            &mut self.pbr_material.uniform,
            self.material_preview.texture_id,
        );
    }

    fn side_panel_tabs(&mut self, context: &egui::Context) {
        const TAB_WIDTH: f32 = 154.0;
        const TAB_HEIGHT: f32 = 40.0;
        const TAB_GAP: f32 = 1.0;

        egui::Area::new(egui::Id::new("right_side_panel_tabs"))
            .anchor(egui::Align2::RIGHT_TOP, egui::Vec2::ZERO)
            .order(egui::Order::Foreground)
            .show(context, |ui| {
                // Tab placement is governed only by the tab dimensions. Expanded
                // window heights never move or re-space the rail.
                ui.spacing_mut().item_spacing.y = TAB_GAP;
                for (index, label) in [
                    "Transform",
                    "Geometry Inspection",
                    "Lighting",
                    "Textures",
                    "Geo Tree",
                ]
                .into_iter()
                .enumerate()
                {
                    let selected = self.active_side_panel == Some(index);
                    if ui
                        .add_sized(
                            [TAB_WIDTH, TAB_HEIGHT],
                            egui::Button::new(label).selected(selected),
                        )
                        .clicked()
                    {
                        self.active_side_panel = if selected { None } else { Some(index) };
                    }
                }
            });
    }

    fn slider_with_number(
        ui: &mut egui::Ui,
        label: &str,
        value: &mut f32,
        range: std::ops::RangeInclusive<f32>,
        speed: f64,
    ) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(label);
            changed |= ui
                .add(egui::Slider::new(value, range.clone()).show_value(false))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(value).speed(speed).range(range))
                .changed();
        });
        changed
    }

    fn asset_drop_ui(&mut self, ui: &mut egui::Ui) {
        let dropped_files = ui.ctx().input(|input| input.raw.dropped_files.clone());

        for file in dropped_files {
            let Some(path) = file.path else {
                continue;
            };

            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            match extension.as_str() {
                "obj" | "fbx" => {
                    self.editor.model_path_input = path.display().to_string();
                    self.editor.asset_name = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("Model")
                        .to_owned();

                    self.editor.pending_asset = Some(path);
                    self.editor.status = "Model queued for loading".to_owned();
                }

                "usd" | "usda" | "usdc" => {
                    self.editor.status = format!("{extension} import is not implemeted");
                }

                _ => {}
            }
        }
        ui.heading("Model Import");
        let path_response = ui.add(
            egui::TextEdit::singleline(&mut self.editor.model_path_input)
                .hint_text("Type or drop an .obj/.fbx file path"),
        );
        let enter_pressed =
            path_response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
        ui.horizontal(|ui| {
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("Browse…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("3D Geometry", &["obj", "fbx"])
                    .pick_file()
                {
                    self.editor.model_path_input = path.display().to_string();
                    self.editor.pending_asset = Some(path);
                    self.editor.status = "Model queued for loading".to_owned();
                }
            }
            if ui.button("Import").clicked() || enter_pressed {
                let path = std::path::PathBuf::from(self.editor.model_path_input.trim());
                if path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        extension.eq_ignore_ascii_case("obj")
                            || extension.eq_ignore_ascii_case("fbx")
                    })
                {
                    self.editor.pending_asset = Some(path);
                    self.editor.status = "Model queued for loading".to_owned();
                } else {
                    self.editor.status =
                        "Model import currently requires an .obj or .fbx file".to_owned();
                }
            }
        });
        ui.label(format!("Current model: {}", self.editor.asset_name));
    }

    fn transform_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Transform");

        if self.instances.is_empty() {
            ui.label("No instances");
            return;
        }

        self.editor.visible_instance_count = self
            .editor
            .visible_instance_count
            .clamp(1, self.instances.len());

        let last_instance = self.editor.visible_instance_count - 1;
        self.editor.selected_instance = self.editor.selected_instance.min(last_instance);

        let item = &mut self.instances[self.editor.selected_instance];

        let mut changed = false;

        // UI--transform interface controls
        ui.label("Translation");
        changed |= Self::slider_with_number(ui, "X", &mut item.position.x, -50.0..=50.0, 0.05);
        changed |= Self::slider_with_number(ui, "Y", &mut item.position.y, -50.0..=50.0, 0.05);
        changed |= Self::slider_with_number(ui, "Z", &mut item.position.z, -50.0..=50.0, 0.05);
        ui.label("Rotation (degrees)");
        changed |=
            Self::slider_with_number(ui, "X", &mut item.rotation_degrees.x, -180.0..=180.0, 0.25);
        changed |=
            Self::slider_with_number(ui, "Y", &mut item.rotation_degrees.y, -180.0..=180.0, 0.25);
        changed |=
            Self::slider_with_number(ui, "Z", &mut item.rotation_degrees.z, -180.0..=180.0, 0.25);
        ui.label("Scale");
        changed |= Self::slider_with_number(
            ui,
            "Global multiplier",
            &mut item.global_scale,
            0.0001..=100.0,
            0.01,
        );
        changed |= Self::slider_with_number(ui, "X", &mut item.scale.x, 0.0001..=10.0, 0.001);
        changed |= Self::slider_with_number(ui, "Y", &mut item.scale.y, 0.0001..=10.0, 0.001);
        changed |= Self::slider_with_number(ui, "Z", &mut item.scale.z, 0.0001..=10.0, 0.001);
        if changed {
            self.instance_buffer_dirty = true;
        }

        ui.separator();
        ui.heading("Instances");
        let amount_response = ui.add(
            egui::Slider::new(
                &mut self.editor.visible_instance_count,
                1..=self.instances.len(),
            )
            .text("Number of Instances")
            .show_value(true),
        );
        if amount_response.changed() {
            self.editor.status = format!("{} of instances", self.editor.visible_instance_count);
        }

        let last_instance = self.editor.visible_instance_count - 1;
        self.editor.selected_instance = self.editor.selected_instance.min(last_instance);
        let instance_response = ui.add(
            egui::Slider::new(&mut self.editor.selected_instance, 0..=last_instance)
                .text("Instance")
                .show_value(true),
        );
        if instance_response.changed() {
            self.editor.status = format!("Selected instance {}", self.editor.selected_instance);
        }
        ui.label(format!(
            "Editing instance {} of {}",
            self.editor.selected_instance + 1,
            self.editor.visible_instance_count,
        ));
    }

    fn texture_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Texture");

        #[cfg(not(target_arch = "wasm32"))]
        {
            ui.label("Material Library (.mtl)");
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.editor.mtl_path_input)
                    .hint_text("Type an .mtl file path"),
            );
            let enter =
                response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            let mut selected_path = None;
            ui.horizontal(|ui| {
                if ui.button("Browse MTL…").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("Wavefront Material Library", &["mtl"])
                        .pick_file()
                {
                    self.editor.mtl_path_input = path.display().to_string();
                    selected_path = Some(path);
                }
                if ui.button("Upload MTL").clicked() || enter {
                    selected_path =
                        Some(std::path::PathBuf::from(self.editor.mtl_path_input.trim()));
                }
            });
            if let Some(path) = selected_path {
                self.import_mtl(ui.ctx(), path);
            }
            ui.separator();
        }

        let graph_button = if self.material_graph.open {
            "Close Material Graph"
        } else {
            "Open Material Graph"
        };
        if ui.button(graph_button).clicked() {
            self.material_graph.toggle_open();
        }

        ui.checkbox(&mut self.editor.texture_enabled, "Texture enabled");

        #[cfg(not(target_arch = "wasm32"))]
        ui.horizontal_wrapped(|ui| {
            if ui.button("Base Color…").clicked() {
                self.editor.pending_texture = pick_material_texture();
            }
            if ui.button("Normal…").clicked() {
                self.editor.pending_normal = pick_material_texture();
            }
            if ui.button("Metal/Rough…").clicked() {
                self.editor.pending_metallic_roughness = pick_material_texture();
            }
        });

        let dropped_files = ui.ctx().input(|input| input.raw.dropped_files.clone());

        for file in dropped_files {
            let Some(path) = file.path else {
                continue;
            };

            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            if matches!(extension.as_str(), "png" | "jpg" | "jpeg") {
                self.editor.texture_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Texture")
                    .to_owned();

                self.editor.pending_texture = Some(path);
                self.editor.status = "Texture queued for loading".to_owned();
            }
        }
    }

    fn viewport_settings_window(&mut self, context: &egui::Context) {
        egui::Area::new(egui::Id::new("viewport_settings_button"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(0.0, -16.0))
            .order(egui::Order::Foreground)
            .show(context, |ui| {
                let cog_color = ui.visuals().text_color().gamma_multiply(0.5);
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new("⚙").size(40.0).color(cog_color))
                            .min_size(egui::vec2(68.0, 68.0))
                            .frame(false),
                    )
                    .on_hover_text("Viewport settings")
                    .clicked()
                {
                    if self.viewport_settings_open {
                        self.viewport_settings_open = false;
                    } else {
                        self.viewport_settings_open = true;
                        self.viewport_settings_just_opened = true;
                    }
                }
            });

        if !self.viewport_settings_open {
            return;
        }
        let mut open = self.viewport_settings_open;
        let mut window = egui::Window::new("Viewport Settings")
            .id(egui::Id::new("viewport_settings_window"))
            .open(&mut open)
            .pivot(egui::Align2::CENTER_CENTER)
            .default_width(280.0)
            .movable(true)
            .resizable(true)
            .constrain(true);
        if self.viewport_settings_just_opened {
            window = window.current_pos(context.content_rect().center());
        }
        window.show(context, |ui| self.viewport_settings_contents(ui));
        self.viewport_settings_open = open;
        self.viewport_settings_just_opened = false;
    }

    fn lighting_window(&mut self, context: &egui::Context) {
        if self.active_side_panel != Some(2) {
            return;
        }
        egui::Window::new("Lighting")
            .id(egui::Id::new("lighting_window"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
            .default_width(360.0)
            .max_height((context.content_rect().height() - 24.0).max(120.0))
            .collapsible(true)
            .constrain(true)
            .vscroll(true)
            .show(context, |ui| {
                self.hdri_ui(ui);
                ui.separator();
                lighting::editor::show(ui, &mut self.lighting);
            });
    }

    fn geometry_inspection_window(&mut self, context: &egui::Context) {
        if self.active_side_panel != Some(1) {
            return;
        }
        egui::Window::new("Geometry Inspection")
            .id(egui::Id::new("geometry_inspection_window"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
            .default_width(280.0)
            .max_height((context.content_rect().height() - 24.0).max(120.0))
            .collapsible(true)
            .constrain(true)
            .vscroll(true)
            .show(context, |ui| {
                ui.heading("Model View");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.model_view_mode, ModelViewMode::Solid, "Solid");
                    let wireframe = ui.add_enabled(
                        self.wireframe_pipeline.is_some(),
                        egui::Button::new("Wireframe")
                            .selected(self.model_view_mode == ModelViewMode::Wireframe),
                    );
                    if wireframe.clicked() {
                        self.model_view_mode = ModelViewMode::Wireframe;
                    }
                    if self.wireframe_pipeline.is_none() {
                        wireframe.on_hover_text("Wireframe is not supported by this GPU backend");
                    }
                    if ui
                        .add(egui::Button::new("UV Overlay").selected(self.show_uv_overlay))
                        .on_hover_text("Overlay a DCC-style UV checker on the shaded model")
                        .clicked()
                    {
                        self.show_uv_overlay = !self.show_uv_overlay;
                        if self.show_uv_overlay {
                            self.model_view_mode = ModelViewMode::Solid;
                        }
                    }
                });
                ui.separator();
                ui.heading("Points");
                ui.checkbox(&mut self.show_points, "Show points over model");
                ui.label("Point size and color are available in Viewport Settings.");
                ui.separator();
                ui.heading("UV Inspection");
                ui.checkbox(&mut self.show_uv_map, "Show UV map");
                ui.separator();
                ui.heading("Normal Markers");
                ui.checkbox(&mut self.show_normals, "Show normals");
                ui.add_enabled_ui(self.show_normals, |ui| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.normal_mode,
                            NormalDisplayMode::Vertex,
                            "Vertex",
                        );
                        ui.selectable_value(
                            &mut self.normal_mode,
                            NormalDisplayMode::Point,
                            "Point",
                        );
                        ui.selectable_value(&mut self.normal_mode, NormalDisplayMode::Face, "Face");
                    });
                    Self::slider_with_number(
                        ui,
                        "Length",
                        &mut self.normal_length,
                        0.001..=10.0,
                        0.01,
                    );
                    ui.horizontal(|ui| {
                        ui.label("Color");
                        ui.color_edit_button_rgba_unmultiplied(&mut self.normal_color);
                    });
                });
            });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn import_uv_texture(
        &mut self,
        context: &egui::Context,
        tile: (i32, i32),
        path: std::path::PathBuf,
    ) {
        self.import_group_texture(context, tile, path, GroupTextureKind::BaseColor);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn import_group_texture(
        &mut self,
        context: &egui::Context,
        tile: (i32, i32),
        path: std::path::PathBuf,
        kind: GroupTextureKind,
    ) {
        let max_preview_side = context.input(|input| input.max_texture_side).max(1) as u32;
        let worker_path = path.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = image::open(&worker_path)
                .map(|image| image.to_rgba8())
                .map(|rgba| {
                    let preview = matches!(kind, GroupTextureKind::BaseColor).then(|| {
                        if rgba.width() <= max_preview_side && rgba.height() <= max_preview_side {
                            rgba.clone()
                        } else {
                            let scale = max_preview_side as f64
                                / f64::from(rgba.width().max(rgba.height()));
                            let width = (f64::from(rgba.width()) * scale).round().max(1.0) as u32;
                            let height = (f64::from(rgba.height()) * scale).round().max(1.0) as u32;
                            image::imageops::resize(
                                &rgba,
                                width,
                                height,
                                image::imageops::FilterType::Triangle,
                            )
                        }
                    });
                    DecodedUvTexture { rgba, preview }
                })
                .map_err(|error| error.to_string());
            let _ = sender.send(result);
        });
        self.pending_uv_textures.push(PendingUvTexture {
            tile,
            path,
            kind,
            receiver,
        });
        self.editor.status = "Loading high-resolution texture…".to_owned();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn finish_pending_uv_texture(&mut self, context: &egui::Context) {
        let completed = self
            .pending_uv_textures
            .iter()
            .enumerate()
            .find_map(|(index, pending)| {
                pending
                    .receiver
                    .try_recv()
                    .ok()
                    .map(|result| (index, result))
            });
        let Some((index, result)) = completed else {
            return;
        };
        let pending = self.pending_uv_textures.remove(index);
        let decoded = match result {
            Ok(decoded) => decoded,
            Err(error) => {
                self.editor.status = format!("Could not load UV texture: {error}");
                return;
            }
        };
        let mut rgba = decoded.rgba;
        if matches!(pending.kind, GroupTextureKind::Roughness) {
            for pixel in rgba.pixels_mut() {
                *pixel = image::Rgba([255, pixel[0], 0, 255]);
            }
        }
        let srgb = matches!(pending.kind, GroupTextureKind::BaseColor);
        let viewport_texture = match texture::Texture::from_rgba8(
            &self.device,
            &self.queue,
            rgba.as_raw(),
            rgba.width(),
            rgba.height(),
            srgb,
            match pending.kind {
                GroupTextureKind::BaseColor => "PBR per-group base color",
                GroupTextureKind::Normal => "PBR per-group normal",
                GroupTextureKind::Roughness => "PBR per-group roughness",
            },
        ) {
            Ok(texture) => texture,
            Err(error) => {
                self.editor.status = format!("Could not upload UV texture: {error:#}");
                return;
            }
        };
        match pending.kind {
            GroupTextureKind::BaseColor => {
                let preview_rgba = decoded.preview.expect("base color preview must exist");
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [
                        preview_rgba.width() as usize,
                        preview_rgba.height() as usize,
                    ],
                    preview_rgba.as_raw(),
                );
                let preview = context.load_texture(
                    format!("uv_inspector_texture_{}_{}", pending.tile.0, pending.tile.1),
                    color_image,
                    egui::TextureOptions::LINEAR,
                );
                self.uv_space_textures.insert(
                    pending.tile,
                    UvSpaceTexture {
                        preview,
                        dimensions: [rgba.width(), rgba.height()],
                        _viewport_texture: viewport_texture,
                    },
                );
                self.show_uv_texture = true;
                self.editor.texture_name = pending
                    .path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Texture")
                    .to_owned();
                self.geo_texture_paths
                    .insert(pending.tile, pending.path.display().to_string());
            }
            GroupTextureKind::Normal => {
                self.geo_normal_textures
                    .insert(pending.tile, viewport_texture);
                self.geo_normal_paths
                    .insert(pending.tile, pending.path.display().to_string());
            }
            GroupTextureKind::Roughness => {
                self.geo_roughness_textures
                    .insert(pending.tile, viewport_texture);
                self.geo_roughness_paths
                    .insert(pending.tile, pending.path.display().to_string());
            }
        }
        self.rebuild_group_material_bind_group(pending.tile);
        self.editor.texture_enabled = true;
        self.pbr_material.uniform.inspection[1] = 0.0;
        self.pbr_material.uniform.udim = [0.0, 0.0, 1.0, 1.0];
        self.pbr_material.uniform.udim_mask = [0; 4];
        let udim = 1001 + pending.tile.0 + pending.tile.1 * 10;
        let kind = match pending.kind {
            GroupTextureKind::BaseColor => "base color",
            GroupTextureKind::Normal => "normal",
            GroupTextureKind::Roughness => "roughness",
        };
        self.editor.status = format!("Applied {kind} texture to UV space {udim}");
    }

    fn rebuild_group_material_bind_group(&mut self, tile: (i32, i32)) {
        let bind_group = self.pbr_material.bind_group_with_overrides(
            &self.device,
            self.uv_space_textures
                .get(&tile)
                .map(|texture| &texture._viewport_texture),
            self.geo_normal_textures.get(&tile),
            self.geo_roughness_textures.get(&tile),
        );
        self.geo_material_bind_groups.insert(tile, bind_group);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn apply_pending_model_materials(&mut self, context: &egui::Context) {
        let Some(root) = self.pending_model_material_root.take() else {
            return;
        };
        let solid_colors = self
            .obj_model
            .meshes
            .iter()
            .filter_map(|mesh| {
                let material = self.obj_model.materials.get(mesh.material)?;
                material
                    .diffuse_texture
                    .is_empty()
                    .then(|| (mesh.uv_tile, material.diffuse, material.name.clone()))
            })
            .collect::<Vec<_>>();
        for (tile, color, name) in solid_colors {
            self.apply_mtl_diffuse_color(context, tile, color, &name);
        }
        let jobs = self
            .obj_model
            .meshes
            .iter()
            .filter_map(|mesh| {
                self.obj_model
                    .materials
                    .get(mesh.material)
                    .map(|material| (mesh.uv_tile, material.clone()))
            })
            .flat_map(|(tile, material)| {
                [
                    (!material.diffuse_texture.is_empty()).then(|| {
                        (
                            tile,
                            root.join(material.diffuse_texture),
                            GroupTextureKind::BaseColor,
                        )
                    }),
                    (!material.normal_texture.is_empty()).then(|| {
                        (
                            tile,
                            root.join(material.normal_texture),
                            GroupTextureKind::Normal,
                        )
                    }),
                    (!material.roughness_texture.is_empty()).then(|| {
                        (
                            tile,
                            root.join(material.roughness_texture),
                            GroupTextureKind::Roughness,
                        )
                    }),
                ]
                .into_iter()
                .flatten()
            })
            .collect::<Vec<_>>();
        for (tile, path, kind) in jobs {
            self.import_group_texture(context, tile, path, kind);
        }
        if !self.obj_model.materials.is_empty() {
            self.editor.status = format!(
                "Loaded {} OBJ material assignment(s)",
                self.obj_model.materials.len()
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn import_mtl(&mut self, context: &egui::Context, path: std::path::PathBuf) {
        let (materials, _) = match tobj::load_mtl(&path) {
            Ok(result) => result,
            Err(error) => {
                self.editor.status = format!("Could not import MTL: {error}");
                return;
            }
        };
        let root = path.parent().unwrap_or(std::path::Path::new("."));
        let mut jobs = Vec::new();
        let mut solid_colors = Vec::new();
        for mesh in &self.obj_model.meshes {
            let Some(material) = materials.get(mesh.material) else {
                continue;
            };
            let roughness = material
                .unknown_param
                .get("map_Pr")
                .or_else(|| material.unknown_param.get("map_roughness"));
            if material.diffuse_texture.is_empty() {
                solid_colors.push((mesh.uv_tile, material.diffuse, material.name.clone()));
            }
            if !material.diffuse_texture.is_empty() {
                jobs.push((
                    mesh.uv_tile,
                    root.join(&material.diffuse_texture),
                    GroupTextureKind::BaseColor,
                ));
            }
            if !material.normal_texture.is_empty() {
                jobs.push((
                    mesh.uv_tile,
                    root.join(&material.normal_texture),
                    GroupTextureKind::Normal,
                ));
            }
            if let Some(texture) = roughness {
                jobs.push((
                    mesh.uv_tile,
                    root.join(texture),
                    GroupTextureKind::Roughness,
                ));
            }
        }
        for (tile, color, name) in solid_colors {
            self.apply_mtl_diffuse_color(context, tile, color, &name);
        }
        for (tile, texture_path, kind) in jobs {
            self.import_group_texture(context, tile, texture_path, kind);
        }
        self.editor.mtl_path_input = path.display().to_string();
        self.editor.status = format!("Imported MTL with {} material(s)", materials.len());
    }

    fn apply_mtl_diffuse_color(
        &mut self,
        context: &egui::Context,
        tile: (i32, i32),
        color: [f32; 3],
        material_name: &str,
    ) {
        let rgba = color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8);
        let pixel = [rgba[0], rgba[1], rgba[2], 255];
        let Ok(viewport_texture) = texture::Texture::from_rgba8(
            &self.device,
            &self.queue,
            &pixel,
            1,
            1,
            true,
            "PBR MTL diffuse color",
        ) else {
            return;
        };
        let preview = context.load_texture(
            format!("mtl_color_{}_{}", tile.0, tile.1),
            egui::ColorImage::from_rgba_unmultiplied([1, 1], &pixel),
            egui::TextureOptions::LINEAR,
        );
        self.uv_space_textures.insert(
            tile,
            UvSpaceTexture {
                preview,
                dimensions: [1, 1],
                _viewport_texture: viewport_texture,
            },
        );
        self.geo_texture_paths
            .insert(tile, format!("MTL: {material_name}"));
        self.rebuild_group_material_bind_group(tile);
        self.editor.texture_enabled = true;
    }

    fn uv_map_window(&mut self, context: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        self.finish_pending_uv_texture(context);
        if !self.show_uv_map {
            return;
        }
        let mut uv_spaces = BTreeMap::<(i32, i32), (String, Vec<([f32; 2], [f32; 2])>)>::new();
        for (group_index, mesh) in self.obj_model.meshes.iter().enumerate() {
            if mesh.geometry_stats.uv_vertex_count == 0 {
                continue;
            }
            let group_name = if mesh.name.trim().is_empty() {
                format!("Group {}", group_index + 1)
            } else {
                mesh.name.clone()
            };
            for triangle in mesh.uv_edges.chunks_exact(3) {
                let center = [
                    (triangle[0].0[0] + triangle[1].0[0] + triangle[2].0[0]) / 3.0,
                    (triangle[0].0[1] + triangle[1].0[1] + triangle[2].0[1]) / 3.0,
                ];
                uv_spaces
                    .entry((center[0].floor() as i32, center[1].floor() as i32))
                    .or_insert_with(|| (group_name.clone(), Vec::new()))
                    .1
                    .extend_from_slice(triangle);
            }
        }
        let uv_spaces = uv_spaces.into_iter().collect::<Vec<_>>();
        self.selected_uv_space = self
            .selected_uv_space
            .min(uv_spaces.len().saturating_sub(1));

        let mut open = self.show_uv_map;
        let mut texture_import_tile = None;
        egui::Window::new("UV Map")
            .id(egui::Id::new("uv_map_window"))
            .open(&mut open)
            .default_pos(egui::pos2(260.0, 80.0))
            .default_width(420.0)
            .min_width(240.0)
            .resizable(true)
            .constrain(true)
            .show(context, |ui| {
                ui.horizontal_wrapped(|ui| {
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some((tile, (group_name, _))) = uv_spaces.get(self.selected_uv_space) {
                        if ui
                            .button(format!(
                                "Import for {} · {} …",
                                1001 + tile.0 + tile.1 * 10,
                                group_name
                            ))
                            .clicked()
                        {
                            texture_import_tile = Some(*tile);
                        }
                    }
                    ui.selectable_value(
                        &mut self.uv_inspector_mode,
                        UvInspectorMode::Wireframe,
                        "Wireframe",
                    );
                    ui.selectable_value(
                        &mut self.uv_inspector_mode,
                        UvInspectorMode::UvMap,
                        "UV Map",
                    );
                });
                ui.horizontal_wrapped(|ui| {
                    ui.add_enabled_ui(!self.uv_space_textures.is_empty(), |ui| {
                        ui.checkbox(&mut self.show_uv_texture, "Show Texture");
                    });
                    ui.checkbox(&mut self.show_uv_lines, "Show UVs");
                    if let Some((tile, _)) = uv_spaces.get(self.selected_uv_space) {
                        if let Some(texture) = self.uv_space_textures.get(tile) {
                            ui.label(format!(
                                "{} × {}",
                                texture.dimensions[0], texture.dimensions[1]
                            ));
                        }
                    }
                });
                ui.horizontal(|ui| {
                    let multiple = uv_spaces.len() > 1;
                    if ui.add_enabled(multiple, egui::Button::new("◀")).clicked() {
                        self.selected_uv_space = if self.selected_uv_space == 0 {
                            uv_spaces.len() - 1
                        } else {
                            self.selected_uv_space - 1
                        };
                        self.show_all_uv_spaces = false;
                    }
                    let space_label = uv_spaces
                        .get(self.selected_uv_space)
                        .map(|((u, v), (group_name, _))| {
                            format!("UDIM {} · {} ({u}, {v})", 1001 + u + v * 10, group_name)
                        })
                        .unwrap_or_else(|| "No UV spaces".to_owned());
                    ui.label(space_label);
                    if ui.add_enabled(multiple, egui::Button::new("▶")).clicked() {
                        self.selected_uv_space = (self.selected_uv_space + 1) % uv_spaces.len();
                        self.show_all_uv_spaces = false;
                    }
                    ui.toggle_value(&mut self.show_all_uv_spaces, "Show All");
                    if self.show_all_uv_spaces && ui.small_button("Reset Layout").clicked() {
                        self.uv_space_offsets.clear();
                    }
                });
                let side = ui.available_width().max(200.0);
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::hover());
                let painter = ui.painter().with_clip_rect(rect);
                painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(18, 20, 22));
                let draw_space = |tile_rect: egui::Rect,
                                  tile: (i32, i32),
                                  group_name: &str,
                                  edges: &[([f32; 2], [f32; 2])]| {
                    painter.rect_filled(tile_rect, 0.0, egui::Color32::from_rgb(18, 20, 22));
                    if self.uv_inspector_mode == UvInspectorMode::UvMap {
                        let checker_size = tile_rect.width() / 10.0;
                        for row in 0..10 {
                            for column in 0..10 {
                                let color = if (row + column) % 2 == 0 {
                                    egui::Color32::from_rgb(38, 42, 46)
                                } else {
                                    egui::Color32::from_rgb(142, 148, 154)
                                };
                                painter.rect_filled(
                                    egui::Rect::from_min_max(
                                        tile_rect.min
                                            + egui::vec2(
                                                column as f32 * checker_size,
                                                row as f32 * checker_size,
                                            ),
                                        tile_rect.min
                                            + egui::vec2(
                                                (column + 1) as f32 * checker_size,
                                                (row + 1) as f32 * checker_size,
                                            ),
                                    ),
                                    0.0,
                                    color,
                                );
                            }
                        }
                    }
                    if self.show_uv_texture {
                        if let Some(texture) = self.uv_space_textures.get(&tile) {
                            painter.image(
                                texture.preview.id(),
                                tile_rect,
                                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                                egui::Color32::WHITE,
                            );
                        }
                    }
                    let grid_color = egui::Color32::from_white_alpha(28);
                    for division in 1..10 {
                        let fraction = division as f32 / 10.0;
                        let x = egui::lerp(tile_rect.left()..=tile_rect.right(), fraction);
                        let y = egui::lerp(tile_rect.top()..=tile_rect.bottom(), fraction);
                        painter.line_segment(
                            [
                                egui::pos2(x, tile_rect.top()),
                                egui::pos2(x, tile_rect.bottom()),
                            ],
                            egui::Stroke::new(1.0, grid_color),
                        );
                        painter.line_segment(
                            [
                                egui::pos2(tile_rect.left(), y),
                                egui::pos2(tile_rect.right(), y),
                            ],
                            egui::Stroke::new(1.0, grid_color),
                        );
                    }
                    let uv_to_screen = |uv: [f32; 2]| {
                        let local_u = uv[0] - tile.0 as f32;
                        let local_v = uv[1] - tile.1 as f32;
                        egui::pos2(
                            egui::lerp(tile_rect.left()..=tile_rect.right(), local_u),
                            egui::lerp(tile_rect.bottom()..=tile_rect.top(), local_v),
                        )
                    };
                    if self.show_uv_lines {
                        let uv_color = egui::Color32::from_rgba_unmultiplied(0, 235, 210, 230);
                        for &(start, end) in edges {
                            painter.line_segment(
                                [uv_to_screen(start), uv_to_screen(end)],
                                egui::Stroke::new(1.0, uv_color),
                            );
                        }
                    }
                    let border = egui::Color32::from_white_alpha(110);
                    painter.rect_stroke(
                        tile_rect,
                        0.0,
                        egui::Stroke::new(1.0, border),
                        egui::StrokeKind::Inside,
                    );
                    painter.text(
                        tile_rect.left_top() + egui::vec2(5.0, 5.0),
                        egui::Align2::LEFT_TOP,
                        format!("{} · {}", 1001 + tile.0 + tile.1 * 10, group_name),
                        egui::FontId::monospace(10.0),
                        egui::Color32::WHITE,
                    );
                };

                if self.show_all_uv_spaces && !uv_spaces.is_empty() {
                    let count = uv_spaces.len();
                    let columns = (count as f32).sqrt().ceil() as usize;
                    let rows = count.div_ceil(columns);
                    let gap = 4.0;
                    let cell_width =
                        (rect.width() - gap * (columns.saturating_sub(1)) as f32) / columns as f32;
                    let cell_height =
                        (rect.height() - gap * (rows.saturating_sub(1)) as f32) / rows as f32;
                    let tile_size = cell_width.min(cell_height);
                    let layout_width =
                        columns as f32 * tile_size + gap * columns.saturating_sub(1) as f32;
                    let layout_height =
                        rows as f32 * tile_size + gap * rows.saturating_sub(1) as f32;
                    let origin = rect.center() - egui::vec2(layout_width, layout_height) * 0.5;
                    for (index, (tile, (group_name, edges))) in uv_spaces.iter().enumerate() {
                        let column = index % columns;
                        let row = index / columns;
                        let base_min = origin
                            + egui::vec2(
                                column as f32 * (tile_size + gap),
                                row as f32 * (tile_size + gap),
                            );
                        let offset = self.uv_space_offsets.entry(*tile).or_default();
                        let tile_rect = egui::Rect::from_min_size(
                            base_min + *offset,
                            egui::vec2(tile_size, tile_size),
                        );
                        let response = ui.interact(
                            tile_rect,
                            egui::Id::new(("uv_space_tile", tile.0, tile.1)),
                            egui::Sense::drag(),
                        );
                        if response.dragged() {
                            *offset += ui.ctx().input(|input| input.pointer.delta());
                        }
                        if response.double_clicked() {
                            self.selected_uv_space = index;
                            self.show_all_uv_spaces = false;
                        }
                        draw_space(tile_rect, *tile, group_name, edges);
                        if tile_size >= 72.0 {
                            let button_rect = egui::Rect::from_center_size(
                                egui::pos2(tile_rect.center().x, tile_rect.bottom() - 14.0),
                                egui::vec2((tile_size - 8.0).min(96.0), 22.0),
                            );
                            if ui
                                .put(button_rect, egui::Button::new("Import…").small())
                                .clicked()
                            {
                                texture_import_tile = Some(*tile);
                            }
                        }
                    }
                } else if let Some((tile, (group_name, edges))) =
                    uv_spaces.get(self.selected_uv_space)
                {
                    draw_space(rect, *tile, group_name, edges);
                } else {
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "No authored UV coordinates",
                        egui::FontId::proportional(14.0),
                        egui::Color32::GRAY,
                    );
                }
            });
        self.show_uv_map = open;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(tile) = texture_import_tile {
            if let Some(path) = pick_material_texture() {
                self.import_uv_texture(context, tile, path);
            }
        }
    }

    fn textures_window(&mut self, context: &egui::Context) {
        if self.active_side_panel != Some(3) {
            return;
        }
        egui::Window::new("Textures")
            .id(egui::Id::new("textures_window"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
            .default_width(340.0)
            .max_height((context.content_rect().height() - 24.0).max(120.0))
            .collapsible(true)
            .constrain(true)
            .vscroll(true)
            .resizable(true)
            .show(context, |ui| self.texture_ui(ui));
    }

    fn geo_tree_window(&mut self, context: &egui::Context) {
        if self.active_side_panel != Some(4) {
            return;
        }
        let group_count = self.obj_model.meshes.len();
        self.geo_group_enabled.resize(group_count, true);
        self.geo_texture_enabled.resize(group_count, true);
        let mut texture_import = None;
        let mut texture_reset = None;

        egui::Window::new("Geo Tree")
            .id(egui::Id::new("geo_tree_window"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
            .default_width(420.0)
            .max_height((context.content_rect().height() - 24.0).max(120.0))
            .collapsible(true)
            .constrain(true)
            .vscroll(true)
            .resizable(true)
            .show(context, |ui| {
                ui.heading("Object Groups");
                ui.label("Geometry and textures are linked to the viewport and UV Inspector.");
                for group_index in 0..group_count {
                    let tile = self.obj_model.meshes[group_index].uv_tile;
                    let group_name = self.obj_model.meshes[group_index].name.clone();
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.geo_group_enabled[group_index], "");
                            ui.strong(if group_name.trim().is_empty() {
                                format!("Group {}", group_index + 1)
                            } else {
                                group_name
                            });
                            ui.label(format!("UDIM {}", 1001 + tile.0 + tile.1 * 10));
                        });

                        ui.checkbox(
                            &mut self.geo_texture_enabled[group_index],
                            "Textures enabled",
                        );
                        ui.horizontal(|ui| {
                            ui.label("Base Color");
                            let texture_path = self.geo_texture_paths.entry(tile).or_default();
                            let response = ui.add(
                                egui::TextEdit::singleline(texture_path).hint_text("Texture path"),
                            );
                            let enter = response.lost_focus()
                                && ui.input(|input| input.key_pressed(egui::Key::Enter));
                            if enter {
                                texture_import = Some((
                                    tile,
                                    std::path::PathBuf::from(texture_path.trim()),
                                    GroupTextureKind::BaseColor,
                                ));
                            }
                            #[cfg(not(target_arch = "wasm32"))]
                            if ui.button("Browse…").clicked()
                                && let Some(path) = pick_material_texture()
                            {
                                *texture_path = path.display().to_string();
                                texture_import = Some((tile, path, GroupTextureKind::BaseColor));
                            }
                            if ui.small_button("Reset").clicked() {
                                texture_reset = Some((tile, GroupTextureKind::BaseColor));
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Normal");
                            let path_text = self.geo_normal_paths.entry(tile).or_default();
                            let response = ui.add(
                                egui::TextEdit::singleline(path_text).hint_text("Normal map path"),
                            );
                            if response.lost_focus()
                                && ui.input(|input| input.key_pressed(egui::Key::Enter))
                            {
                                texture_import = Some((
                                    tile,
                                    std::path::PathBuf::from(path_text.trim()),
                                    GroupTextureKind::Normal,
                                ));
                            }
                            #[cfg(not(target_arch = "wasm32"))]
                            if ui.button("Browse…").clicked()
                                && let Some(path) = pick_material_texture()
                            {
                                *path_text = path.display().to_string();
                                texture_import = Some((tile, path, GroupTextureKind::Normal));
                            }
                            if ui.small_button("Reset").clicked() {
                                texture_reset = Some((tile, GroupTextureKind::Normal));
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Roughness");
                            let path_text = self.geo_roughness_paths.entry(tile).or_default();
                            let response = ui.add(
                                egui::TextEdit::singleline(path_text)
                                    .hint_text("Roughness map path"),
                            );
                            if response.lost_focus()
                                && ui.input(|input| input.key_pressed(egui::Key::Enter))
                            {
                                texture_import = Some((
                                    tile,
                                    std::path::PathBuf::from(path_text.trim()),
                                    GroupTextureKind::Roughness,
                                ));
                            }
                            #[cfg(not(target_arch = "wasm32"))]
                            if ui.button("Browse…").clicked()
                                && let Some(path) = pick_material_texture()
                            {
                                *path_text = path.display().to_string();
                                texture_import = Some((tile, path, GroupTextureKind::Roughness));
                            }
                            if ui.small_button("Reset").clicked() {
                                texture_reset = Some((tile, GroupTextureKind::Roughness));
                            }
                        });
                    });
                }
            });

        #[cfg(not(target_arch = "wasm32"))]
        if let Some((tile, path, kind)) = texture_import {
            self.import_group_texture(context, tile, path, kind);
        }
        if let Some((tile, kind)) = texture_reset {
            match kind {
                GroupTextureKind::BaseColor => {
                    self.uv_space_textures.remove(&tile);
                    self.geo_texture_paths.remove(&tile);
                }
                GroupTextureKind::Normal => {
                    self.geo_normal_textures.remove(&tile);
                    self.geo_normal_paths.remove(&tile);
                }
                GroupTextureKind::Roughness => {
                    self.geo_roughness_textures.remove(&tile);
                    self.geo_roughness_paths.remove(&tile);
                }
            }
            if self.uv_space_textures.contains_key(&tile)
                || self.geo_normal_textures.contains_key(&tile)
                || self.geo_roughness_textures.contains_key(&tile)
            {
                self.rebuild_group_material_bind_group(tile);
            } else {
                self.geo_material_bind_groups.remove(&tile);
            }
            self.editor.status = format!(
                "Reset texture field for UDIM {}",
                1001 + tile.0 + tile.1 * 10
            );
        }
    }

    fn viewport_settings_contents(&mut self, ui: &mut egui::Ui) {
        ui.heading("Background");
        ui.horizontal(|ui| {
            ui.label("Viewport color");
            if ui
                .color_edit_button_rgba_unmultiplied(&mut self.editor.background)
                .changed()
            {
                self.background_color = wgpu::Color {
                    r: self.editor.background[0] as f64,
                    g: self.editor.background[1] as f64,
                    b: self.editor.background[2] as f64,
                    a: self.editor.background[3] as f64,
                };
            }
        });

        ui.separator();
        ui.heading("Points");
        ui.add_enabled_ui(self.show_points, |ui| {
            Self::slider_with_number(ui, "Point size", &mut self.point_size, 2.0..=24.0, 0.5);
            ui.horizontal(|ui| {
                ui.label("Point color");
                ui.color_edit_button_rgba_unmultiplied(&mut self.point_color);
            });
        });

        ui.separator();
        ui.heading("Grid");
        ui.checkbox(&mut self.editor.show_grid, "Show grid");
        let color_changed = ui
            .horizontal(|ui| {
                ui.label("Grid color");

                ui.color_edit_button_rgba_unmultiplied(&mut self.editor.grid_color)
                    .changed()
            })
            .inner;

        let spacing_changed = Self::slider_with_number(
            ui,
            "Grid spacing",
            &mut self.editor.grid_spacing,
            0.1..=10.0,
            0.1,
        );

        if color_changed || spacing_changed {
            self.displayed_grid_spacing = 0.0;
            self.displayed_grid_extent = 0.0;
            self.grid.rebuild(
                &self.device,
                self.editor.grid_size,
                self.editor.grid_spacing,
                self.editor.grid_color,
            );
        }
    }

    fn queue_hdri(&mut self, path: std::path::PathBuf) {
        let is_environment = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                matches!(
                    extension.to_ascii_lowercase().as_str(),
                    "hdr" | "exr" | "rat"
                )
            });
        if !is_environment {
            self.editor.status = "Environment files must use .hdr, .exr, or .rat".to_owned();
            return;
        }

        self.editor.hdri_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned);
        self.editor.pending_hdri = Some(path);
        self.editor.status = "HDRI queued for loading".to_owned();
    }

    fn hdri_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("HDRI Environment");

        let drop_zone = egui::Frame::group(ui.style())
            .inner_margin(12.0)
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.vertical_centered(|ui| {
                    ui.label(
                        self.editor
                            .hdri_name
                            .as_deref()
                            .unwrap_or("Drop a .hdr, .exr, or .rat file here"),
                    );
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(path) = ui
                        .button("Browse…")
                        .clicked()
                        .then(|| {
                            rfd::FileDialog::new()
                                .add_filter("HDR environments", &["hdr", "exr", "rat"])
                                .pick_file()
                        })
                        .flatten()
                    {
                        self.queue_hdri(path);
                    }
                    #[cfg(target_arch = "wasm32")]
                    ui.label("Drag and drop a .hdr or .exr file");
                });
            });

        if drop_zone.response.hovered() {
            let dropped = ui.ctx().input(|input| input.raw.dropped_files.clone());
            for file in dropped {
                if let Some(path) = file.path {
                    self.queue_hdri(path);
                }
            }
        }

        if self.environment.is_some() {
            ui.checkbox(&mut self.editor.hdri_image_disabled, "Disable image")
                .on_hover_text(
                    "Hide the HDRI backdrop while keeping the environment texture loaded",
                );
            ui.label("Intensity, exposure, rotation, and enable state are controlled by the Environment light in the Lighting editor.");
            if ui.button("Remove HDRI").clicked() {
                self.environment = None;
                self.lighting
                    .lights
                    .retain(|light| !matches!(light.kind, lighting::LightKind::Environment));
                self.lighting.touch();
                let (texture, bind_group) = environment::fallback_lighting_bind_group(
                    &self.device,
                    &self.queue,
                    &self.environment_lighting_layout,
                );
                self._fallback_environment_texture = texture;
                self.environment_lighting_bind_group = bind_group;
                self.editor.hdri_name = None;
                self.editor.status = "HDRI removed".to_owned();
            }
        }
    }

    fn load_pending_hdri(&mut self) {
        let Some(path) = self.editor.pending_hdri.take() else {
            return;
        };
        match environment::Environment::from_path(
            &self.device,
            &self.queue,
            self.config.format,
            &path,
        ) {
            Ok(environment) => {
                self.environment_lighting_bind_group = environment
                    .create_lighting_bind_group(&self.device, &self.environment_lighting_layout);
                self.environment = Some(environment);
                if !self
                    .lighting
                    .lights
                    .iter()
                    .any(|light| matches!(light.kind, lighting::LightKind::Environment))
                {
                    self.lighting.add(lighting::LightKind::Environment);
                }
                self.lighting.mode = lighting::ViewportLightingMode::SceneLights;
                self.editor.hdri_image_disabled = false;
                self.editor.status = format!("Loaded HDRI: {}", path.display());
            }
            Err(error) => {
                self.editor.hdri_name = None;
                self.editor.status = format!("Could not load HDRI: {error:#}");
            }
        }
    }

    fn load_pending_texture(&mut self) {
        let mut shared_maps_changed = false;
        if let Some(path) = self.editor.pending_texture.take() {
            match self
                .pbr_material
                .load_base_color(&self.device, &self.queue, &path)
            {
                Ok(()) => self.editor.status = format!("Loaded base color: {}", path.display()),
                Err(error) => self.editor.status = format!("Could not load texture: {error:#}"),
            }
        }
        if let Some(path) = self.editor.pending_normal.take() {
            match self
                .pbr_material
                .load_normal(&self.device, &self.queue, &path)
            {
                Ok(()) => {
                    shared_maps_changed = true;
                    self.editor.status = format!("Loaded normal map: {}", path.display());
                }
                Err(error) => self.editor.status = format!("Could not load normal map: {error:#}"),
            }
        }
        if let Some(path) = self.editor.pending_metallic_roughness.take() {
            match self
                .pbr_material
                .load_metallic_roughness(&self.device, &self.queue, &path)
            {
                Ok(()) => {
                    shared_maps_changed = true;
                    self.editor.status = format!("Loaded metallic/roughness: {}", path.display())
                }
                Err(error) => {
                    self.editor.status = format!("Could not load metallic/roughness map: {error:#}")
                }
            }
        }
        if shared_maps_changed {
            let tiles = self
                .geo_material_bind_groups
                .keys()
                .copied()
                .collect::<Vec<_>>();
            for tile in tiles {
                self.rebuild_group_material_bind_group(tile);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_pending_asset(&mut self) {
        let Some(path) = self.editor.pending_asset.take() else {
            return;
        };
        let loaded_model = match resources::load_model_from_path(&path, &self.device) {
            Ok(model) if !model.meshes.is_empty() => model,
            Ok(_) => {
                self.editor.status = "Imported OBJ contains no meshes".to_owned();
                return;
            }
            Err(error) => {
                self.editor.status = format!("Could not import model: {error:#}");
                return;
            }
        };

        let default_material =
            match material::PbrMaterial::new_untextured(&self.device, &self.queue) {
                Ok(material) => material,
                Err(error) => {
                    self.editor.status = format!("Could not reset material: {error:#}");
                    return;
                }
            };

        // Preserve lighting/environment state exactly; all model-owned and
        // inspection state returns to its launch defaults for the new asset.
        let lighting_editor_state = (
            self.editor.pending_hdri.take(),
            self.editor.hdri_name.clone(),
            self.editor.hdri_image_disabled,
            self.editor.hdri_intensity,
            self.editor.hdri_exposure,
            self.editor.hdri_rotation,
        );
        let mut default_editor = editor_ui::EditorUi::default();
        default_editor.pending_hdri = lighting_editor_state.0;
        default_editor.hdri_name = lighting_editor_state.1;
        default_editor.hdri_image_disabled = lighting_editor_state.2;
        default_editor.hdri_intensity = lighting_editor_state.3;
        default_editor.hdri_exposure = lighting_editor_state.4;
        default_editor.hdri_rotation = lighting_editor_state.5;
        default_editor.asset_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Model")
            .to_owned();
        default_editor.model_path_input = path.display().to_string();
        default_editor.status = format!("Imported model: {}", path.display());

        self.obj_model = loaded_model;
        self.pending_uv_textures.clear();
        self.pending_model_material_root = Some(
            path.parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf(),
        );
        self.pbr_material = default_material;
        self.material_graph = material_graph::MaterialGraphEditor::default();
        self.editor = default_editor;
        self.instances.clone_from(&self.initial_instances);
        self.instance_buffer_dirty = true;
        self.model_view_mode = ModelViewMode::Solid;
        self.show_points = false;
        self.point_size = 7.0;
        self.point_color = [0.0, 0.8, 0.72, 1.0];
        self.show_normals = false;
        self.normal_mode = NormalDisplayMode::Vertex;
        self.normal_length = 0.01;
        self.normal_color = [0.0, 0.8, 0.72, 1.0];
        self.show_uv_map = false;
        self.show_uv_overlay = false;
        self.selected_uv_space = 0;
        self.show_all_uv_spaces = false;
        self.uv_space_offsets.clear();
        self.uv_space_textures.clear();
        self.geo_group_enabled = vec![true; self.obj_model.meshes.len()];
        self.geo_texture_enabled = vec![true; self.obj_model.meshes.len()];
        self.geo_texture_paths.clear();
        self.geo_normal_paths.clear();
        self.geo_roughness_paths.clear();
        self.geo_normal_textures.clear();
        self.geo_roughness_textures.clear();
        self.geo_material_bind_groups.clear();
        self.show_uv_texture = true;
        self.show_uv_lines = true;
        self.uv_inspector_mode = UvInspectorMode::Wireframe;
        self.background_color = wgpu::Color::BLACK;
        self.camera.eye = (8.0, -8.0, 6.0).into();
        self.camera.target = (0.0, 0.0, 0.0).into();
        self.camera.up = cgmath::Vector3::unit_z();
        self.camera.fovy = 45.0;
        self.camera.projection_mode = ProjectionMode::Perspective;
        self.camera.ortho_scale = 10.0;
        self.current_view = None;
        self.displayed_grid_spacing = 0.0;
        self.displayed_grid_extent = 0.0;
        self.grid.rebuild(
            &self.device,
            self.editor.grid_size,
            self.editor.grid_spacing,
            self.editor.grid_color,
        );
        self.window.request_redraw();
    }

    #[cfg(target_arch = "wasm32")]
    fn load_pending_asset(&mut self) {
        if self.editor.pending_asset.take().is_some() {
            self.editor.status =
                "Local model-path import is unavailable in the web build".to_owned();
        }
    }

    fn render(&mut self) -> anyhow::Result<()> {
        self.fps_frames = self.fps_frames.saturating_add(1);
        let sample_elapsed = self.fps_sample_started.elapsed();
        if sample_elapsed.as_secs_f64() >= 0.25 {
            self.displayed_fps = f64::from(self.fps_frames) / sample_elapsed.as_secs_f64();
            self.fps_frames = 0;
            self.fps_sample_started = Instant::now();
        }
        let raw_input = self.egui.state.take_egui_input(&self.window);
        let context = self.egui.context.clone();
        let full_output = context.run_ui(raw_input, |ctx| self.build_editor_ui(ctx));
        self.load_pending_asset();
        self.load_pending_hdri();
        self.load_pending_texture();

        // The gizmo and material widgets mutate render state while this UI frame
        // is being built. Refresh every dependent uniform here so the viewport
        // rendered below uses that exact state and keeps it after the pointer is
        // released.
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        self.material_graph
            .apply_to_uniform(&mut self.pbr_material.uniform);
        self.sync_environment_light();
        self.pbr_material.uniform.properties[3] =
            self.editor.hdri_intensity * self.editor.hdri_exposure.exp2();
        self.pbr_material.uniform.options[0] = self.editor.hdri_rotation.to_radians();
        self.pbr_material.uniform.options[2] = if self.editor.texture_enabled {
            1.0
        } else {
            0.0
        };
        self.pbr_material.uniform.inspection[0] = if self.show_uv_overlay { 1.0 } else { 0.0 };
        self.pbr_material.upload(&self.queue);
        self.lighting_gpu
            .upload(&self.queue, &self.lighting, &self.camera);
        let point_uniform = PointUniform {
            color: self.point_color,
            settings: [
                self.point_size,
                self.config.width.max(1) as f32,
                self.config.height.max(1) as f32,
                0.0,
            ],
        };
        self.queue.write_buffer(
            &self.point_uniform_buffer,
            0,
            bytemuck::bytes_of(&point_uniform),
        );
        let normal_uniform = NormalMarkerUniform {
            color: self.normal_color,
            settings: [
                self.normal_length,
                3.0,
                self.config.width.max(1) as f32,
                self.config.height.max(1) as f32,
            ],
        };
        self.queue.write_buffer(
            &self.normal_uniform_buffer,
            0,
            bytemuck::bytes_of(&normal_uniform),
        );
        self.egui
            .state
            .handle_platform_output(&self.window, full_output.platform_output);
        let paint_jobs = self
            .egui
            .context
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set {
            self.egui
                .renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                anyhow::bail!("Lost Device");
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        if self.material_graph.open {
            self.material_preview.render(
                &mut encoder,
                &self.pbr_material.bind_group,
                &self.environment_lighting_bind_group,
            );
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.background_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            if let Some(environment) = (!self.editor.hdri_image_disabled
                && self.lighting.active_environment().is_some())
            .then_some(self.environment.as_ref())
            .flatten()
            {
                environment.draw(&mut render_pass);
            }
            if self.editor.show_grid {
                self.grid.draw(&mut render_pass, &self.camera_bind_group);
            }
            // Switch from the grid pipeline to the selected model view.
            let model_pipeline = match self.model_view_mode {
                ModelViewMode::Solid => &self.render_pipeline,
                ModelViewMode::Wireframe => self
                    .wireframe_pipeline
                    .as_ref()
                    .unwrap_or(&self.render_pipeline),
            };
            render_pass.set_pipeline(model_pipeline);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(2, &self.environment_lighting_bind_group, &[]);
            render_pass.set_bind_group(3, &self.lighting_gpu.bind_group, &[]);

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            use model::DrawModel;

            for (mesh_index, mesh) in self.obj_model.meshes.iter().enumerate() {
                if !self
                    .geo_group_enabled
                    .get(mesh_index)
                    .copied()
                    .unwrap_or(true)
                {
                    continue;
                }
                let tile = mesh.uv_tile;
                let material_bind_group = self
                    .geo_material_bind_groups
                    .get(&tile)
                    .filter(|_| {
                        self.geo_texture_enabled
                            .get(mesh_index)
                            .copied()
                            .unwrap_or(true)
                    })
                    .unwrap_or(&self.pbr_material.bind_group);
                render_pass.set_bind_group(0, material_bind_group, &[]);
                render_pass.draw_mesh_instanced(mesh, 0..self.editor.visible_instance_count as u32);
            }

            if self.show_points {
                render_pass.set_pipeline(&self.point_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &self.point_bind_group, &[]);
                let stride = std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress;
                for (mesh_index, mesh) in self.obj_model.meshes.iter().enumerate() {
                    if !self
                        .geo_group_enabled
                        .get(mesh_index)
                        .copied()
                        .unwrap_or(true)
                    {
                        continue;
                    }
                    render_pass.set_vertex_buffer(0, mesh.point_marker_buffer.slice(..));
                    for instance in 0..self.editor.visible_instance_count as wgpu::BufferAddress {
                        let start = instance * stride;
                        render_pass.set_vertex_buffer(
                            1,
                            self.instance_buffer.slice(start..start + stride),
                        );
                        render_pass.draw(0..mesh.point_marker_count, 0..1);
                    }
                }
            }
            if self.show_normals {
                render_pass.set_pipeline(&self.normal_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &self.normal_bind_group, &[]);
                let stride = std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress;
                for (mesh_index, mesh) in self.obj_model.meshes.iter().enumerate() {
                    if !self
                        .geo_group_enabled
                        .get(mesh_index)
                        .copied()
                        .unwrap_or(true)
                    {
                        continue;
                    }
                    let (buffer, count) = match self.normal_mode {
                        NormalDisplayMode::Vertex => {
                            (&mesh.vertex_normal_buffer, mesh.vertex_normal_count)
                        }
                        NormalDisplayMode::Point => {
                            (&mesh.point_normal_buffer, mesh.point_normal_count)
                        }
                        NormalDisplayMode::Face => {
                            (&mesh.face_normal_buffer, mesh.face_normal_count)
                        }
                    };
                    render_pass.set_vertex_buffer(0, buffer.slice(..));
                    for instance in 0..self.editor.visible_instance_count as wgpu::BufferAddress {
                        let start = instance * stride;
                        render_pass.set_vertex_buffer(
                            1,
                            self.instance_buffer.slice(start..start + stride),
                        );
                        render_pass.draw(0..count, 0..1);
                    }
                }
            }
        }
        let extra = self.egui.renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            let mut pass = pass.forget_lifetime();
            self.egui.renderer.render(&mut pass, &paint_jobs, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.egui.renderer.free_texture(id);
        }
        self.queue
            .submit(extra.into_iter().chain(std::iter::once(encoder.finish())));
        output.present();
        // Request after presenting, not while handling the current redraw.
        // This guarantees UI-driven material changes get a subsequent update
        // and GPU upload instead of being coalesced into the current frame.
        self.window.request_redraw();

        Ok(())
    }

    //----------------------------------------//

    fn handle_key(&mut self, _event_loop: &ActiveEventLoop, code: KeyCode, state: ElementState) {
        let pressed = state == ElementState::Pressed;

        match code {
            KeyCode::Space => {
                self.houdini_navigation.space_held = pressed;

                if !pressed {
                    self.houdini_navigation.end_mouse_action();
                }
            }

            KeyCode::AltLeft | KeyCode::AltRight => {
                self.houdini_navigation.alt_held = pressed;

                if !pressed {
                    self.houdini_navigation.end_mouse_action();
                }
            }

            KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                self.houdini_navigation.shift_held = pressed;
            }

            KeyCode::ControlLeft | KeyCode::ControlRight => {
                self.houdini_navigation.control_held = pressed;
            }

            KeyCode::Escape if pressed => {
                self.houdini_navigation.end_mouse_action();
            }

            _ if pressed && self.houdini_navigation.view_mode_active() => {
                self.handle_houdini_view_key(code);
            }

            _ => {}
        }
    }
    // KEYBINDING //
    fn handle_houdini_view_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::KeyO => {
                self.camera.projection_mode = match self.camera.projection_mode {
                    ProjectionMode::Perspective => ProjectionMode::Orthographic,
                    ProjectionMode::Orthographic => ProjectionMode::Perspective,
                };
            }

            KeyCode::Digit1 => {
                self.camera.projection_mode = ProjectionMode::Perspective;
                self.current_view = None;
            }

            KeyCode::Digit2 => {
                self.snap_camera_to_view(ViewDirection::Top);
            }

            KeyCode::Digit3 => {
                self.snap_camera_to_view(ViewDirection::Front);
            }

            KeyCode::Digit4 => {
                self.snap_camera_to_view(ViewDirection::Right);
            }

            KeyCode::KeyH => {
                self.home_grid_view();
            }

            KeyCode::KeyA => {
                self.frame_all_instances();
            }

            KeyCode::KeyG => {
                self.frame_selected_instance();
            }

            KeyCode::KeyF => {
                self.frame_selected_instance();
            }

            KeyCode::KeyZ => {
                self.set_tumble_pivot_from_cursor();
            }

            _ => {}
        }
    }
}

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

#[cfg(not(target_arch = "wasm32"))]
fn pick_material_texture() -> Option<std::path::PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Texture image", &["png", "jpg", "jpeg"])
        .pick_file()
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        #[cfg(not(target_arch = "wasm32"))]
        {
            window_attributes =
                window_attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        //run future asyc and use proxy to send results to event loop
        if let Some(proxy) = self.proxy.take() {
            wasm_bindgen_futures::spawn_local(async move {
                assert!(
                    proxy
                        .send_event(State::new(window).await.expect("Unable to create canvas!!"))
                        .is_ok()
                )
            });
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _eventloop: &ActiveEventLoop, mut event: State) {
        // This is where proxy.send_event() ends up
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        let response = state.egui.state.on_window_event(&state.window, &event);
        if response.repaint {
            state.window.request_redraw();
        }

        // Continue processing close, resize, and redraw events regardless.
        // For camera keyboard/mouse controls:

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("{e}");
                        event_loop.exit();
                    }
                }
            }

            // Houdini navigation is a viewport-wide application invariant. Route
            // its modifiers and any active gesture before egui's consumed guard,
            // including when the pointer is over a panel or a light gizmo.
            WindowEvent::Focused(false) => {
                state.houdini_navigation = HoudiniNavigation::default();
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                ..
            } => {
                state.houdini_navigation.end_mouse_action();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } if matches!(
                code,
                KeyCode::Space
                    | KeyCode::AltLeft
                    | KeyCode::AltRight
                    | KeyCode::ShiftLeft
                    | KeyCode::ShiftRight
                    | KeyCode::ControlLeft
                    | KeyCode::ControlRight
            ) =>
            {
                state.handle_key(event_loop, code, key_state)
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } if state.houdini_navigation.view_mode_active() => {
                state.houdini_navigation.begin_mouse_action(button);
            }

            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                state.houdini_navigation.cursor_position = Some(current);

                let previous = state.houdini_navigation.previous_cursor.replace(current);

                if let (Some(previous), Some(action)) =
                    (previous, state.houdini_navigation.active_action)
                {
                    let delta_x = (current.0 - previous.0) as f32;
                    let delta_y = (current.1 - previous.1) as f32;

                    match action {
                        NavigationAction::Tumble => state.tumble_camera(delta_x, delta_y),
                        NavigationAction::Track => state.track_camera(delta_x, delta_y),
                        NavigationAction::Dolly => state.dolly_camera(delta_y),
                        NavigationAction::Tilt => state.tilt_camera(delta_x),
                        NavigationAction::LensZoom => state.zoom_camera_lens(delta_y),
                    }
                    state.window.request_redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. }
                if state.houdini_navigation.view_mode_active() =>
            {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y * 20.0,
                    MouseScrollDelta::PixelDelta(position) => -position.y as f32,
                };
                state.dolly_camera(amount);
                state.window.request_redraw();
            }

            _ if response.consumed => {}

            // MouseINPUT //
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                if button_state == ElementState::Pressed {
                    state.houdini_navigation.begin_mouse_action(button);
                }
            }

            // Cursor Leaving the window //
            WindowEvent::CursorLeft { .. } => {
                state.houdini_navigation.end_mouse_action();
            }

            // Mouse Wheel //
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y * 20.0,
                    MouseScrollDelta::PixelDelta(position) => -position.y as f32,
                };

                state.dolly_camera(amount);
                state.window.request_redraw();
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Keep the viewport render loop alive independently of pointer events.
        // Material-graph topology changes (notably connecting Metallic) can be
        // committed at the end of an egui frame; the following frame must run
        // immediately so the main viewport consumes the newly evaluated value.
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = App::new();
        event_loop.run_app(&mut app)?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let app = App::new(&event_loop);
        event_loop.spawn_app(app);
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
