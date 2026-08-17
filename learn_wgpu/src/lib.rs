use std::sync::Arc;
use cgmath::prelude::*;
use serde::{Deserialize, Serialize};
use crate::texture::Texture;
use model::Vertex;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop,
    EventLoop},
    keyboard::{KeyCode,
    PhysicalKey},
    window::{Window}
};

use wgpu::util::DeviceExt;
use navigation_gizmo::ViewDirection;

mod texture;
mod model;
mod resources;
mod editor_ui;
mod egui_renderer;
mod grid;
mod navigation_gizmo;


#[cfg(target_arch ="wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;


const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];
//----------------------------------------//
//CAMERA CONTROLLER

struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed:false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn handle_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.is_right_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();
        let right = forward_norm.cross(camera.up);

        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }


        if self.is_right_pressed{
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
         if self.is_left_pressed{
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum CompareFunction{
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
                    cgmath::perspective(
                        cgmath::Deg(self.fovy),
                        self.aspect,
                        self.znear,
                        self.zfar,
                    )
            }

            ProjectionMode::Orthographic => {
                let half_height = self.ortho_scale *0.5;

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
        return OPENGL_TO_WGPU_MATRIX * projection * view
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
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
//----------------------------------------//
//Instace Buffer

#[derive(Clone)]
struct Instance {
    position: cgmath::Vector3<f32>,
    rotation_degrees: cgmath::Vector3<f32>,
    scale: cgmath::Vector3<f32>,
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
            self.scale.x, self.scale.y, self.scale.z,
        );

        InstanceRaw {
            model: (translation * rz * ry * rx * scale).into(),
    }
}}


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
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    obj_model: model::Model,
    depth_texture: Texture,

    egui: egui_renderer::EguiRenderer,
    editor: editor_ui::EditorUi,
    instance_buffer_dirty: bool,
    background_color: wgpu::Color,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    
    grid: grid::Grid,
    initial_camera: Camera,
    initial_instances: Vec<Instance>,
    initial_editor: editor_ui::EditorUi,
    
    current_view: Option<ViewDirection>,
}

impl State{
        async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();

        const NUM_INSTANCES_PER_ROW: u32 =10;
        
            let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let position = cgmath::Vector3 { x: x as f32, y: z as f32, z: 0.0,};

                    let rotation = if position.is_zero(){
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                Instance {
                    position, 
                    rotation_degrees: cgmath::Vector3::new(0.0, 45.0, 0.0), 
                    scale: cgmath::Vector3::new(0.01, 0.01, 0.01),
                }
            })
        }).collect::<Vec<_>>();


        //handle GPU
        //Backend::PRIMARY=> Vulkan + Metal +DX12
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor{
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
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
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            }
        );


        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );
        let num_indices =INDICES.len() as u32;

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

        let egui= egui_renderer::EguiRenderer::new(
            &window,
            &device,
            config.format,
        );
        let editor = editor_ui::EditorUi::default();
        let background_color = wgpu::Color {
            r: editor.background[0] as f64,
            g: editor.background[1] as f64,
            b: editor.background[2] as f64,
            a: editor.background[3] as f64,
        };

        let diffuse_bytes = include_bytes!("../res/happy-tree.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, &diffuse_image, Some("happy-tree.png")).unwrap();

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
        });

        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );



//----------------------------------------//
//Depth Texture
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");


//----------------------------------------//
        let camera_controller = CameraController::new(0.2);

        let camera = Camera { 
            eye: (8.0, -8.0, 6.0).into(), 
            target: (0.0, 0.0, 0.0).into(), 
            up: cgmath::Vector3::unit_z(), 
            aspect: config.width as f32 / config.height as f32, 
            fovy: 45.0, 
            znear: 0.1, 
            zfar: 100.0,

            projection_mode: ProjectionMode::Perspective,
            ortho_scale: 10.0, 
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_bund_group_layout"),
        });

        let camera_bind_group = device.create_bind_group (&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });

        let render_pipeline_layout = device.create_pipeline_layout (
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&texture_bind_group_layout),
                    Some(&camera_bind_group_layout),
                ],
                immediate_size:0,
            }
        );


//----------------------------------------//
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipline"),
            layout: Some(&render_pipeline_layout),
            fragment: Some(wgpu::FragmentState{
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
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
                buffers: &[
                    model::ModelVertex::desc(),
                    InstanceRaw::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },

            multiview_mask: None,
            cache: None,

        });

        let obj_model = resources::load_model(
            "t-pose.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        )
        .await?;

//----------------------------------------//

        let grid= grid::Grid::new(
            &device,
            config.format,
            &camera_bind_group_layout,
            editor.grid_size,
            editor.grid_spacing,
            editor.grid_color,
        );

//----------------------------------------//
        let initial_camera = camera.clone();
        let initial_instances = instances.clone();
        let initial_editor = editor.clone();

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            window,
            diffuse_bind_group,
            diffuse_texture,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            instances,
            instance_buffer,
            obj_model,
            depth_texture,

            egui,
            editor,
            instance_buffer_dirty: false,
            background_color,
            texture_bind_group_layout,
            grid,
            initial_camera,
            initial_instances,
            initial_editor,

            current_view: None,
        })
    }



//----------------------------------------//
    pub fn resize(&mut self, width:u32, height:u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;

            self.camera.aspect = width as f32 / height as f32;

            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");

            self.is_surface_configured = true;
            self.window.request_redraw();
        }

    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
        if self.instance_buffer_dirty {
            self.upload_instances();
            self.instance_buffer_dirty = false;
        }
    }

    fn upload_instances(&self) {
        let data = self.instances
        .iter()
        .map(Instance::to_raw)
        .collect::<Vec<_>>();

        self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&data),
        );
    }

    fn reset_all(&mut self) {
        let editor_window_offset = self.editor.editor_window_offset;

        self.instances.clone_from(&self.initial_instances);
        self.editor.clone_from(&self.initial_editor);
        self.editor.editor_window_offset = editor_window_offset;
        self.camera.clone_from(&self.initial_camera);

        // Keep the projection correct if the window was resized after startup.
        self.camera.aspect = self.config.width as f32 / self.config.height as f32;

        self.background_color = wgpu::Color {
            r: self.editor.background[0] as f64,
            g: self.editor.background[1] as f64,
            b: self.editor.background[2] as f64,
            a: self.editor.background[3] as f64,
        };

        self.grid.rebuild(
            &self.device,
            self.editor.grid_size,
            self.editor.grid_spacing,
            self.editor.grid_color,
        );

        self.camera_controller = CameraController::new(0.2);

        self.instance_buffer_dirty = true;
        self.window.request_redraw();
    }

    fn snap_camera_to_view( &mut self, view: ViewDirection,) {
        let distance = (self.camera.eye - self.camera.target)
            .magnitude()
            .max(0.01);

        let direction = match view {
            ViewDirection::Right => {
                cgmath::Vector3::new(1.0,0.0, 0.0)
            }

            ViewDirection::Left => {
                cgmath::Vector3::new(-1.0,0.0, 0.0)
            }

            ViewDirection::Back => {
                cgmath::Vector3::new(0.0,1.0, 0.0)
            }

            ViewDirection::Front => {
                cgmath::Vector3::new(0.0,-1.0, 0.0)
            }

            ViewDirection::Top => {
                cgmath::Vector3::new(0.0,0.0, 1.0)
            }

            ViewDirection::Bottom => {
                cgmath::Vector3::new(0.0,0.0, -1.0)
            }
        };

        self.camera.eye = self.camera.target + direction * distance;

        self.camera.up = match view {
            ViewDirection::Right
            | ViewDirection::Left
            | ViewDirection::Back
            | ViewDirection::Front => {
                cgmath::Vector3::unit_z()
            }

            ViewDirection::Top => {
                cgmath::Vector3::unit_y()
            }

            ViewDirection::Bottom => {
                -cgmath::Vector3::unit_y()
            }
        };

        self.camera.projection_mode = ProjectionMode::Orthographic;

        self.current_view = Some(view);

        // Clear held camera movement keys
        self.camera_controller = CameraController::new(0.2);

        self.window.request_redraw();
    }

    fn current_view_name(&self) -> &'static str {
        match self.current_view {
            Some(ViewDirection::Right) => {"Right Orthographic"}
            Some(ViewDirection::Left) => {"Left Orthographic"}
            Some(ViewDirection::Back) => {"Back Orthographic"}
            Some(ViewDirection::Front) => {"Front Orthographic"}
            Some(ViewDirection::Top) => {"Top Orthographic"}
            Some(ViewDirection::Bottom) => {"Bottom Orthographic"}
            None => match self.camera.projection_mode {
                ProjectionMode::Perspective => {"User Perspective"} 
                ProjectionMode::Orthographic => {"User Orthographic"}
            },
        }
    }

    fn camera_information_ui(
        &self,
        context: &egui::Context,
    ) {
        egui::Area::new(
            egui::Id::new("camera_information"),
        )
        .anchor(
            egui::Align2::LEFT_TOP,
            egui::vec2(12.0, 12.0),
        )
        .show(context, |ui| {
            egui::Frame::new()
                .fill(
                    egui::Color32::from_black_alpha(160),
                )
                .corner_radius(5.0)
                .inner_margin(8.0)
                .show(ui, |ui| {
                    ui.label(self.current_view_name());

                    ui.label(format!(
                        "Eye: {:.2}, {:.2}, {:.2}",
                        self.camera.eye.x,
                        self.camera.eye.y,
                        self.camera.eye.z,
                    ));

                    ui.label(format!(
                        "Target: {:.2}, {:.2}, {:.2}",
                        self.camera.target.x,
                        self.camera.target.y,
                        self.camera.target.z,
                    ));

                    let distance = (self.camera.eye - self.camera.target ).magnitude();

                    ui.label(format!(
                        "Distance: {:.2}",
                        distance,
                    ));
                });
        });
    }

    //--------------------------------------------------------------------//
    // UI STUFF//
    fn build_editor_ui(&mut self, ui: &mut egui::Ui) {
        let editor_window_offset = self.editor.editor_window_offset;
        let mut editor_drag_delta = egui::Vec2::ZERO;

        egui::Window::new("Viewport Editor")
        .id(egui::Id::new("viewport_editor_window",))
        // Anchored egui windows are positioned from the current viewport on
        // every frame, so this reliably follows the upper-right corner when
        // the native window changes size.
        .anchor(
            egui::Align2::RIGHT_TOP,
            egui::vec2(editor_window_offset[0], editor_window_offset[1]),
        )
        .default_size(egui::vec2(240.0, 650.0))
        .min_width(240.0)
        .resizable(true)
        .collapsible(true)
        .constrain(true)
        .vscroll(true)
        .show(ui, |ui| {
            let (drag_rect, drag_response) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 26.0),
                egui::Sense::drag(),
            );

            let drag_color = if drag_response.dragged() || drag_response.hovered() {
                ui.visuals().widgets.hovered.fg_stroke.color
            } else {
                ui.visuals().weak_text_color()
            };

            ui.painter().text(
                drag_rect.center(),
                egui::Align2::CENTER_CENTER,
                "☰ Drag window",
                egui::FontId::proportional(13.0),
                drag_color,
            );

            if drag_response.dragged() {
                editor_drag_delta = ui.ctx().input(|input| input.pointer.delta());
            }

            ui.separator();

            if ui.button("Reset All").clicked() { self.reset_all();}

            ui.separator();
            self.asset_drop_ui(ui);
            ui.separator();
            self.transform_ui(ui);
            ui.separator();
            self.texture_ui(ui);
            ui.separator();
            self.viewport_ui(ui);
            ui.separator();
            ui.label(format!("Status: {}", self.editor.status));
        });

        if editor_drag_delta != egui::Vec2::ZERO {
            self.editor.editor_window_offset[0] += editor_drag_delta.x;
            self.editor.editor_window_offset[1] += editor_drag_delta.y;
        }

        let context = ui.ctx().clone();

        let clicked_view = navigation_gizmo::show(
            &context,
            self.camera.eye,
            self.camera.target,
            self.camera.up,
        );

        if let Some(view) = clicked_view {
            self.snap_camera_to_view(view);
        }

        self.camera_information_ui(&context);
    }

    fn slider_with_number (
        ui: &mut egui::Ui,
        label: &str,
        value: &mut f32,
        range: std::ops::RangeInclusive<f32>,
        speed: f64,
    ) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(label);
            changed |= ui.add(
                egui::Slider::new(value, range.clone()).show_value(false)
            ).changed();
            changed |= ui.add(
                egui::DragValue::new(value).speed(speed).range(range)
            ).changed();
        });
        changed
    }

    fn asset_drop_ui(&mut self, ui: &mut egui::Ui) {
        let dropped_files = ui.ctx().input(|input|{
            input.raw.dropped_files.clone()
        });

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
                "obj" => {
                    self.editor.asset_name = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("Model")
                        .to_owned();

                    self.editor.pending_asset = Some(path);
                    self.editor.status =
                        "Model queued for loading".to_owned();
                }

                "fbx" | "usd" | "usda" | "usdc" => {
                    self.editor.status = format!("{extension} import is not implemeted");
                }

                _=>{}
            }
        }
        ui.label(format!("Model: {}", self.editor.asset_name));
    }

    fn transform_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Transform");

        if self.instances.is_empty() {
            ui.label("No instances");
            return;
        }

        self.editor.visible_instance_count = self.editor.visible_instance_count.clamp(1, self.instances.len());
        
        let amount_response = ui.add(
            egui::Slider::new(
                &mut self.editor.visible_instance_count, 1..=self.instances.len(),
            )
            .text("Number of Instances")
            .show_value(true),
        );

        if amount_response.changed() {
            self.editor.status = format!( "{} of instances", self.editor.visible_instance_count,);
        }
        
        let last_instance = self.editor.visible_instance_count -1;

        self.editor.selected_instance = self.editor.selected_instance.min(last_instance);

        let instance_response = ui.add(
            egui::Slider::new(
                &mut self.editor.selected_instance,
                0..=last_instance,
            )
            .text("Instance")
            .show_value(true),
        );

        if instance_response.changed() {
            self.editor.status = format!(
                "Selected instance {}", self.editor.selected_instance,
            );
        }

        ui.label(format!(
            "Editing instance {} of {}",self.editor.selected_instance + 1, self.editor.visible_instance_count,
        ));

        let item = &mut self.instances[self.editor.selected_instance];

        let mut changed = false;

        // UI--transform interface controls
        ui.label("Translation");
        changed |= Self::slider_with_number(ui, "X", &mut item.position.x, -50.0..=50.0, 0.05);
        changed |= Self::slider_with_number(ui, "Y", &mut item.position.y, -50.0..=50.0, 0.05);
        changed |= Self::slider_with_number(ui, "Z", &mut item.position.z, -50.0..=50.0, 0.05);
        ui.label("Rotation (degrees)");
        changed |= Self::slider_with_number(ui, "X", &mut item.rotation_degrees.x, -180.0..=180.0, 0.25);
        changed |= Self::slider_with_number(ui, "Y", &mut item.rotation_degrees.y, -180.0..=180.0, 0.25);
        changed |= Self::slider_with_number(ui, "Z", &mut item.rotation_degrees.z, -180.0..=180.0, 0.25);
        ui.label("Scale");
        changed |= Self::slider_with_number(ui, "X", &mut item.scale.x, 0.0001..=10.0, 0.001);
        changed |= Self::slider_with_number(ui, "Y", &mut item.scale.y, 0.0001..=10.0, 0.001);
        changed |= Self::slider_with_number(ui, "Z", &mut item.scale.z, 0.0001..=10.0, 0.001);
        if changed { self.instance_buffer_dirty = true; }
    }

    fn texture_ui (&mut self, ui: &mut egui::Ui) {
        ui.heading("Texture");

        ui.checkbox(&mut self.editor.texture_enabled, "Texture enabled");
        
        let dropped_files = ui.ctx().input(|input| {
            input.raw.dropped_files.clone()
        });

        for file in dropped_files {
            let Some(path) = file.path else {
                continue;
            };

            let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

            if matches!(extension.as_str(), "png" | "jpg" | "jpeg" ) {
                self.editor.texture_name =path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Texture")
                .to_owned();

                self.editor.pending_texture = Some(path);
                self.editor.status = "Texture queued for loading".to_owned();
            }
        }
    }

    fn viewport_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Viewport");
        if ui.color_edit_button_rgba_unmultiplied(
            &mut self.editor.background
        ).changed() {
            self.background_color =wgpu::Color {
                r: self.editor.background[0] as f64,
                g: self.editor.background[1] as f64,
                b: self.editor.background[2] as f64,
                a: self.editor.background[3] as f64,
            };
        }
        ui.checkbox(&mut self.editor.show_grid, "Show grid");

        let color_changed = ui.horizontal(|ui| {
            ui.label("Grid color");

            ui.color_edit_button_rgba_unmultiplied(
                &mut self.editor.grid_color,
            )
            .changed()
        })
        .inner;

        let size_changed = Self::slider_with_number(
            ui,
            "Grid size",
            &mut self.editor.grid_size,
            0.1..=100.0,
            0.5,
        );

        let spacing_changed = Self::slider_with_number(
            ui,
            "Grid spacing",
            &mut self.editor.grid_spacing,
            0.1..=10.0,
            0.1,
        );

        if color_changed || size_changed || spacing_changed {
            self.grid.rebuild(
                &self.device,
                self.editor.grid_size,
                self.editor.grid_spacing,
                self.editor.grid_color,
            );
        }
    }

    fn render(&mut self) -> anyhow::Result<()>{
        self.window.request_redraw();
        
        let raw_input = self.egui.state.take_egui_input (&self.window);
        let context = self.egui.context.clone();
        let full_output = context.run_ui(raw_input, |ctx| self.build_editor_ui(ctx));
        self.egui.state.handle_platform_output (
            &self.window,
            full_output.platform_output,
        );
        let paint_jobs = self.egui.context.tessellate(
            full_output.shapes,
            full_output.pixels_per_point,
        );
        for (id, delta) in &full_output.textures_delta.set {
            self.egui.renderer.update_texture(
                &self.device, &self.queue, *id, delta
            );
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        if !self.is_surface_configured{
            return Ok(());
        }



        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation =>{
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
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
    
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        

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
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment{
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops:None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        if self.editor.show_grid {
            self.grid.draw(
                &mut render_pass,
                &self.camera_bind_group,
            );
        }
        // Switch from the grid pipeline back to the model pipeline.
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

        use model::DrawModel;
        
        render_pass.draw_mesh_instanced(&self.obj_model.meshes[0], 0..self.editor.visible_instance_count as u32);
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
        self.queue.submit(
            extra.into_iter().chain(std::iter::once(encoder.finish()))
        );
    output.present();

    Ok(())
    }

//----------------------------------------//

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        if code == KeyCode::Escape && is_pressed {
            event_loop.exit();
        } else {
            let handled = self.camera_controller.handle_key(code, is_pressed);

            if handled && is_pressed {
                self.current_view = None;
                self.camera.projection_mode = ProjectionMode::Perspective;
            }
        }
    }

}



pub struct App{
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App{
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop:&EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }

        
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

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
            wasm_bindgen_futures::spawn_local(async move{
                assert!(proxy
                    .send_event(
                        State::new(window)
                            .await
                            .expect("Unable to create canvas!!")
                    )
                    .is_ok())
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
            let state = match &mut self.state{
                Some(canvas) => canvas,
                None => return,
            };

            let response = state.egui.state.on_window_event (
                &state.window,
                &event,
            );
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
                        Err(e) =>{
                            log::error!("{e}");
                            event_loop.exit();
                        }
                    }
                }


                _ if response.consumed => return,

                WindowEvent::MouseInput {  state, button, .. } => match (button,state.is_pressed()){
                    (MouseButton::Left,true) => {}
                    (MouseButton::Right,true) => {}
                    _ =>{}
                },
                WindowEvent::KeyboardInput {
                    event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
                } => state.handle_key(event_loop, code, key_state.is_pressed()),
                _ => {}
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
