use anyhow::Context;
use crate::texture::Texture;
use cgmath::prelude::*;
use model::Vertex;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap, collections::HashMap, collections::HashSet, sync::Arc, time::Instant,
};

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

const GENERATED_SHAPE_LIGHT_TARGET_BASE: usize = usize::MAX / 2;

#[cfg(not(target_arch = "wasm32"))]
mod asset_explorer;
mod editor_ui;
mod demos;
mod user_data;
mod preferences;
mod egui_renderer;
mod environment;
mod grid;
mod ground_plane;
mod houdini_navigation;
mod lighting;
mod material;
mod material_graph;
mod material_library;
mod material_preview;
mod model;
mod navigation_gizmo;
mod object_gizmo;
mod part_selection;
mod resources;
mod usd_import;
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

#[derive(Clone, PartialEq)]
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

fn ray_box_distance(
    origin: cgmath::Vector3<f32>,
    direction: cgmath::Vector3<f32>,
    minimum: cgmath::Vector3<f32>,
    maximum: cgmath::Vector3<f32>,
) -> Option<f32> {
    let mut near = f32::NEG_INFINITY;
    let mut far = f32::INFINITY;
    for (origin, direction, minimum, maximum) in [
        (origin.x, direction.x, minimum.x, maximum.x),
        (origin.y, direction.y, minimum.y, maximum.y),
        (origin.z, direction.z, minimum.z, maximum.z),
    ] {
        if direction.abs() < 1.0e-8 {
            if origin < minimum || origin > maximum {
                return None;
            }
            continue;
        }
        let first = (minimum - origin) / direction;
        let second = (maximum - origin) / direction;
        near = near.max(first.min(second));
        far = far.min(first.max(second));
        if near > far {
            return None;
        }
    }
    (far >= 0.0).then_some(near.max(0.0))
}

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

#[derive(Clone, PartialEq)]
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum GroupTextureKind {
    Emissive,
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
    enable_emission: bool,
    receiver: std::sync::mpsc::Receiver<Result<DecodedUvTexture, String>>,
}

#[cfg(not(target_arch = "wasm32"))]
struct PendingModelLoad {
    path: std::path::PathBuf,
    time_code: Option<f64>,
    loading_fx_project: bool,
    receiver: std::sync::mpsc::Receiver<Result<resources::PreparedModel, String>>,
}

#[derive(Clone, PartialEq)]
struct GeneratedGeometryUndo {
    scene_id: u64,
    generated: bool,
    selected: bool,
    snap_bottom_to_grid: bool,
    kind: ground_plane::BasicShape,
    visible: bool,
    wireframe: bool,
    hide_surface: bool,
    size: f32,
    height: f32,
    position_x: f32,
    position_y: f32,
    rotation_degrees: [f32; 3],
    shape_height: f32,
    thickness: f32,
    subdivisions: u32,
    triangulate_subdivision: bool,
    uv_scale: f32,
    material: material::MaterialUniform,
    base_color_path: String,
    normal_path: String,
    roughness_path: String,
    emissive_path: String,
}

#[derive(Clone, PartialEq)]
struct UndoSnapshot {
    deconstruction: demos::Deconstruction,
    instances: Vec<Instance>,
    gizmo_at_bottom_by_instance: Vec<bool>,
    gizmo_always_at_bottom: bool,
    lighting: lighting::LightingManager,
    generated_objects: Vec<GeneratedGeometryUndo>,
    pbr_material: material::MaterialUniform,
    model_view_mode: ModelViewMode,
    show_points: bool,
    point_size: f32,
    point_color: [f32; 4],
    show_normals: bool,
    normal_mode: NormalDisplayMode,
    normal_length: f32,
    normal_color: [f32; 4],
    geo_group_enabled: Vec<bool>,
    geo_texture_enabled: Vec<bool>,
    geo_texture_scales: BTreeMap<(i32, i32), f32>,
    geo_texture_colors: BTreeMap<(i32, i32), [f32; 4]>,
    geo_normal_strengths: BTreeMap<(i32, i32), f32>,
    geo_color_settings: BTreeMap<(i32, i32), GroupColorSettings>,
}

type UvEdge = ([f32; 2], [f32; 2]);
type CachedUvSpaces = Vec<((i32, i32), (Arc<str>, Arc<[UvEdge]>))>;

const FX_PROJECT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct FxProject {
    #[serde(default)]
    deconstruction: demos::Deconstruction,
    format: String,
    version: u32,
    imported_model: Option<FxImportedModel>,
    generated_objects: Vec<FxGeneratedObject>,
    material: FxMaterial,
    material_graph: material_graph::MaterialGraph,
    lights: FxLighting,
    grid: FxGrid,
    viewport: FxViewport,
    environment: FxEnvironment,
    #[serde(default)]
    material_library: Vec<FxLibraryMaterial>,
    #[serde(default)]
    mesh_material_assignments: Vec<u64>,
    #[serde(default)]
    face_material_assignments: Vec<Vec<material_library::FaceMaterialAssignment>>,
    #[serde(default)]
    generated_material_assignments: BTreeMap<u64, u64>,
}

#[derive(Serialize, Deserialize)]
struct FxImportedModel {
    #[serde(default)]
    usd_time_code: Option<f64>,
    path: String,
    instances: Vec<FxTransform>,
    visible_instance_count: usize,
    selected_instance: usize,
    gizmo_at_bottom: Vec<bool>,
    group_visibility: Vec<bool>,
    group_texture_enabled: Vec<bool>,
    textures: Vec<FxGroupTextures>,
}

#[derive(Serialize, Deserialize)]
struct FxTransform {
    position: [f32; 3],
    rotation_degrees: [f32; 3],
    scale: [f32; 3],
    global_scale: f32,
}

#[derive(Serialize, Deserialize)]
struct FxGeneratedObject {
    scene_id: u64,
    selected: bool,
    kind: ground_plane::BasicShape,
    visible: bool,
    snap_bottom_to_grid: bool,
    wireframe: bool,
    hide_surface: bool,
    size: f32,
    height: f32,
    position_x: f32,
    position_y: f32,
    rotation_degrees: [f32; 3],
    shape_height: f32,
    #[serde(default)]
    thickness: f32,
    subdivisions: u32,
    triangles: bool,
    uv_scale: f32,
    material: FxMaterial,
}

#[derive(Clone, Serialize, Deserialize)]
struct FxMaterial {
    #[serde(default = "material::default_transparency")]
    transparency: [f32; 4],
    #[serde(default)]
    color_adjustments: [f32; 4],
    #[serde(default = "default_texture_color")]
    emissive_color: [f32; 4],
    base_color: [f32; 4],
    properties: [f32; 4],
    options: [f32; 4],
    inspection: [f32; 4],
    base_color_path: String,
    normal_path: String,
    roughness_path: String,
    #[serde(default)]
    emissive_path: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct FxLibraryMaterial {
    id: u64,
    name: String,
    material: FxMaterial,
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct GroupColorSettings {
    hue: f32,
    emission_hue: f32,
    intensity: f32,
    tint: [f32; 4],
    transparency: [f32; 4],
    double_sided: bool,
}
impl Default for GroupColorSettings {
    fn default() -> Self { Self { hue: 0.0, emission_hue: 0.0, intensity: 0.0, tint: [1.0; 4], transparency: material::default_transparency(), double_sided: false } }
}

#[derive(Serialize, Deserialize)]
struct FxGroupTextures {
    tile: (i32, i32),
    base_color: String,
    normal: String,
    roughness: String,
    #[serde(default)]
    emissive: String,
    #[serde(default)]
    colors: Option<GroupColorSettings>,
    #[serde(default = "default_texture_scale")]
    scale: f32,
    #[serde(default = "default_texture_color")]
    color: [f32; 4],
    #[serde(default = "default_normal_strength")]
    normal_strength: f32,
}

fn default_texture_scale() -> f32 {
    1.0
}

fn default_texture_color() -> [f32; 4] {
    [1.0; 4]
}

fn default_normal_strength() -> f32 {
    1.0
}

#[derive(Serialize, Deserialize)]
struct FxLighting {
    mode: lighting::ViewportLightingMode,
    lights: Vec<lighting::SceneLight>,
    selected_light: Option<lighting::LightId>,
    area_samples: u32,
    environment_samples: u32,
}

#[derive(Serialize, Deserialize)]
struct FxGrid {
    visible: bool,
    size: f32,
    spacing: f32,
    color: [f32; 4],
}

#[derive(Serialize, Deserialize)]
struct FxViewport {
    background: [f32; 4],
    gizmo_always_at_bottom: bool,
    wireframe: bool,
    show_points: bool,
    point_size: f32,
    point_color: [f32; 4],
    show_normals: bool,
    normal_mode: u8,
    normal_length: f32,
    normal_color: [f32; 4],
    show_uv_map: bool,
    show_uv_overlay: bool,
    show_uv_texture: bool,
    show_uv_lines: bool,
    camera_eye: [f32; 3],
    camera_target: [f32; 3],
    camera_up: [f32; 3],
    camera_fovy: f32,
    orthographic: bool,
    ortho_scale: f32,
}

#[derive(Serialize, Deserialize)]
struct FxEnvironment {
    path: String,
    disabled: bool,
    intensity: f32,
    exposure: f32,
    rotation: f32,
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
    transparent_pipeline: wgpu::RenderPipeline,
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
    uv_import_kind: GroupTextureKind,
    show_all_uv_spaces: bool,
    uv_space_offsets: BTreeMap<(i32, i32), egui::Vec2>,
    uv_space_cache: CachedUvSpaces,
    uv_inspector_mode: UvInspectorMode,
    uv_space_textures: BTreeMap<(i32, i32), UvSpaceTexture>,
    geo_group_enabled: Vec<bool>,
    geo_texture_enabled: Vec<bool>,
    geo_texture_paths: BTreeMap<(i32, i32), String>,
    geo_normal_paths: BTreeMap<(i32, i32), String>,
    geo_roughness_paths: BTreeMap<(i32, i32), String>,
    geo_emissive_paths: BTreeMap<(i32, i32), String>,
    geo_normal_textures: BTreeMap<(i32, i32), texture::Texture>,
    geo_roughness_textures: BTreeMap<(i32, i32), texture::Texture>,
    geo_emissive_textures: BTreeMap<(i32, i32), texture::Texture>,
    geo_material_bind_groups: BTreeMap<(i32, i32), wgpu::BindGroup>,
    geo_material_uniform_buffers: BTreeMap<(i32, i32), wgpu::Buffer>,
    geo_texture_scales: BTreeMap<(i32, i32), f32>,
    geo_texture_colors: BTreeMap<(i32, i32), [f32; 4]>,
    geo_normal_strengths: BTreeMap<(i32, i32), f32>,
    geo_color_settings: BTreeMap<(i32, i32), GroupColorSettings>,
    #[cfg(not(target_arch = "wasm32"))]
    pending_uv_textures: Vec<PendingUvTexture>,
    #[cfg(not(target_arch = "wasm32"))]
    pending_model_material_root: Option<std::path::PathBuf>,
    pending_project: Option<FxProject>,
    #[cfg(not(target_arch = "wasm32"))]
    pending_model_load: Option<PendingModelLoad>,
    project_path: String,
    fx_load_started: Option<Instant>,
    material_texture_cache: HashMap<(std::path::PathBuf, bool, u64, u128), Arc<texture::Texture>>,
    generated_mesh_cache:
        HashMap<(ground_plane::BasicShape, u32, bool), ground_plane::SharedPrimitiveMesh>,
    show_uv_texture: bool,
    show_uv_lines: bool,
    preferences: user_data::Preferences,
    #[cfg(not(target_arch = "wasm32"))]
    preference_store: user_data::PreferenceStore,
    #[cfg(not(target_arch = "wasm32"))]
    user_library: user_data::UserLibrary,
    viewport_settings_open: bool,
    viewport_settings_just_opened: bool,
    gizmo_at_bottom_by_instance: Vec<bool>,
    gizmo_always_at_bottom: bool,
    wireframe_pipeline: Option<wgpu::RenderPipeline>,
    quad_line_pipeline: wgpu::RenderPipeline,
    model_view_mode: ModelViewMode,
    pbr_material: material::PbrMaterial,
    wireframe_material: material::PbrMaterial,
    material_graph: material_graph::MaterialGraphEditor,
    material_preview: material_preview::MaterialPreview,
    material_previews: HashMap<material_library::MaterialId, material_preview::MaterialPreview>,
    material_library: Vec<material_library::SceneMaterial>,
    next_material_id: u64,
    mesh_material_assignments: Vec<material_library::MaterialId>,
    face_material_assignments: Vec<Vec<material_library::FaceMaterialAssignment>>,
    generated_material_assignments: BTreeMap<u64, material_library::MaterialId>,
    material_browser_search: String,
    selected_library_material: material_library::MaterialId,
    dragged_material: Option<material_library::MaterialId>,
    face_assign_first: u32,
    face_assign_end: u32,
    selected_material_mesh: usize,
    part_selection: part_selection::PartSelection,
    lighting: lighting::LightingManager,
    shadow_renderer: lighting::shadow::ShadowRenderer,
    lighting_gpu: lighting::LightingGpu,
    deconstruction: demos::Deconstruction,
    demo_buffers: Vec<wgpu::Buffer>,
    demo_cached: Option<(demos::Deconstruction, Vec<Instance>)>,
    active_side_panel: Option<usize>,
    explorer_open: bool,
    #[cfg(not(target_arch = "wasm32"))]
    asset_explorer: asset_explorer::AssetExplorer,
    generate_popup_open: bool,
    generate_popup_position: egui::Pos2,
    outliner_search: String,
    environment_lighting_layout: wgpu::BindGroupLayout,
    environment_lighting_bind_group: wgpu::BindGroup,
    _fallback_environment_texture: wgpu::Texture,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    grid_camera_buffer: wgpu::Buffer,
    grid_camera_bind_group: wgpu::BindGroup,
    houdini_navigation: HoudiniNavigation,
    undo_modifier_held: bool,
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
    ground_plane: ground_plane::GroundPlane,
    generated_objects: Vec<ground_plane::GroundPlane>,
    next_generated_id: u64,
    generate_kind: ground_plane::BasicShape,
    displayed_grid_spacing: f32,
    displayed_grid_extent: f32,
    initial_instances: Vec<Instance>,

    current_view: Option<ViewDirection>,
    fps_sample_started: Instant,
    fps_frames: u32,
    displayed_fps: f64,
    undo_stack: Vec<UndoSnapshot>,
    undo_last_snapshot: Option<UndoSnapshot>,
    undo_transaction_active: bool,
}

impl State {
    fn capture_generated(object: &ground_plane::GroundPlane) -> GeneratedGeometryUndo {
        GeneratedGeometryUndo {
            scene_id: object.scene_id,
            generated: object.generated,
            selected: object.selected,
            snap_bottom_to_grid: object.snap_bottom_to_grid,
            kind: object.kind,
            visible: object.visible,
            wireframe: object.wireframe,
            hide_surface: object.hide_surface,
            size: object.size,
            height: object.height,
            position_x: object.position_x,
            position_y: object.position_y,
            rotation_degrees: object.rotation_degrees,
            shape_height: object.shape_height,
            thickness: object.thickness,
            subdivisions: object.subdivisions,
            triangulate_subdivision: object.triangulate_subdivision,
            uv_scale: object.uv_scale,
            material: object.material.uniform,
            base_color_path: object.base_color_path.clone(),
            normal_path: object.normal_path.clone(),
            roughness_path: object.roughness_path.clone(),
            emissive_path: object.emissive_path.clone(),
        }
    }

    fn capture_undo_snapshot(&self) -> UndoSnapshot {
        UndoSnapshot {
            deconstruction: self.deconstruction.clone(),
            instances: self.instances.clone(),
            gizmo_at_bottom_by_instance: self.gizmo_at_bottom_by_instance.clone(),
            gizmo_always_at_bottom: self.gizmo_always_at_bottom,
            lighting: self.lighting.clone(),
            generated_objects: self
                .generated_objects
                .iter()
                .chain(std::iter::once(&self.ground_plane))
                .filter(|object| object.generated)
                .map(Self::capture_generated)
                .collect(),
            pbr_material: self.pbr_material.uniform,
            model_view_mode: self.model_view_mode,
            show_points: self.show_points,
            point_size: self.point_size,
            point_color: self.point_color,
            show_normals: self.show_normals,
            normal_mode: self.normal_mode,
            normal_length: self.normal_length,
            normal_color: self.normal_color,
            geo_group_enabled: self.geo_group_enabled.clone(),
            geo_texture_enabled: self.geo_texture_enabled.clone(),
            geo_texture_scales: self.geo_texture_scales.clone(),
            geo_texture_colors: self.geo_texture_colors.clone(),
            geo_normal_strengths: self.geo_normal_strengths.clone(),
            geo_color_settings: self.geo_color_settings.clone(),
        }
    }

    fn restore_undo_snapshot(&mut self, snapshot: UndoSnapshot) {
        self.deconstruction = snapshot.deconstruction;
        self.instances = snapshot.instances;
        self.gizmo_at_bottom_by_instance = snapshot.gizmo_at_bottom_by_instance;
        self.gizmo_always_at_bottom = snapshot.gizmo_always_at_bottom;
        self.lighting = snapshot.lighting;
        self.pbr_material.uniform = snapshot.pbr_material;
        self.geo_texture_scales = snapshot.geo_texture_scales;
        self.geo_texture_colors = snapshot.geo_texture_colors;
        self.geo_normal_strengths = snapshot.geo_normal_strengths;
        self.geo_color_settings = snapshot.geo_color_settings;
        self.pbr_material.upload(&self.queue);
        for entry in &self.material_library {
            entry.material.upload(&self.queue);
        }
        for (tile, buffer) in &self.geo_material_uniform_buffers {
            let mut uniform = self.pbr_material.uniform;
            uniform.inspection[2] = self
                .geo_texture_scales
                .get(tile)
                .copied()
                .unwrap_or(1.0)
                .clamp(0.05, 100.0);
            uniform.base_color = self
                .geo_texture_colors
                .get(tile)
                .copied()
                .unwrap_or(self.pbr_material.uniform.base_color);
            uniform.properties[2] = self
                .geo_normal_strengths
                .get(tile)
                .copied()
                .unwrap_or(self.pbr_material.uniform.properties[2])
                .clamp(0.0, 2.0);
            self.apply_group_colors(*tile, &mut uniform);
            self.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
        }
        self.model_view_mode = snapshot.model_view_mode;
        self.show_points = snapshot.show_points;
        self.point_size = snapshot.point_size;
        self.point_color = snapshot.point_color;
        self.show_normals = snapshot.show_normals;
        self.normal_mode = snapshot.normal_mode;
        self.normal_length = snapshot.normal_length;
        self.normal_color = snapshot.normal_color;
        self.geo_group_enabled = snapshot.geo_group_enabled;
        self.geo_texture_enabled = snapshot.geo_texture_enabled;

        let Ok(placeholder) = ground_plane::GroundPlane::new(&self.device, &self.queue) else {
            self.editor.status = "Could not restore generated-object undo state".to_owned();
            return;
        };
        let active = std::mem::replace(&mut self.ground_plane, placeholder);
        let mut pool = std::mem::take(&mut self.generated_objects);
        pool.push(active);
        let mut restored_active = None;
        for generated in snapshot.generated_objects {
            let mut object = pool
                .iter()
                .position(|object| object.scene_id == generated.scene_id)
                .map(|index| pool.swap_remove(index))
                .or_else(|| ground_plane::GroundPlane::new(&self.device, &self.queue).ok());
            let Some(mut object) = object.take() else {
                continue;
            };
            object.scene_id = generated.scene_id;
            object.generated = generated.generated;
            object.selected = generated.selected;
            object.snap_bottom_to_grid = generated.snap_bottom_to_grid;
            object.kind = generated.kind;
            object.visible = generated.visible;
            object.wireframe = generated.wireframe;
            object.hide_surface = generated.hide_surface;
            object.size = generated.size;
            object.height = generated.height;
            object.position_x = generated.position_x;
            object.position_y = generated.position_y;
            object.rotation_degrees = generated.rotation_degrees;
            object.shape_height = generated.shape_height;
            object.thickness = generated.thickness;
            object.subdivisions = generated.subdivisions;
            object.triangulate_subdivision = generated.triangulate_subdivision;
            object.uv_scale = generated.uv_scale;
            object.material.uniform = generated.material;
            object.base_color_path = generated.base_color_path;
            object.normal_path = generated.normal_path;
            object.roughness_path = generated.roughness_path;
            object.emissive_path = generated.emissive_path;
            let mesh_key = (
                object.kind,
                object.subdivisions,
                object.triangulate_subdivision,
            );
            if let Some(mesh) = self.generated_mesh_cache.get(&mesh_key) {
                object.use_shared_mesh(mesh);
            } else {
                object.rebuild_shape(&self.device);
                self.generated_mesh_cache
                    .insert(mesh_key, object.shared_mesh());
            }
            object.upload(&self.queue);
            if object.selected && restored_active.is_none() {
                restored_active = Some(object);
            } else {
                object.selected = false;
                self.generated_objects.push(object);
            }
        }
        if let Some(active) = restored_active {
            self.ground_plane = active;
        }
        self.next_generated_id = self
            .generated_objects
            .iter()
            .chain(std::iter::once(&self.ground_plane))
            .map(|object| object.scene_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.instance_buffer_dirty = true;
        self.apply_user_preferences();
        self.editor.status = "Undid last action".to_owned();
        self.undo_transaction_active = false;
        self.undo_last_snapshot = Some(self.capture_undo_snapshot());
        self.window.request_redraw();
    }

    fn undo_last_action(&mut self) {
        if let Some(snapshot) = self.undo_stack.pop() {
            self.restore_undo_snapshot(snapshot);
        } else {
            self.editor.status = "Nothing to undo".to_owned();
        }
    }

    fn record_undo_state(&mut self, pointer_down: bool) {
        let current = self.capture_undo_snapshot();
        if let Some(previous) = self.undo_last_snapshot.take()
            && current != previous
        {
            if !self.undo_transaction_active {
                self.undo_stack.push(previous);
                if self.undo_stack.len() > 100 {
                    self.undo_stack.remove(0);
                }
            }
            if pointer_down {
                self.undo_transaction_active = true;
            }
        }
        if !pointer_down {
            self.undo_transaction_active = false;
        }
        self.undo_last_snapshot = Some(current);
    }

    async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();

        let instances = Vec::<Instance>::new();

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

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (std::mem::size_of::<InstanceRaw>() * 100) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
        let mut wireframe_material = material::PbrMaterial::new_untextured(&device, &queue)?;
        wireframe_material.uniform.base_color = [1.0, 0.32, 0.04, 1.0];
        wireframe_material.uniform.properties[0] = 0.0;
        wireframe_material.uniform.properties[1] = 1.0;
        wireframe_material.uniform.options[1] = 1.5;
        wireframe_material.uniform.options[2] = 0.0;
        wireframe_material.upload(&queue);
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
        let default_library_preview = material_preview::MaterialPreview::new(
            &device,
            &mut egui.renderer,
            &pbr_material.layout,
            &environment_lighting_layout,
        );
        let material_previews =
            HashMap::from([(material_library::MaterialId(1), default_library_preview)]);
        let material_library = vec![material_library::SceneMaterial {
            id: material_library::MaterialId(1),
            name: "Default Material".to_owned(),
            material: material::PbrMaterial::new_instance_from(&device, &pbr_material),
            base_color_path: String::new(),
            normal_path: String::new(),
            roughness_path: String::new(),
            emissive_path: String::new(),
        }];
        let lighting = lighting::LightingManager::default();
        let shadow_renderer = lighting::shadow::ShadowRenderer::new(&device, &pbr_material.layout);
        let lighting_gpu = lighting::LightingGpu::new(&device, &shadow_renderer);

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
        let grid_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let grid_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_camera_buffer.as_entire_binding(),
            }],
            label: Some("grid_camera_bind_group"),
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
            source: wgpu::ShaderSource::Wgsl(concat!(include_str!("material_alpha.wgsl"), "\n", include_str!("shader.wgsl")).into()),
        });

        let create_model_pipeline =
            |label, topology, polygon_mode, cull_mode, depth_write, depth_compare| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&render_pipeline_layout),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(if label == "Transparent model pipeline" { "fs_transparent" } else { "fs_main" }),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: config.format,
                            blend: Some(if label == "Transparent model pipeline" { wgpu::BlendState::ALPHA_BLENDING } else { wgpu::BlendState::REPLACE }),
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
                        depth_write_enabled: Some(depth_write),
                        depth_compare: Some(depth_compare),
                        stencil: wgpu::StencilState::default(),
                        // Pull overlays slightly toward the camera so coplanar wire
                        // edges remain visible over the shaded surface.
                        bias: if label == "Transparent model pipeline" || depth_write || topology != wgpu::PrimitiveTopology::TriangleList {
                            wgpu::DepthBiasState::default()
                        } else {
                            wgpu::DepthBiasState {
                                constant: -1,
                                slope_scale: -1.0,
                                clamp: 0.0,
                            }
                        },
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
            None,
            true,
            wgpu::CompareFunction::Less,
        );
        let wireframe_pipeline = wireframe_supported.then(|| {
            create_model_pipeline(
                "Wireframe model pipeline",
                wgpu::PrimitiveTopology::TriangleList,
                wgpu::PolygonMode::Line,
                None,
                false,
                wgpu::CompareFunction::LessEqual,
            )
        });
        let quad_line_pipeline = create_model_pipeline(
            "Generated quad edge pipeline",
            wgpu::PrimitiveTopology::LineList,
            wgpu::PolygonMode::Fill,
            None,
            false,
            wgpu::CompareFunction::LessEqual,
        );

        let transparent_pipeline = create_model_pipeline(
            "Transparent model pipeline", wgpu::PrimitiveTopology::TriangleList,
            wgpu::PolygonMode::Fill, None, false, wgpu::CompareFunction::LessEqual,
        );

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

        let obj_model = model::Model {
            import_warnings: Vec::new(),
            imported_scene: None,
            meshes: Vec::new(),
            materials: Vec::new(),
        };

        //----------------------------------------//

        let grid = grid::Grid::new(
            &device,
            config.format,
            &camera_bind_group_layout,
            editor.grid_size,
            editor.grid_spacing,
            editor.grid_color,
        );
        let ground_plane = ground_plane::GroundPlane::new(&device, &queue)?;

        //----------------------------------------//
        let initial_instances = instances.clone();
        let displayed_grid_spacing = editor.grid_spacing;
        let displayed_grid_extent = editor.grid_size;
        let group_count = obj_model.meshes.len();
        #[cfg(not(target_arch = "wasm32"))]
        let preference_store = user_data::PreferenceStore::load();
        let mut state = Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            transparent_pipeline,
            window,
            pbr_material,
            wireframe_material,
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
            uv_import_kind: GroupTextureKind::BaseColor,
            show_all_uv_spaces: false,
            uv_space_offsets: BTreeMap::new(),
            uv_space_cache: Vec::new(),
            uv_inspector_mode: UvInspectorMode::Wireframe,
            uv_space_textures: BTreeMap::new(),
            geo_group_enabled: vec![true; group_count],
            geo_texture_enabled: vec![true; group_count],
            geo_texture_paths: BTreeMap::new(),
            geo_normal_paths: BTreeMap::new(),
            geo_roughness_paths: BTreeMap::new(),
            geo_emissive_paths: BTreeMap::new(),
            geo_normal_textures: BTreeMap::new(),
            geo_roughness_textures: BTreeMap::new(),
            geo_emissive_textures: BTreeMap::new(),
            geo_material_bind_groups: BTreeMap::new(),
            geo_material_uniform_buffers: BTreeMap::new(),
            geo_texture_scales: BTreeMap::new(),
            geo_texture_colors: BTreeMap::new(),
            geo_normal_strengths: BTreeMap::new(),
            geo_color_settings: BTreeMap::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pending_uv_textures: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            pending_model_material_root: None,
            pending_project: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_model_load: None,
            project_path: String::new(),
            fx_load_started: None,
            material_texture_cache: HashMap::new(),
            generated_mesh_cache: HashMap::new(),
            show_uv_texture: true,
            show_uv_lines: true,
            #[cfg(not(target_arch = "wasm32"))]
            preferences: preference_store.saved.clone(),
            #[cfg(target_arch = "wasm32")]
            preferences: user_data::Preferences::default(),
            #[cfg(not(target_arch = "wasm32"))]
            preference_store,
            #[cfg(not(target_arch = "wasm32"))]
            user_library: user_data::UserLibrary::load(),
            viewport_settings_open: false,
            viewport_settings_just_opened: false,
            gizmo_at_bottom_by_instance: vec![false; instances.len()],
            gizmo_always_at_bottom: false,
            wireframe_pipeline,
            quad_line_pipeline,
            model_view_mode: ModelViewMode::Solid,
            material_graph: material_graph::MaterialGraphEditor::default(),
            material_preview,
            material_previews,
            material_library,
            next_material_id: 2,
            mesh_material_assignments: Vec::new(),
            face_material_assignments: Vec::new(),
            generated_material_assignments: BTreeMap::new(),
            material_browser_search: String::new(),
            selected_library_material: material_library::MaterialId(1),
            dragged_material: None,
            face_assign_first: 0,
            face_assign_end: 1,
            selected_material_mesh: 0,
            part_selection: Default::default(),
            lighting,
            shadow_renderer,
            lighting_gpu,
            deconstruction: demos::Deconstruction::default(),
            demo_buffers: Vec::new(),
            demo_cached: None,
            active_side_panel: Some(0),
            explorer_open: false,
            #[cfg(not(target_arch = "wasm32"))]
            asset_explorer: asset_explorer::AssetExplorer::default(),
            generate_popup_open: false,
            generate_popup_position: egui::pos2(320.0, 180.0),
            outliner_search: String::new(),
            environment_lighting_layout,
            environment_lighting_bind_group,
            _fallback_environment_texture,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            grid_camera_buffer,
            grid_camera_bind_group,
            houdini_navigation: HoudiniNavigation::default(),
            undo_modifier_held: false,
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
            ground_plane,
            generated_objects: Vec::new(),
            next_generated_id: 1,
            generate_kind: ground_plane::BasicShape::Plane,
            displayed_grid_spacing,
            displayed_grid_extent,
            initial_instances,

            current_view: None,
            fps_sample_started: Instant::now(),
            fps_frames: 0,
            displayed_fps: 0.0,
            undo_stack: Vec::new(),
            undo_last_snapshot: None,
            undo_transaction_active: false,
        };
        state.apply_user_preferences();
        Ok(state)
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
        self.sync_light_targets();
        self.update_adaptive_grid();
        self.update_camera_clip_planes();
        self.update_grid_camera();
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
        for entry in &self.material_library {
            entry.material.upload(&self.queue);
        }
        self.ground_plane.upload(&self.queue);
        for object in &mut self.generated_objects {
            object.upload(&self.queue);
        }
        self.shadow_renderer
            .prepare(&self.queue, &self.lighting, &self.camera);
        self.lighting_gpu.upload(
            &self.queue,
            &self.lighting,
            &self.camera,
            &self.shadow_renderer,
        );
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

    fn sync_light_targets(&mut self) {
        let target_centers = self
            .instances
            .iter()
            .take(self.editor.visible_instance_count)
            .map(|instance| {
                self.instance_world_bounds(instance)
                    .map(|(minimum, maximum)| (minimum + maximum) * 0.5)
                    .unwrap_or(instance.position)
            })
            .collect::<Vec<_>>();
        let generated_target_centers = self
            .generated_objects
            .iter()
            .chain(std::iter::once(&self.ground_plane))
            .filter(|object| object.generated)
            .map(|object| {
                (
                    object.scene_id,
                    cgmath::Vector3::new(object.position_x, object.position_y, object.height),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut changed = false;
        for light in &mut self.lighting.lights {
            let Some(target_index) = light.target_instance else {
                continue;
            };
            let target = if let Some(scene_id) = Self::generated_target_id(target_index) {
                generated_target_centers.get(&scene_id).copied()
            } else {
                target_centers.get(target_index).copied()
            };
            let Some(target) = target else {
                light.target_instance = None;
                changed = true;
                continue;
            };
            let position = cgmath::Vector3::from(light.position);
            let offset = target - position;
            if offset.magnitude2() <= f32::EPSILON {
                continue;
            }
            let direction = offset.normalize();
            let pitch = (-direction.z).clamp(-1.0, 1.0).acos();
            let yaw = (-direction.x).atan2(direction.y);
            let rotation = [pitch.to_degrees(), 0.0, yaw.to_degrees()];
            if light.rotation_degrees != rotation {
                light.rotation_degrees = rotation;
                changed = true;
            }
        }
        if changed {
            self.lighting.touch();
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
        let mut bounds = self.instances.iter().take(self.editor.visible_instance_count)
            .filter_map(|instance| self.instance_world_bounds(instance));
        let Some((mut minimum, mut maximum)) = bounds.next() else {
            self.home_grid_view();
            return;
        };
        for (low, high) in bounds {
            minimum.x = minimum.x.min(low.x);
            minimum.y = minimum.y.min(low.y);
            minimum.z = minimum.z.min(low.z);
            maximum.x = maximum.x.max(high.x);
            maximum.y = maximum.y.max(high.y);
            maximum.z = maximum.z.max(high.z);
        }
        let center = (minimum + maximum) * 0.5;
        let radius = ((maximum - minimum).magnitude() * 0.5).max(0.01);
        self.frame_point(cgmath::Point3::from_vec(center), radius);
    }

    fn frame_selected_instance(&mut self) {
        let Some((minimum, maximum)) = self.instances.get(self.editor.selected_instance)
            .and_then(|instance| self.instance_world_bounds(instance)) else { return; };
        self.frame_point(cgmath::Point3::from_vec((minimum + maximum) * 0.5), ((maximum - minimum).magnitude() * 0.5).max(0.01));
    }

    fn frame_point(&mut self, center: cgmath::Point3<f32>, radius: f32) {
        let view_direction = (self.camera.eye - self.camera.target).normalize();
        self.camera.target = center;
        match self.camera.projection_mode {
            ProjectionMode::Perspective => {
                let vertical_half = (self.camera.fovy.to_radians() * 0.5).max(0.01);
                let horizontal_half = (vertical_half.tan() * self.camera.aspect.max(0.01)).atan();
                let half_fov = vertical_half.min(horizontal_half);
                let distance = (radius * 1.15 / half_fov.sin()).max(radius * 2.0);
                self.camera.eye = center + view_direction * distance;
            }
            ProjectionMode::Orthographic => {
                self.camera.ortho_scale = (radius * 2.4 / self.camera.aspect.min(1.0).max(0.01)).max(0.01);
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

    fn model_local_bounds(&self) -> Option<(cgmath::Vector3<f32>, cgmath::Vector3<f32>)> {
        let first = self.obj_model.meshes.first()?;
        let mut minimum = cgmath::Vector3::from(first.bounds_min);
        let mut maximum = cgmath::Vector3::from(first.bounds_max);
        for mesh in &self.obj_model.meshes[1..] {
            let mesh_min = cgmath::Vector3::from(mesh.bounds_min);
            let mesh_max = cgmath::Vector3::from(mesh.bounds_max);
            minimum.x = minimum.x.min(mesh_min.x);
            minimum.y = minimum.y.min(mesh_min.y);
            minimum.z = minimum.z.min(mesh_min.z);
            maximum.x = maximum.x.max(mesh_max.x);
            maximum.y = maximum.y.max(mesh_max.y);
            maximum.z = maximum.z.max(mesh_max.z);
        }
        Some((minimum, maximum))
    }

    fn instance_world_bounds(
        &self,
        instance: &Instance,
    ) -> Option<(cgmath::Vector3<f32>, cgmath::Vector3<f32>)> {
        if self.obj_model.meshes.is_empty() { return None; }
        let mut world_min = cgmath::Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut world_max = cgmath::Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for (group, mesh) in self.obj_model.meshes.iter().enumerate() {
            let model: cgmath::Matrix4<f32> = self.demo_instance_raw(instance, group).model.into();
            for x in [mesh.bounds_min[0], mesh.bounds_max[0]] {
                for y in [mesh.bounds_min[1], mesh.bounds_max[1]] {
                    for z in [mesh.bounds_min[2], mesh.bounds_max[2]] {
                        let point = model.transform_point(cgmath::Point3::new(x, y, z)).to_vec();
                        world_min.x = world_min.x.min(point.x);
                        world_min.y = world_min.y.min(point.y);
                        world_min.z = world_min.z.min(point.z);
                        world_max.x = world_max.x.max(point.x);
                        world_max.y = world_max.y.max(point.y);
                        world_max.z = world_max.z.max(point.z);
                    }
                }
            }
        }
        Some((world_min, world_max))
    }

    fn update_camera_clip_planes(&mut self) {
        let view = self.camera.target - self.camera.eye;
        if view.magnitude2() <= f32::EPSILON {
            return;
        }
        let forward = view.normalize();
        let eye = self.camera.eye.to_vec();
        let mut nearest = f32::INFINITY;
        let mut farthest = 0.0_f32;

        let mut include_bounds = |minimum: cgmath::Vector3<f32>, maximum: cgmath::Vector3<f32>| {
            let mut bounds_near = f32::INFINITY;
            let mut bounds_far = f32::NEG_INFINITY;
            for x in [minimum.x, maximum.x] {
                for y in [minimum.y, maximum.y] {
                    for z in [minimum.z, maximum.z] {
                        let depth = (cgmath::Vector3::new(x, y, z) - eye).dot(forward);
                        bounds_near = bounds_near.min(depth);
                        bounds_far = bounds_far.max(depth);
                    }
                }
            }
            if bounds_far > 0.0 {
                farthest = farthest.max(bounds_far);
                nearest = if bounds_near <= 0.0 {
                    nearest.min(0.001)
                } else {
                    nearest.min(bounds_near)
                };
            }
        };

        for instance in self
            .instances
            .iter()
            .take(self.editor.visible_instance_count)
        {
            if let Some((minimum, maximum)) = self.instance_world_bounds(instance) {
                include_bounds(minimum, maximum);
            }
        }
        if self.ground_plane.generated && self.ground_plane.visible {
            let center = cgmath::Vector3::new(
                self.ground_plane.position_x,
                self.ground_plane.position_y,
                self.ground_plane.height,
            );
            let radius = self
                .ground_plane
                .size
                .max(self.ground_plane.shape_height)
                .max(0.1);
            include_bounds(
                center - cgmath::Vector3::new(radius, radius, radius),
                center + cgmath::Vector3::new(radius, radius, radius),
            );
        }
        for object in &self.generated_objects {
            if !object.generated || !object.visible {
                continue;
            }
            let center = cgmath::Vector3::new(object.position_x, object.position_y, object.height);
            let radius = object.size.max(object.shape_height).max(0.1);
            include_bounds(
                center - cgmath::Vector3::new(radius, radius, radius),
                center + cgmath::Vector3::new(radius, radius, radius),
            );
        }
        let camera_distance = view.magnitude().max(0.01);
        if !nearest.is_finite() || farthest <= 0.0 {
            nearest = camera_distance * 0.5;
            farthest = camera_distance * 2.0 + 10.0;
        }
        let far = (farthest * 1.5).max(camera_distance + 10.0).max(10.0);
        // Keep the useful depth range within five orders of magnitude. This
        // prevents distant coplanar layers from collapsing to the same value.
        let near = (nearest * 0.25).max(far / 100_000.0).max(0.001);
        self.camera.znear = near.min(far * 0.5);
        self.camera.zfar = far;
    }

    fn update_grid_camera(&self) {
        let mut grid_camera = self.camera.clone();
        let distance = (grid_camera.eye - grid_camera.target).magnitude().max(0.01);
        let extent = self
            .displayed_grid_extent
            .max(self.editor.grid_size)
            .max(1.0);
        // The grid gets its own broad clip range. Its line pipeline does not
        // write depth, so this range cannot reduce mesh depth precision.
        grid_camera.zfar = (distance + extent * 2.0).max(100.0);
        grid_camera.znear = (grid_camera.zfar / 10_000_000.0).max(0.0001);
        let mut uniform = CameraUniform::new();
        uniform.update_view_proj(&grid_camera);
        self.queue.write_buffer(
            &self.grid_camera_buffer,
            0,
            bytemuck::cast_slice(&[uniform]),
        );
    }

    fn selected_gizmo_pivot(&self) -> Option<cgmath::Vector3<f32>> {
        let instance = self.instances.get(self.editor.selected_instance)?;
        let (minimum, maximum) = self.instance_world_bounds(instance)?;
        let mut center = (minimum + maximum) * 0.5;
        let selected_prefers_bottom = self
            .gizmo_at_bottom_by_instance
            .get(self.editor.selected_instance)
            .copied()
            .unwrap_or(false);
        if selected_prefers_bottom || self.gizmo_always_at_bottom {
            center.z = minimum.z;
        }
        Some(center)
    }

    fn select_instance_at_cursor(&mut self) {
        let Some((cursor_x, cursor_y)) = self.houdini_navigation.cursor_position else {
            return;
        };
        let width = self.config.width.max(1) as f32;
        let height = self.config.height.max(1) as f32;
        let ndc_x = cursor_x as f32 / width * 2.0 - 1.0;
        let ndc_y = 1.0 - cursor_y as f32 / height * 2.0;
        let Some(inverse_view_projection) = self.camera.build_view_projection_matrix().invert()
        else {
            return;
        };
        let near_clip = inverse_view_projection * cgmath::Vector4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_clip = inverse_view_projection * cgmath::Vector4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near_clip.truncate() / near_clip.w;
        let far = far_clip.truncate() / far_clip.w;
        let ray = (far - near).normalize();
        let mut closest: Option<(usize, f32)> = None;
        for (index, instance) in self.instances.iter().take(self.editor.visible_instance_count).enumerate() {
            for (group, mesh) in self.obj_model.meshes.iter().enumerate() {
                if !self.geo_group_enabled.get(group).copied().unwrap_or(true) { continue; }
                let model: cgmath::Matrix4<f32> = self.demo_instance_raw(instance, group).model.into();
                let Some(inverse_model) = model.invert() else { continue; };
                let local_origin = inverse_model.transform_point(cgmath::Point3::from_vec(near)).to_vec();
                let local_direction = inverse_model.transform_vector(ray);
                if let Some(distance) = ray_box_distance(local_origin, local_direction, mesh.bounds_min.into(), mesh.bounds_max.into()) {
                    let hit = model.transform_point(cgmath::Point3::from_vec(local_origin + local_direction * distance)).to_vec();
                    let world_distance = (hit - near).magnitude2();
                    if closest.is_none_or(|(_, d)| world_distance < d) { closest = Some((index, world_distance)); }
                }
            }
        }
        let mut stored_generated_hit = None;
        let mut stored_generated_distance = f32::INFINITY;
        for (stored_index, object) in self.generated_objects.iter().enumerate() {
            if !object.generated || !object.visible {
                continue;
            }
            let model = object.model_matrix();
            let Some(inverse_model) = model.invert() else {
                continue;
            };
            let local_origin = inverse_model
                .transform_point(cgmath::Point3::from_vec(near))
                .to_vec();
            let local_direction = inverse_model.transform_vector(ray);
            let half_z = 0.5;
            if let Some(distance) = ray_box_distance(
                local_origin,
                local_direction,
                cgmath::Vector3::new(-0.5, -0.5, -half_z),
                cgmath::Vector3::new(0.5, 0.5, half_z),
            ) && distance < stored_generated_distance
            {
                stored_generated_distance = distance;
                stored_generated_hit = Some(stored_index);
            }
        }
        if let Some(index) = stored_generated_hit {
            self.select_stored_generated(index);
            return;
        }
        if self.ground_plane.generated && self.ground_plane.visible {
            let model = self.ground_plane.model_matrix();
            if let Some(inverse_model) = model.invert() {
                let local_origin = inverse_model
                    .transform_point(cgmath::Point3::from_vec(near))
                    .to_vec();
                let local_direction = inverse_model.transform_vector(ray);
                let half_z = 0.5;
                if let Some(local_distance) = ray_box_distance(
                    local_origin,
                    local_direction,
                    cgmath::Vector3::new(-0.5, -0.5, -half_z),
                    cgmath::Vector3::new(0.5, 0.5, half_z),
                ) {
                    let local_hit = local_origin + local_direction * local_distance;
                    let world_hit = model
                        .transform_point(cgmath::Point3::from_vec(local_hit))
                        .to_vec();
                    let world_distance = (world_hit - near).magnitude2();
                    if closest.is_none_or(|(_, distance)| world_distance < distance) {
                        self.ground_plane.selected = true;
                        self.lighting.selected_light = None;
                        self.editor.status =
                            format!("Selected generated {}", self.ground_plane.kind.name());
                        return;
                    }
                }
            }
        }
        if let Some((index, _)) = closest {
            self.editor.selected_instance = index;
            self.lighting.selected_light = None;
            self.ground_plane.selected = false;
            self.editor.status = format!("Selected instance {} in viewport", index + 1);
        }
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
        if self.ground_plane.generated && self.ground_plane.selected {
            let stats = self.ground_plane.geometry_stats();
            let dimensions = [
                stats.bounds_max[0] - stats.bounds_min[0],
                stats.bounds_max[1] - stats.bounds_min[1],
                stats.bounds_max[2] - stats.bounds_min[2],
            ];
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
                            ui.label(format!(
                                "Asset: {} {}",
                                self.ground_plane.kind.name(),
                                self.ground_plane.scene_id
                            ));
                            ui.label("Meshes: 1");
                            ui.label(format!("Points: {}", stats.points));
                            ui.label(format!("Vertices: {}", stats.vertices));
                            ui.label(format!("Lines / edges: {}", stats.edges));
                            ui.label(format!("Triangles: {}", stats.triangles));
                            ui.label(format!("Indices: {}", stats.indices));
                            ui.label(format!(
                                "Bounds min: {:.3}, {:.3}, {:.3}",
                                stats.bounds_min[0], stats.bounds_min[1], stats.bounds_min[2]
                            ));
                            ui.label(format!(
                                "Bounds max: {:.3}, {:.3}, {:.3}",
                                stats.bounds_max[0], stats.bounds_max[1], stats.bounds_max[2]
                            ));
                            ui.label(format!(
                                "Dimensions: {:.3} × {:.3} × {:.3}",
                                dimensions[0], dimensions[1], dimensions[2]
                            ));
                            ui.label(format!(
                                "Geometry buffers: {:.2} KiB",
                                (stats.vertices * std::mem::size_of::<model::ModelVertex>()
                                    + stats.indices * std::mem::size_of::<u32>())
                                    as f64
                                    / 1024.0
                            ));
                        });
                });
            return;
        }
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
        self.asset_explorer_window(&context);
        self.viewport_generate_popup(&context);
        let panel_max_height = (context.content_rect().height() - 24.0).max(120.0);
        let imported_object_controls = !self.ground_plane.selected && !self.instances.is_empty();
        let object_panel_width = if imported_object_controls {
            (context.content_rect().width() * 0.38).clamp(440.0, 680.0)
        } else {
            380.0
        };

        if self.active_side_panel == Some(0) {
            egui::Window::new("Object")
                // Version the id when the sizing policy changes so an old forced
                // height cannot override the new content-driven initial layout.
                .id(egui::Id::new("viewport_editor_window_scrollable_v6"))
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
                .default_width(object_panel_width)
                .default_height(panel_max_height)
                .min_width(320.0)
                .min_height(160.0)
                .max_width(object_panel_width)
                .max_height(panel_max_height)
                .resizable(true)
                .collapsible(true)
                .constrain(true)
                .vscroll(true)
                .show(ui, |ui| {
                    ui.set_width(object_panel_width - 24.0);
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                    egui::CollapsingHeader::new("Project & Model Import")
                        .default_open(true)
                        .show(ui, |ui| self.asset_drop_ui(ui));
                    ui.separator();
                    self.generate_ground_plane_ui(ui);
                    if !self.ground_plane.selected && !self.instances.is_empty() {
                        ui.separator();
                        ui.heading("Selected Imported Object");
                        if ui.button("Reset Transform").clicked() {
                            self.reset_model_transforms();
                        }
                        self.transform_ui(ui);
                        ui.separator();
                        self.texture_ui(ui);
                        ui.separator();
                        self.imported_texture_groups_ui(ui);
                    } else if !self.ground_plane.selected {
                        ui.separator();
                        ui.label("Select an object to edit its controls.");
                    }
                    ui.separator();
                    ui.add(egui::Label::new(format!("Status: {}", self.editor.status)).wrap());
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
        lighting::gizmo::show(
            &context,
            &mut self.lighting,
            &self.camera,
            self.houdini_navigation.view_mode_active(),
        );
        let light_gizmo_active = self.lighting.mode == lighting::ViewportLightingMode::SceneLights
            && self.lighting.selected_light.is_some_and(|selected_id| {
                self.lighting
                    .lights
                    .iter()
                    .any(|light| light.id == selected_id && light.viewport_enabled)
            });
        if light_gizmo_active {
            self.ground_plane.selected = false;
        } else if self.ground_plane.generated && self.ground_plane.selected {
            let pivot = cgmath::Vector3::new(
                self.ground_plane.position_x,
                self.ground_plane.position_y,
                self.ground_plane.height,
            );
            if let Some(change) = object_gizmo::show(
                &context,
                &self.camera,
                pivot,
                self.houdini_navigation.view_mode_active(),
                "generated_shape",
            ) {
                if change.translation.z.abs() > f32::EPSILON {
                    self.ground_plane.snap_bottom_to_grid = false;
                }
                self.ground_plane.position_x += change.translation.x;
                self.ground_plane.position_y += change.translation.y;
                self.ground_plane.height += change.translation.z;
                for axis in 0..3 {
                    self.ground_plane.rotation_degrees[axis] = (self.ground_plane.rotation_degrees
                        [axis]
                        + change.rotation_degrees[axis]
                        + 180.0)
                        .rem_euclid(360.0)
                        - 180.0;
                }
                self.editor.status =
                    format!("Transformed generated {}", self.ground_plane.kind.name());
            }
        } else if let Some(pivot) = self.selected_gizmo_pivot().filter(|_| !self.part_selection.active) {
            if let Some(change) = object_gizmo::show(
                &context,
                &self.camera,
                pivot,
                self.houdini_navigation.view_mode_active(),
                "selected_object",
            ) {
                if let Some(instance) = self.instances.get_mut(self.editor.selected_instance) {
                    instance.position += change.translation;
                    instance.rotation_degrees.x =
                        (instance.rotation_degrees.x + change.rotation_degrees.x + 180.0)
                            .rem_euclid(360.0)
                            - 180.0;
                    instance.rotation_degrees.y =
                        (instance.rotation_degrees.y + change.rotation_degrees.y + 180.0)
                            .rem_euclid(360.0)
                            - 180.0;
                    instance.rotation_degrees.z =
                        (instance.rotation_degrees.z + change.rotation_degrees.z + 180.0)
                            .rem_euclid(360.0)
                            - 180.0;
                    self.instance_buffer_dirty = true;
                    self.editor.status = if change.rotation_degrees.magnitude2() > f32::EPSILON {
                        format!(
                            "Rotated instance {} with viewport gizmo",
                            self.editor.selected_instance + 1
                        )
                    } else {
                        format!(
                            "Moved instance {} with viewport gizmo",
                            self.editor.selected_instance + 1
                        )
                    };
                }
            }
        }
        self.grid_measurement_labels_ui(&context);
        self.viewport_settings_window(&context);
        self.geometry_inspection_window(&context);
        self.lighting_window(&context);
        self.part_selection_window(&context);
        self.material_browser_window(&context);
        self.demos_window(&context);
        #[cfg(not(target_arch = "wasm32"))]
        self.apply_pending_model_materials(&context);
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
                    (0, "Object"),
                    (1, "Geometry Inspection"),
                    (2, "Lighting"),
                    (4, "Scene Outliner"),
                    (5, "Materials"),
                    (6, "Demos"),
                ] {
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
                ui.separator();
                if ui
                    .add_sized(
                        [TAB_WIDTH, TAB_HEIGHT],
                        egui::Button::new("Explorer").selected(self.explorer_open),
                    )
                    .on_hover_text("Independent asset browser; remains open with other tabs")
                    .clicked()
                {
                    self.explorer_open = !self.explorer_open;
                }
            });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn asset_explorer_window(&mut self, context: &egui::Context) {
        let mut load = None;
        self.asset_explorer.show_with_library(context, &mut self.explorer_open, |ui| {
            load = self.user_library.ui(ui);
        });
        self.preferences.explorer_grid = self.asset_explorer.grid;
        self.preferences.explorer_hidden = self.asset_explorer.show_hidden;
        self.preferences.explorer_library = self.asset_explorer.user_library;
        if let Some(id) = load { self.add_user_material_to_scene(&id); }
    }

    #[cfg(target_arch = "wasm32")]
    fn asset_explorer_window(&mut self, _context: &egui::Context) {}

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

    fn explorer_drop(response: &egui::Response, extensions: &[&str]) -> Option<std::path::PathBuf> {
        let path = response
            .dnd_release_payload::<std::path::PathBuf>()?
            .as_ref()
            .clone();
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        extensions
            .iter()
            .any(|allowed| extension.eq_ignore_ascii_case(allowed))
            .then_some(path)
    }

    fn outliner_eye_toggle(ui: &mut egui::Ui, visible: &mut bool) -> bool {
        let (rect, response) = ui.allocate_exact_size(egui::vec2(20.0, 18.0), egui::Sense::click());
        if response.hovered() {
            ui.painter()
                .rect_filled(rect, 2.0, egui::Color32::from_white_alpha(18));
        }
        let center = rect.center();
        let stroke = egui::Stroke::new(
            1.35,
            if response.hovered() {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_gray(175)
            },
        );
        if *visible {
            let upper = [-7.0, -3.5, 0.0, 3.5, 7.0]
                .into_iter()
                .zip([0.0, -3.0, -4.0, -3.0, 0.0])
                .map(|(x, y)| center + egui::vec2(x, y))
                .collect();
            let lower = [-7.0, -3.5, 0.0, 3.5, 7.0]
                .into_iter()
                .zip([0.0, 3.0, 4.0, 3.0, 0.0])
                .map(|(x, y)| center + egui::vec2(x, y))
                .collect();
            ui.painter().add(egui::Shape::line(upper, stroke));
            ui.painter().add(egui::Shape::line(lower, stroke));
            ui.painter().circle_filled(center, 2.25, stroke.color);
        } else {
            let lid = [-7.0, -3.5, 0.0, 3.5, 7.0]
                .into_iter()
                .zip([-1.0, 1.5, 2.5, 1.5, -1.0])
                .map(|(x, y)| center + egui::vec2(x, y))
                .collect();
            ui.painter().add(egui::Shape::line(lid, stroke));
            for x in [-4.5, 0.0, 4.5] {
                ui.painter().line_segment(
                    [center + egui::vec2(x, 2.0), center + egui::vec2(x, 4.5)],
                    stroke,
                );
            }
        }
        if response.clicked() {
            *visible = !*visible;
            true
        } else {
            false
        }
    }

    fn select_generated_shape(&mut self, kind: ground_plane::BasicShape) {
        let Ok(mut object) = ground_plane::GroundPlane::new(&self.device, &self.queue) else {
            self.editor.status = "Could not allocate generated object".to_owned();
            return;
        };
        object.scene_id = self.next_generated_id;
        self.next_generated_id = self.next_generated_id.saturating_add(1);
        object.kind = kind;
        object.generated = true;
        object.visible = true;
        object.selected = true;
        object.triangulate_subdivision = self.ground_plane.triangulate_subdivision;
        object.rebuild_shape(&self.device);
        if self.ground_plane.generated {
            self.ground_plane.selected = false;
            let previous = std::mem::replace(&mut self.ground_plane, object);
            self.generated_objects.push(previous);
        } else {
            self.ground_plane = object;
        }
        self.lighting.selected_light = None;
        self.editor.status = format!(
            "{} {} added to scene",
            kind.name(),
            self.ground_plane.scene_id
        );
    }

    fn select_stored_generated(&mut self, index: usize) {
        if index >= self.generated_objects.len() {
            return;
        }
        if self.ground_plane.generated {
            self.ground_plane.selected = false;
            let stored = &mut self.generated_objects[index];
            std::mem::swap(&mut self.ground_plane, stored);
            stored.selected = false;
        } else {
            self.ground_plane = self.generated_objects.swap_remove(index);
        }
        self.ground_plane.selected = true;
        self.lighting.selected_light = None;
        self.editor.status = format!(
            "Selected generated {} {}",
            self.ground_plane.kind.name(),
            self.ground_plane.scene_id
        );
    }

    fn generated_target_index(scene_id: u64) -> usize {
        GENERATED_SHAPE_LIGHT_TARGET_BASE.saturating_add(scene_id as usize)
    }

    fn generated_target_id(target: usize) -> Option<u64> {
        target
            .checked_sub(GENERATED_SHAPE_LIGHT_TARGET_BASE)
            .map(|scene_id| scene_id as u64)
    }

    fn generated_object_count(&self) -> usize {
        self.generated_objects.len() + usize::from(self.ground_plane.generated)
    }

    fn viewport_generate_popup(&mut self, context: &egui::Context) {
        if !self.generate_popup_open {
            return;
        }

        let mut open = true;
        let mut selected = None;
        egui::Window::new("Generate")
            .id(egui::Id::new("viewport_generate_popup"))
            .fixed_pos(self.generate_popup_position)
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(context, |ui| {
                egui::CollapsingHeader::new("Geometry")
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::CollapsingHeader::new("Basic Shapes")
                            .default_open(true)
                            .show(ui, |ui| {
                                for kind in ground_plane::BasicShape::ALL {
                                    if ui
                                        .add_sized([190.0, 26.0], egui::Button::new(kind.name()))
                                        .clicked()
                                    {
                                        selected = Some(kind);
                                    }
                                }
                            });
                    });
            });

        if let Some(kind) = selected {
            self.select_generated_shape(kind);
            self.generate_popup_open = false;
        } else if !open || context.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.generate_popup_open = false;
        }
    }

    fn generate_ground_plane_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Add Primitive");
        egui::ComboBox::from_label("Shape")
            .selected_text(self.generate_kind.name())
            .show_ui(ui, |ui| {
                for kind in ground_plane::BasicShape::ALL {
                    ui.selectable_value(&mut self.generate_kind, kind, kind.name());
                }
            });
        if ui
            .button(format!("Add New {}", self.generate_kind.name()))
            .clicked()
        {
            self.select_generated_shape(self.generate_kind);
        }
        if !self.ground_plane.generated || !self.ground_plane.selected {
            return;
        }
        ui.separator();
        ui.heading("Selected Generated Object");
        ui.strong(format!(
            "{} {}",
            self.ground_plane.kind.name(),
            self.ground_plane.scene_id
        ));
        if ui
            .button(format!(
                "Remove {} from Scene",
                self.ground_plane.kind.name()
            ))
            .clicked()
        {
            self.ground_plane.generated = false;
            self.ground_plane.selected = false;
            self.editor.status = format!("{} removed from scene", self.ground_plane.kind.name());
            return;
        }
        ui.checkbox(&mut self.ground_plane.visible, "Visible");
        if ui
            .checkbox(&mut self.ground_plane.selected, "Selected")
            .changed()
            && self.ground_plane.selected
        {
            self.lighting.selected_light = None;
        }
        if self.ground_plane.kind != ground_plane::BasicShape::Plane {
            let label = if self.ground_plane.snap_bottom_to_grid {
                "✓ Bottom Snapped to Grid"
            } else {
                "Snap Bottom to Grid"
            };
            if ui.button(label).clicked() {
                self.ground_plane.snap_bottom_to_grid = !self.ground_plane.snap_bottom_to_grid;
                self.editor.status = if self.ground_plane.snap_bottom_to_grid {
                    format!("{} bottom locked to grid", self.ground_plane.kind.name())
                } else {
                    format!(
                        "{} bottom grid lock disabled",
                        self.ground_plane.kind.name()
                    )
                };
            }
        }
        let wireframe = ui.checkbox(&mut self.ground_plane.wireframe, "Wireframe Overlay");
        if self.ground_plane.triangulate_subdivision
            && self.wireframe_pipeline.is_none()
            && wireframe.hovered()
        {
            wireframe.on_hover_text("Wireframe rendering is unavailable on this GPU");
        }
        ui.add_enabled(
            self.ground_plane.wireframe,
            egui::Checkbox::new(&mut self.ground_plane.hide_surface, "Hide Surface"),
        );
        Self::slider_with_number(ui, "Size", &mut self.ground_plane.size, 0.1..=500.0, 0.25);
        if self.ground_plane.kind == ground_plane::BasicShape::Plane {
            Self::slider_with_number(
                ui,
                "Thickness",
                &mut self.ground_plane.thickness,
                0.0..=50.0,
                0.01,
            );
            ui.small("Thickness extends downward from the plane elevation.");
        }
        let subdivisions_changed = ui
            .horizontal(|ui| {
                ui.label("Subdivisions");
                ui.add(egui::Slider::new(
                    &mut self.ground_plane.subdivisions,
                    1..=8,
                ))
                .changed()
            })
            .inner;
        if subdivisions_changed {
            self.ground_plane.rebuild_shape(&self.device);
            self.editor.status = format!(
                "{} subdivisions: {}",
                self.ground_plane.kind.name(),
                self.ground_plane.subdivisions
            );
        }
        ui.separator();
        ui.label("Position");
        Self::slider_with_number(
            ui,
            "X",
            &mut self.ground_plane.position_x,
            -50.0..=50.0,
            0.05,
        );
        Self::slider_with_number(
            ui,
            "Y",
            &mut self.ground_plane.position_y,
            -50.0..=50.0,
            0.05,
        );
        if matches!(
            self.ground_plane.kind,
            ground_plane::BasicShape::Cylinder | ground_plane::BasicShape::Cone
        ) {
            Self::slider_with_number(
                ui,
                "Shape height",
                &mut self.ground_plane.shape_height,
                0.1..=500.0,
                0.1,
            );
        }
        let elevation_changed = Self::slider_with_number(
            ui,
            "Elevation",
            &mut self.ground_plane.height,
            -50.0..=50.0,
            0.05,
        );
        if elevation_changed {
            self.ground_plane.snap_bottom_to_grid = false;
            self.editor.status = format!(
                "{} moved vertically; bottom grid lock disabled",
                self.ground_plane.kind.name()
            );
        }
        ui.label("Rotation");
        for (axis, label) in ["X", "Y", "Z"].into_iter().enumerate() {
            Self::slider_with_number(
                ui,
                label,
                &mut self.ground_plane.rotation_degrees[axis],
                -180.0..=180.0,
                0.25,
            );
        }

        ui.separator();
        ui.heading("Material");
        ui.horizontal(|ui| {
            ui.label("Base color");
            ui.color_edit_button_rgba_unmultiplied(
                &mut self.ground_plane.material.uniform.base_color,
            );
        });
        Self::slider_with_number(
            ui,
            "Roughness",
            &mut self.ground_plane.material.uniform.properties[1],
            0.0..=1.0,
            0.01,
        );
        Self::slider_with_number(
            ui,
            "Normal strength",
            &mut self.ground_plane.material.uniform.properties[2],
            0.0..=2.0,
            0.01,
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let base_texture_button = ui.button("Load Base Color Texture…");
            let base_texture_path = if base_texture_button.clicked() {
                pick_ground_texture()
            } else {
                Self::explorer_drop(
                    &base_texture_button,
                    &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                )
            };
            if let Some(path) = base_texture_path {
                match self.cached_material_texture(&path, true) {
                    Ok(texture) => {
                        self.ground_plane
                            .material
                            .set_base_color_texture(&self.device, texture);
                        self.ground_plane.base_color_path = path.display().to_string();
                    }
                    Err(error) => self.editor.status = format!("Ground texture failed: {error:#}"),
                }
            }
            if !self.ground_plane.base_color_path.is_empty() {
                ui.small(&self.ground_plane.base_color_path);
            }
            let normal_button = ui.button("Load Normal Map…");
            let normal_path = if normal_button.clicked() {
                pick_ground_texture()
            } else {
                Self::explorer_drop(&normal_button, &["png", "jpg", "jpeg", "tga", "bmp", "exr"])
            };
            if let Some(path) = normal_path {
                match self.cached_material_texture(&path, false) {
                    Ok(texture) => {
                        self.ground_plane
                            .material
                            .set_normal_texture(&self.device, texture);
                        self.ground_plane.normal_path = path.display().to_string();
                    }
                    Err(error) => self.editor.status = format!("Ground normal failed: {error:#}"),
                }
            }
            if !self.ground_plane.normal_path.is_empty() {
                ui.small(&self.ground_plane.normal_path);
            }
            let roughness_button = ui.button("Load Roughness Map…");
            let roughness_path = if roughness_button.clicked() {
                pick_ground_texture()
            } else {
                Self::explorer_drop(
                    &roughness_button,
                    &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                )
            };
            if let Some(path) = roughness_path {
                match self.cached_material_texture(&path, false) {
                    Ok(texture) => {
                        self.ground_plane
                            .material
                            .set_metallic_roughness_texture(&self.device, texture);
                        self.ground_plane.roughness_path = path.display().to_string();
                    }
                    Err(error) => {
                        self.editor.status = format!("Ground roughness failed: {error:#}")
                    }
                }
            }
            if !self.ground_plane.roughness_path.is_empty() {
                ui.small(&self.ground_plane.roughness_path);
            }
            let emissive_button = ui.button("Load Emissive Map…");
            let emissive_path = if emissive_button.clicked() {
                pick_ground_texture()
            } else {
                Self::explorer_drop(
                    &emissive_button,
                    &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                )
            };
            if let Some(path) = emissive_path {
                match self.cached_material_texture(&path, true) {
                    Ok(texture) => {
                        self.ground_plane
                            .material
                            .set_emissive_texture(&self.device, texture);
                        self.ground_plane.material.uniform.options[1] = 1.0;
                        self.ground_plane.emissive_path = path.display().to_string();
                    }
                    Err(error) => {
                        self.editor.status = format!("Ground emissive failed: {error:#}")
                    }
                }
            }
            if !self.ground_plane.emissive_path.is_empty() {
                ui.small(&self.ground_plane.emissive_path);
            }
        }
        ui.separator();
        material::color_controls(ui, "generated_material", &mut self.ground_plane.material.uniform);
        ui.label("The shape is opaque, depth-tested scene geometry.");
    }

    fn fx_material(
        uniform: material::MaterialUniform,
        base_color_path: String,
        normal_path: String,
        roughness_path: String,
        emissive_path: String,
    ) -> FxMaterial {
        FxMaterial {
            transparency: uniform.transparency,
            color_adjustments: uniform.color_adjustments,
            emissive_color: uniform.emissive_color,
            base_color: uniform.base_color,
            properties: uniform.properties,
            options: uniform.options,
            inspection: uniform.inspection,
            base_color_path,
            normal_path,
            roughness_path,
            emissive_path,
        }
    }

    fn capture_fx_project(&self) -> FxProject {
        let imported_model = (!self.obj_model.meshes.is_empty() || self.obj_model.imported_scene.is_some()).then(|| FxImportedModel {
            usd_time_code: (!self.editor.usd_use_stage_start).then_some(self.editor.usd_time_code),
            path: self.editor.model_path_input.clone(),
            instances: self
                .instances
                .iter()
                .map(|instance| FxTransform {
                    position: instance.position.into(),
                    rotation_degrees: instance.rotation_degrees.into(),
                    scale: instance.scale.into(),
                    global_scale: instance.global_scale,
                })
                .collect(),
            visible_instance_count: self.editor.visible_instance_count,
            selected_instance: self.editor.selected_instance,
            gizmo_at_bottom: self.gizmo_at_bottom_by_instance.clone(),
            group_visibility: self.geo_group_enabled.clone(),
            group_texture_enabled: self.geo_texture_enabled.clone(),
            textures: self
                .geo_texture_paths
                .keys()
                .chain(self.geo_normal_paths.keys())
                .chain(self.geo_roughness_paths.keys())
                .chain(self.geo_emissive_paths.keys())
                .chain(self.geo_color_settings.keys())
                .chain(self.geo_texture_scales.keys())
                .chain(self.geo_texture_colors.keys())
                .chain(self.geo_normal_strengths.keys())
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .map(|tile| FxGroupTextures {
                    emissive: self.geo_emissive_paths.get(&tile).cloned().unwrap_or_default(),
                    colors: self.geo_color_settings.get(&tile).copied(),
                    tile,
                    base_color: self
                        .geo_texture_paths
                        .get(&tile)
                        .cloned()
                        .unwrap_or_default(),
                    normal: self
                        .geo_normal_paths
                        .get(&tile)
                        .cloned()
                        .unwrap_or_default(),
                    roughness: self
                        .geo_roughness_paths
                        .get(&tile)
                        .cloned()
                        .unwrap_or_default(),
                    scale: self.geo_texture_scales.get(&tile).copied().unwrap_or(1.0),
                    color: self
                        .geo_texture_colors
                        .get(&tile)
                        .copied()
                        .unwrap_or([1.0; 4]),
                    normal_strength: self.geo_normal_strengths.get(&tile).copied().unwrap_or(1.0),
                })
                .collect(),
        });
        let generated_objects = self
            .generated_objects
            .iter()
            .chain(std::iter::once(&self.ground_plane))
            .filter(|object| object.generated)
            .map(|object| FxGeneratedObject {
                scene_id: object.scene_id,
                selected: object.selected,
                kind: object.kind,
                visible: object.visible,
                snap_bottom_to_grid: object.snap_bottom_to_grid,
                wireframe: object.wireframe,
                hide_surface: object.hide_surface,
                size: object.size,
                height: object.height,
                position_x: object.position_x,
                position_y: object.position_y,
                rotation_degrees: object.rotation_degrees,
                shape_height: object.shape_height,
                thickness: object.thickness,
                subdivisions: object.subdivisions,
                triangles: object.triangulate_subdivision,
                uv_scale: object.uv_scale,
                material: Self::fx_material(
                    object.material.uniform,
                    object.base_color_path.clone(),
                    object.normal_path.clone(),
                    object.roughness_path.clone(),
                    object.emissive_path.clone(),
                ),
            })
            .collect();
        FxProject {
            deconstruction: self.deconstruction.clone(),
            format: "FX Scene Project".to_owned(),
            version: FX_PROJECT_VERSION,
            imported_model,
            generated_objects,
            material: Self::fx_material(
                self.pbr_material.uniform,
                self.editor.base_color_path.clone(),
                self.editor.normal_path.clone(),
                self.editor.metallic_roughness_path.clone(),
                self.editor.emissive_path.clone(),
            ),
            material_graph: self.material_graph.graph.clone(),
            lights: FxLighting {
                mode: self.lighting.mode,
                lights: self.lighting.lights.clone(),
                selected_light: self.lighting.selected_light,
                area_samples: self.lighting.area_samples,
                environment_samples: self.lighting.environment_samples,
            },
            grid: FxGrid {
                visible: self.editor.show_grid,
                size: self.editor.grid_size,
                spacing: self.editor.grid_spacing,
                color: self.editor.grid_color,
            },
            viewport: FxViewport {
                background: self.editor.background,
                gizmo_always_at_bottom: self.gizmo_always_at_bottom,
                wireframe: self.model_view_mode == ModelViewMode::Wireframe,
                show_points: self.show_points,
                point_size: self.point_size,
                point_color: self.point_color,
                show_normals: self.show_normals,
                normal_mode: match self.normal_mode {
                    NormalDisplayMode::Vertex => 0,
                    NormalDisplayMode::Point => 1,
                    NormalDisplayMode::Face => 2,
                },
                normal_length: self.normal_length,
                normal_color: self.normal_color,
                show_uv_map: self.show_uv_map,
                show_uv_overlay: self.show_uv_overlay,
                show_uv_texture: self.show_uv_texture,
                show_uv_lines: self.show_uv_lines,
                camera_eye: self.camera.eye.into(),
                camera_target: self.camera.target.into(),
                camera_up: self.camera.up.into(),
                camera_fovy: self.camera.fovy,
                orthographic: self.camera.projection_mode == ProjectionMode::Orthographic,
                ortho_scale: self.camera.ortho_scale,
            },
            environment: FxEnvironment {
                path: self.editor.hdri_path.clone(),
                disabled: self.editor.hdri_image_disabled,
                intensity: self.editor.hdri_intensity,
                exposure: self.editor.hdri_exposure,
                rotation: self.editor.hdri_rotation,
            },
            material_library: self
                .material_library
                .iter()
                .map(|entry| FxLibraryMaterial {
                    id: entry.id.0,
                    name: entry.name.clone(),
                    material: Self::fx_material(
                        entry.material.uniform,
                        entry.base_color_path.clone(),
                        entry.normal_path.clone(),
                        entry.roughness_path.clone(),
                    entry.emissive_path.clone(),
                    ),
                })
                .collect(),
            mesh_material_assignments: self
                .mesh_material_assignments
                .iter()
                .map(|id| id.0)
                .collect(),
            face_material_assignments: self.face_material_assignments.clone(),
            generated_material_assignments: self
                .generated_material_assignments
                .iter()
                .map(|(object, material)| (*object, material.0))
                .collect(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_fx_project(&mut self, mut path: std::path::PathBuf) {
        if path.extension().and_then(|value| value.to_str()) != Some("fx") {
            path.set_extension("fx");
        }
        let result = serde_json::to_string_pretty(&self.capture_fx_project())
            .map_err(anyhow::Error::from)
            .and_then(|json| std::fs::write(&path, json).map_err(anyhow::Error::from));
        match result {
            Ok(()) => {
                self.project_path = path.display().to_string();
                self.editor.status = format!("Saved FX project: {}", path.display());
            }
            Err(error) => self.editor.status = format!("Could not save FX project: {error:#}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn queue_fx_project(&mut self, path: std::path::PathBuf) {
        self.fx_load_started = Some(Instant::now());
        let result = std::fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|json| serde_json::from_str::<FxProject>(&json).map_err(anyhow::Error::from));
        match result {
            Ok(project)
                if project.format == "FX Scene Project"
                    && project.version <= FX_PROJECT_VERSION =>
            {
                if let Some(model) = &project.imported_model
                    && !model.path.is_empty()
                {
                    self.editor.pending_asset = Some(std::path::PathBuf::from(&model.path));
                }
                self.pending_project = Some(project);
                self.project_path = path.display().to_string();
                self.editor.status = "FX project queued for loading".to_owned();
            }
            Ok(project) => {
                self.editor.status = format!("Unsupported FX project version {}", project.version)
            }
            Err(error) => self.editor.status = format!("Could not load FX project: {error:#}"),
        }
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
                "fx" => {
                    #[cfg(not(target_arch = "wasm32"))]
                    self.queue_fx_project(path);
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.editor.status =
                            "FX project loading is unavailable in web builds".to_owned();
                    }
                }
                "obj" | "fbx" | "usd" | "usda" | "usdc" | "usdz" => {
                    self.editor.model_path_input = path.display().to_string();
                    self.editor.asset_name = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("Model")
                        .to_owned();

                    self.editor.pending_asset = Some(path);
                    self.editor.status = "Model queued for loading".to_owned();
                }

                _ => {}
            }
        }
        ui.heading("Project");
        #[cfg(not(target_arch = "wasm32"))]
        {
            let project_drop = ui
                .label("Drop an .fx project here from Explorer")
                .on_hover_text("Loads the dropped project");
            if let Some(path) = Self::explorer_drop(&project_drop, &["fx"]) {
                self.queue_fx_project(path);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        ui.horizontal_wrapped(|ui| {
            if ui.button("Save .fx…").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("FX Project", &["fx"])
                    .set_file_name("scene.fx")
                    .save_file()
            {
                self.save_fx_project(path);
            }
            if ui.button("Load .fx…").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("FX Project", &["fx"])
                    .pick_file()
            {
                self.queue_fx_project(path);
            }
        });
        if !self.project_path.is_empty() {
            ui.small(format!("Project: {}", self.project_path));
        }
        ui.separator();
        ui.heading("Model Import");
        let path_response = ui.add(
            egui::TextEdit::singleline(&mut self.editor.model_path_input)
                .hint_text("Type or drop an OBJ, FBX, or USD file path"),
        );
        if let Some(path) = Self::explorer_drop(&path_response, usd_import::MODEL_EXTENSIONS) {
            self.editor.model_path_input = path.display().to_string();
            self.editor.pending_asset = Some(path);
            self.editor.status = "Model dropped from Explorer".to_owned();
        }
        let enter_pressed =
            path_response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
        ui.horizontal(|ui| {
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("Browse…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("3D Geometry", usd_import::MODEL_EXTENSIONS)
                    .pick_file()
                {
                    self.editor.model_path_input = path.display().to_string();
                    self.editor.pending_asset = Some(path);
                    self.editor.status = "Model queued for loading".to_owned();
                }
            }
            if ui.button("Import").clicked() || enter_pressed {
                let path = std::path::PathBuf::from(self.editor.model_path_input.trim());
                if usd_import::supported_model(&path)
                {
                    self.editor.pending_asset = Some(path);
                    self.editor.status = "Model queued for loading".to_owned();
                } else {
                    self.editor.status =
                        "Choose an OBJ, FBX, USD, USDA, USDC, or USDZ file".to_owned();
                }
            }
        });
        if usd_import::is_usd(std::path::Path::new(self.editor.model_path_input.trim())) {
            ui.checkbox(&mut self.editor.usd_use_stage_start, "USD: use stage start time");
            ui.add_enabled_ui(!self.editor.usd_use_stage_start, |ui| {
                ui.horizontal(|ui| {
                    ui.label("USD time code");
                    ui.add(egui::DragValue::new(&mut self.editor.usd_time_code).speed(1.0));
                });
            });
            ui.add(egui::Label::new(egui::RichText::new("Import evaluates the composed scene at this time. Reimport to change the snapshot.").small()).wrap());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.pending_model_load.is_some() {
            ui.horizontal(|ui| { ui.spinner(); ui.label("Loading model…"); });
        }
        ui.label(format!("Current model: {}", self.editor.asset_name));
        let mut selected_usd_camera = None;
        let mut add_usd_lights = false;
        if let Some(scene) = &self.obj_model.imported_scene {
            if !scene.lights.is_empty() || !scene.cameras.is_empty() {
                egui::CollapsingHeader::new("USD cameras & lights").show(ui, |ui| {
                    if !scene.lights.is_empty() {
                        add_usd_lights = ui.button(format!("Add {} USD scene lights", scene.lights.len())).clicked();
                    }
                    for camera in &scene.cameras {
                        if ui.button(format!("View {}", camera.name)).clicked() { selected_usd_camera = Some(camera.clone()); }
                    }
                });
            }
        }
        if let Some(camera) = selected_usd_camera {
            self.camera.eye = camera.eye.into(); self.camera.target = camera.target.into(); self.camera.up = camera.up.into();
            self.camera.fovy = camera.fovy.clamp(1.0, 175.0);
            self.camera.projection_mode = if camera.orthographic { ProjectionMode::Orthographic } else { ProjectionMode::Perspective };
            self.camera.ortho_scale = camera.ortho_scale.max(0.001);
            self.current_view = None;
            self.editor.status = format!("Viewing USD camera {}", camera.name);
            self.window.request_redraw();
        }
        if add_usd_lights { self.add_imported_usd_lights(); }

        if !self.obj_model.import_warnings.is_empty() {
            egui::CollapsingHeader::new(format!("Import notes ({})", self.obj_model.import_warnings.len()))
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                        for warning in &self.obj_model.import_warnings { ui.add(egui::Label::new(warning).wrap()); }
                    });
                });
        }
        if ui
            .add_enabled(
                !self.obj_model.meshes.is_empty() || self.obj_model.imported_scene.is_some(),
                egui::Button::new("Clear Imported Model"),
            )
            .clicked()
        {
            self.clear_imported_model();
        }
    }

    fn clear_imported_model(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        { self.pending_model_load = None; }
        self.part_selection = Default::default();
        self.obj_model = model::Model {
            import_warnings: Vec::new(),
            imported_scene: None,
            meshes: Vec::new(),
            materials: Vec::new(),
        };
        self.instances.clear();
        self.initial_instances.clear();
        self.gizmo_at_bottom_by_instance.clear();
        self.editor.selected_instance = 0;
        self.editor.visible_instance_count = 0;
        self.editor.pending_asset = None;
        self.editor.asset_name = "No model loaded".to_owned();
        self.editor.model_path_input.clear();
        self.editor.texture_enabled = false;
        self.editor.texture_name = "None".to_owned();
        self.editor.base_color_path.clear();
        self.editor.normal_path.clear();
        self.editor.metallic_roughness_path.clear();
        self.editor.emissive_path.clear();
        self.instance_buffer_dirty = true;

        self.show_points = false;
        self.show_normals = false;
        self.show_uv_map = false;
        self.show_uv_overlay = false;
        self.selected_uv_space = 0;
        self.show_all_uv_spaces = false;
        self.uv_space_offsets.clear();
        self.uv_space_cache.clear();
        self.uv_space_textures.clear();
        self.geo_group_enabled.clear();
        self.geo_texture_enabled.clear();
        self.geo_texture_paths.clear();
        self.geo_normal_paths.clear();
        self.geo_roughness_paths.clear();
        self.geo_emissive_paths.clear();
        self.geo_normal_textures.clear();
        self.geo_roughness_textures.clear();
        self.geo_emissive_textures.clear();
        self.geo_material_bind_groups.clear();
        self.geo_material_uniform_buffers.clear();
        self.geo_texture_scales.clear();
        self.geo_texture_colors.clear();
        self.geo_normal_strengths.clear();
        self.geo_color_settings.clear();
        self.mesh_material_assignments.clear();
        self.face_material_assignments.clear();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pending_uv_textures.clear();
            self.pending_model_material_root = None;
        }
        if let Ok(material) = material::PbrMaterial::new_untextured(&self.device, &self.queue) {
            self.pbr_material = material;
        }
        self.material_graph = material_graph::MaterialGraphEditor::default();

        let mut light_targets_changed = false;
        for light in &mut self.lighting.lights {
            if light
                .target_instance
                .is_some_and(|target| Self::generated_target_id(target).is_none())
            {
                light.target_instance = None;
                light_targets_changed = true;
            }
        }
        if light_targets_changed {
            self.lighting.touch();
        }

        // Imported GPU buffers are released here and are not retained as an
        // undo payload. Start a clean history baseline to prevent stale
        // instances from being restored without their mesh buffers.
        self.undo_stack.clear();
        self.undo_last_snapshot = None;
        self.undo_transaction_active = false;
        self.editor.status = "Imported model cleared".to_owned();
        self.window.request_redraw();
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
        ui.label("Gizmo Pivot");
        if ui.button("Snap Gizmo to Object Bottom").clicked() {
            if let Some(at_bottom) = self
                .gizmo_at_bottom_by_instance
                .get_mut(self.editor.selected_instance)
            {
                *at_bottom = true;
            }
            self.editor.status = "Gizmo snapped to selected object bottom".to_owned();
        }
        let selected_gizmo_at_bottom = self
            .gizmo_at_bottom_by_instance
            .get(self.editor.selected_instance)
            .copied()
            .unwrap_or(false);
        if selected_gizmo_at_bottom
            && !self.gizmo_always_at_bottom
            && ui.button("Return Gizmo to Object Center").clicked()
        {
            if let Some(at_bottom) = self
                .gizmo_at_bottom_by_instance
                .get_mut(self.editor.selected_instance)
            {
                *at_bottom = false;
            }
            self.editor.status = "Gizmo returned to selected object center".to_owned();
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
            self.lighting.selected_light = None;
            self.ground_plane.selected = false;
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
            if let Some(path) = Self::explorer_drop(&response, &["mtl"]) {
                self.editor.mtl_path_input = path.display().to_string();
                self.import_mtl(ui.ctx(), path);
            }
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
        material::color_controls(ui, "global_material", &mut self.pbr_material.uniform);

        #[cfg(not(target_arch = "wasm32"))]
        ui.horizontal_wrapped(|ui| {
            let base = ui.button("Base Color…");
            if base.clicked() {
                self.editor.pending_texture = pick_material_texture();
            }
            if let Some(path) =
                Self::explorer_drop(&base, &["png", "jpg", "jpeg", "tga", "bmp", "exr"])
            {
                self.editor.pending_texture = Some(path);
            }
            let normal = ui.button("Normal…");
            if normal.clicked() {
                self.editor.pending_normal = pick_material_texture();
            }
            if let Some(path) =
                Self::explorer_drop(&normal, &["png", "jpg", "jpeg", "tga", "bmp", "exr"])
            {
                self.editor.pending_normal = Some(path);
            }
            let roughness = ui.button("Metal/Rough…");
            if roughness.clicked() {
                self.editor.pending_metallic_roughness = pick_material_texture();
            }
            if let Some(path) =
                Self::explorer_drop(&roughness, &["png", "jpg", "jpeg", "tga", "bmp", "exr"])
            {
                self.editor.pending_metallic_roughness = Some(path);
            }
            let emissive = ui.button("Emissive…");
            if emissive.clicked() {
                self.editor.pending_emissive = pick_material_texture();
            }
            if let Some(path) =
                Self::explorer_drop(&emissive, &["png", "jpg", "jpeg", "tga", "bmp", "exr"])
            {
                self.editor.pending_emissive = Some(path);
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

    fn imported_texture_groups_ui(&mut self, ui: &mut egui::Ui) {
        let group_count = self.obj_model.meshes.len();
        if group_count == 0 {
            return;
        }
        self.geo_group_enabled.resize(group_count, true);
        self.geo_texture_enabled.resize(group_count, true);
        let mut texture_import = None;
        let mut texture_reset = None;

        egui::CollapsingHeader::new("Texture Groups / UDIM")
            .default_open(true)
            .show(ui, |ui| {
                ui.small("Assign maps independently to each uploaded mesh group.");
                for group_index in 0..group_count {
                    let mesh = &self.obj_model.meshes[group_index];
                    let tile = mesh.uv_tile;
                    let name = if mesh.name.trim().is_empty() {
                        format!("Group {}", group_index + 1)
                    } else {
                        mesh.name.clone()
                    };
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.geo_group_enabled[group_index], "");
                            ui.strong(name);
                            ui.label(format!("UDIM {}", 1001 + tile.0 + tile.1 * 10));
                            });
                        ui.checkbox(
                            &mut self.geo_texture_enabled[group_index],
                            "Textures enabled",
                        );
                        let scale = self.geo_texture_scales.entry(tile).or_insert(1.0);
                        ui.horizontal(|ui| {
                            ui.label("Texture detail");
                            ui.add(
                                egui::Slider::new(scale, 0.25..=8.0)
                                    .logarithmic(true)
                                    .custom_formatter(|value, _| format!("{value:.2}×")),
                            )
                            .on_hover_text(
                                "Higher values repeat the texture more densely; lower values make it larger",
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Texture color");
                            let color = self
                                .geo_texture_colors
                                .entry(tile)
                                .or_insert([1.0; 4]);
                            ui.color_edit_button_rgba_unmultiplied(color)
                                .on_hover_text("Tint the base-color texture for this UV space");
                        });
                        ui.horizontal(|ui| {
                            ui.label("Normal strength");
                            let strength =
                                self.geo_normal_strengths.entry(tile).or_insert(1.0);
                            ui.add(egui::Slider::new(strength, 0.0..=2.0).fixed_decimals(2))
                                .on_hover_text("0 disables the normal map; 1 uses its authored strength");
                        });

                        ui.label("Base Color");
                        ui.horizontal(|ui| {
                            let path_text = self.geo_texture_paths.entry(tile).or_default();
                            let response = ui.add(
                                egui::TextEdit::singleline(path_text).hint_text("Texture path"),
                            );
                            if let Some(path) = Self::explorer_drop(
                                &response,
                                &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                            ) {
                                *path_text = path.display().to_string();
                                texture_import =
                                    Some((tile, path, GroupTextureKind::BaseColor));
                            }
                            if response.lost_focus()
                                && ui.input(|input| input.key_pressed(egui::Key::Enter))
                            {
                                texture_import = Some((
                                    tile,
                                    std::path::PathBuf::from(path_text.trim()),
                                    GroupTextureKind::BaseColor,
                                ));
                            }
                            #[cfg(not(target_arch = "wasm32"))]
                            if ui.small_button("…").on_hover_text("Browse").clicked()
                                && let Some(path) = pick_material_texture()
                            {
                                *path_text = path.display().to_string();
                                texture_import = Some((tile, path, GroupTextureKind::BaseColor));
                            }
                            if ui.small_button("×").on_hover_text("Clear").clicked() {
                                texture_reset = Some((tile, GroupTextureKind::BaseColor));
                            }
                        });

                        ui.label("Normal Map");
                        ui.horizontal(|ui| {
                            let path_text = self.geo_normal_paths.entry(tile).or_default();
                            let response = ui.add(
                                egui::TextEdit::singleline(path_text).hint_text("Normal map path"),
                            );
                            if let Some(path) = Self::explorer_drop(
                                &response,
                                &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                            ) {
                                *path_text = path.display().to_string();
                                texture_import = Some((tile, path, GroupTextureKind::Normal));
                            }
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
                            if ui.small_button("…").on_hover_text("Browse").clicked()
                                && let Some(path) = pick_material_texture()
                            {
                                *path_text = path.display().to_string();
                                texture_import = Some((tile, path, GroupTextureKind::Normal));
                            }
                            if ui.small_button("×").on_hover_text("Clear").clicked() {
                                texture_reset = Some((tile, GroupTextureKind::Normal));
                            }
                        });

                        self.group_color_ui(ui, tile, &mut texture_import, &mut texture_reset);
                        ui.label("Roughness Map");
                        ui.horizontal(|ui| {
                            let path_text = self.geo_roughness_paths.entry(tile).or_default();
                            let response = ui.add(
                                egui::TextEdit::singleline(path_text)
                                    .hint_text("Roughness map path"),
                            );
                            if let Some(path) = Self::explorer_drop(
                                &response,
                                &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                            ) {
                                *path_text = path.display().to_string();
                                texture_import = Some((tile, path, GroupTextureKind::Roughness));
                            }
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
                            if ui.small_button("…").on_hover_text("Browse").clicked()
                                && let Some(path) = pick_material_texture()
                            {
                                *path_text = path.display().to_string();
                                texture_import = Some((tile, path, GroupTextureKind::Roughness));
                            }
                            if ui.small_button("×").on_hover_text("Clear").clicked() {
                                texture_reset = Some((tile, GroupTextureKind::Roughness));
                            }
                        });
                    });
                }
            });

        #[cfg(not(target_arch = "wasm32"))]
        if let Some((tile, path, kind)) = texture_import
            && !path.as_os_str().is_empty()
        {
            self.import_group_texture(ui.ctx(), tile, path, kind);
        }
        if let Some((tile, kind)) = texture_reset {
            self.reset_group_texture(tile, kind);
        }
    }

    fn apply_group_colors(&self, tile: (i32, i32), uniform: &mut material::MaterialUniform) {
        if let Some(colors) = self.geo_color_settings.get(&tile) {
            uniform.color_adjustments[0] = colors.hue;
            uniform.color_adjustments[1] = colors.emission_hue;
            uniform.options[1] = colors.intensity;
            uniform.emissive_color = colors.tint;
            uniform.transparency = colors.transparency;
            uniform.color_adjustments[3] = if colors.double_sided { 1.0 } else { 0.0 };
        }
        if self.geo_emissive_textures.contains_key(&tile) {
            uniform.color_adjustments[2] = 1.0;
        }
    }

    fn group_color_ui(
        &mut self,
        ui: &mut egui::Ui,
        tile: (i32, i32),
        import: &mut Option<((i32, i32), std::path::PathBuf, GroupTextureKind)>,
        reset: &mut Option<((i32, i32), GroupTextureKind)>,
    ) {
        let assigned = self
            .obj_model
            .meshes
            .iter()
            .position(|mesh| mesh.uv_tile == tile)
            .and_then(|index| self.mesh_material_assignments.get(index))
            .copied()
            .and_then(|id| {
                self.material_library
                    .iter()
                    .position(|entry| entry.id == id && id.0 != 0)
            });
        let mut uniform = assigned.map_or(self.pbr_material.uniform, |index| {
            self.material_library[index].material.uniform
        });
        if assigned.is_none() {
            self.apply_group_colors(tile, &mut uniform);
        } else {
            ui.add(egui::Label::new("These controls edit the assigned scene material, including other groups sharing it.").wrap());
        }
        let before = uniform;
        material::color_controls(ui, ("group_material", tile), &mut uniform);
        if uniform != before {
            if let Some(index) = assigned {
                self.material_library[index].material.uniform = uniform;
            } else {
                self.geo_color_settings.insert(
                    tile,
                    GroupColorSettings {
                        transparency: uniform.transparency,
                        double_sided: uniform.color_adjustments[3] > 0.5,
                        hue: uniform.color_adjustments[0],
                        emission_hue: uniform.color_adjustments[1],
                        intensity: uniform.options[1],
                        tint: uniform.emissive_color,
                    },
                );
                self.rebuild_group_material_bind_group(tile);
            }
        }
        ui.horizontal(|ui| {
            ui.label("Emissive");
            let path = self.geo_emissive_paths.entry(tile).or_default();
            let response = ui.add(
                egui::TextEdit::singleline(path)
                    .desired_width((ui.available_width() - 108.0).max(60.0))
                    .hint_text("Emissive texture path"),
            );
            if response.lost_focus()
                && ui.input(|input| input.key_pressed(egui::Key::Enter))
                && !path.trim().is_empty()
            {
                *import = Some((
                    tile,
                    std::path::PathBuf::from(path.trim()),
                    GroupTextureKind::Emissive,
                ));
            }
            if let Some(dropped) = Self::explorer_drop(&response, &["png", "jpg", "jpeg", "exr"]) {
                *import = Some((tile, dropped, GroupTextureKind::Emissive));
            }
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("Import…").clicked()
                && let Some(path) = pick_material_texture()
            {
                *import = Some((tile, path, GroupTextureKind::Emissive));
            }
            if ui.small_button("Reset").clicked() {
                *reset = Some((tile, GroupTextureKind::Emissive));
            }
        });
    }

    fn reset_group_texture(&mut self, tile: (i32, i32), kind: GroupTextureKind) {
        match kind {
            GroupTextureKind::Emissive => {
                self.geo_emissive_textures.remove(&tile);
                self.geo_emissive_paths.remove(&tile);
                self.geo_color_settings.entry(tile).or_default().intensity = 0.0;
            }
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
            || self.geo_emissive_textures.contains_key(&tile)
            || self.geo_color_settings.contains_key(&tile)
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
        let mut geometry_instances =
            (0..self.editor.visible_instance_count.min(self.instances.len()))
                .map(|index| {
                    (
                        index,
                        format!("{} — Instance {}", self.editor.asset_name, index + 1),
                    )
                })
                .collect::<Vec<_>>();
        for object in self
            .generated_objects
            .iter()
            .chain(std::iter::once(&self.ground_plane))
            .filter(|object| object.generated)
        {
            geometry_instances.push((
                Self::generated_target_index(object.scene_id),
                format!("Generated — {} {}", object.kind.name(), object.scene_id),
            ));
        }
        egui::Window::new("Lighting")
            .id(egui::Id::new("lighting_window_scrollable_v3"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
            .default_width(360.0)
            .default_height((context.content_rect().height() - 24.0).max(120.0))
            .min_height(160.0)
            .max_height((context.content_rect().height() - 24.0).max(120.0))
            .resizable(true)
            .collapsible(true)
            .constrain(true)
            .vscroll(true)
            .show(context, |ui| {
                self.hdri_ui(ui);
                ui.separator();
                lighting::editor::show(ui, &mut self.lighting, &geometry_instances);
            });
    }

    fn geometry_inspection_window(&mut self, context: &egui::Context) {
        if self.active_side_panel != Some(1) {
            return;
        }
        egui::Window::new("Geometry Inspection")
            .id(egui::Id::new("geometry_inspection_window_scrollable_v3"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
            .default_width(280.0)
            .default_height((context.content_rect().height() - 24.0).max(120.0))
            .min_height(160.0)
            .max_height((context.content_rect().height() - 24.0).max(120.0))
            .resizable(true)
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
        self.import_group_texture(context, tile, path, self.uv_import_kind);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn import_group_texture(
        &mut self,
        context: &egui::Context,
        tile: (i32, i32),
        path: std::path::PathBuf,
        kind: GroupTextureKind,
    ) {
        self.queue_group_texture(context, tile, path, kind, true);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn queue_group_texture(
        &mut self,
        context: &egui::Context,
        tile: (i32, i32),
        path: std::path::PathBuf,
        kind: GroupTextureKind,
        enable_emission: bool,
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
            enable_emission,
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
        let srgb = matches!(pending.kind, GroupTextureKind::BaseColor | GroupTextureKind::Emissive);
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
                GroupTextureKind::Emissive => "PBR per-group emission",
            },
        ) {
            Ok(texture) => texture,
            Err(error) => {
                self.editor.status = format!("Could not upload UV texture: {error:#}");
                return;
            }
        };
        match pending.kind {
            GroupTextureKind::Emissive => {
                self.geo_emissive_textures.insert(pending.tile, viewport_texture);
                self.geo_emissive_paths.insert(pending.tile, pending.path.display().to_string());
                let colors = self.geo_color_settings.entry(pending.tile).or_default();
                if pending.enable_emission && colors.intensity == 0.0 { colors.intensity = 1.0; }
            }
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
            GroupTextureKind::Emissive => "emissive",
        };
        self.editor.status = format!("Applied {kind} texture to UV space {udim}");
    }

    fn rebuild_group_material_bind_group(&mut self, tile: (i32, i32)) {
        let mut uniform = self.pbr_material.uniform;
        uniform.inspection[2] = self
            .geo_texture_scales
            .get(&tile)
            .copied()
            .unwrap_or(1.0)
            .clamp(0.05, 100.0);
        uniform.base_color = self
            .geo_texture_colors
            .get(&tile)
            .copied()
            .unwrap_or(self.pbr_material.uniform.base_color);
        uniform.properties[2] = self
            .geo_normal_strengths
            .get(&tile)
            .copied()
            .unwrap_or(self.pbr_material.uniform.properties[2])
            .clamp(0.0, 2.0);
        self.apply_group_colors(tile, &mut uniform);
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Per-group material uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = self.pbr_material.bind_group_with_overrides_and_uniform(
            &self.device,
            self.uv_space_textures
                .get(&tile)
                .map(|texture| &texture._viewport_texture),
            self.geo_normal_textures.get(&tile),
            self.geo_roughness_textures.get(&tile),
            self.geo_emissive_textures.get(&tile),
            &uniform_buffer,
        );
        self.geo_material_uniform_buffers
            .insert(tile, uniform_buffer);
        self.geo_material_bind_groups.insert(tile, bind_group);
    }

    fn add_imported_usd_lights(&mut self) {
        let Some(scene) = &self.obj_model.imported_scene else { return; };
        let lights = scene.lights.clone();
        for source in lights {
            let name = format!("USD {}", source.name);
            if self.lighting.lights.iter().any(|light| light.name == name) { continue; }
            if self.lighting.lights.len() >= self.lighting.max_lights {
                self.editor.status = format!("Viewport light limit ({}) reached", self.lighting.max_lights);
                break;
            }
            let kind = match source.kind.as_str() {
                "directional" => lighting::LightKind::Directional,
                "point" => lighting::LightKind::Point,
                "spot" => lighting::LightKind::Spot,
                "rectangle" => lighting::LightKind::Area(lighting::AreaShape::Rectangle),
                "disk" => lighting::LightKind::Area(lighting::AreaShape::Disk),
                "tube" => lighting::LightKind::Area(lighting::AreaShape::Tube),
                "sphere" => lighting::LightKind::Area(lighting::AreaShape::Sphere),
                "environment" => lighting::LightKind::Environment,
                _ => continue,
            };
            self.lighting.add(kind);
            if let Some(light) = self.lighting.selected_mut() {
                light.name = name; light.position = source.position; light.rotation_degrees = source.rotation_degrees;
                light.color = source.color; light.intensity = source.intensity.max(0.0); light.exposure = source.exposure;
                light.temperature_kelvin = source.temperature; light.use_temperature = source.use_temperature;
                light.radius = source.radius.max(0.001); light.size = source.size.map(|v| v.max(0.001));
                light.outer_angle_degrees = source.cone_angle.clamp(0.0, 180.0);
                light.inner_angle_degrees = light.outer_angle_degrees * (1.0 - source.cone_softness.clamp(0.0, 1.0));
                light.diffuse_contribution = source.diffuse; light.specular_contribution = source.specular;
            }
            if !source.environment_path.is_empty() {
                self.editor.pending_hdri = Some(source.environment_path.into());
            }
        }
        self.lighting.mode = lighting::ViewportLightingMode::SceneLights;
        self.lighting.touch();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn apply_imported_pbr_materials(&mut self) {
        let sources = self.obj_model.materials.clone();
        let mut assignments = Vec::with_capacity(sources.len());
        for source in sources {
            let result = (|| -> anyhow::Result<material_library::SceneMaterial> {
                let mut material = material::PbrMaterial::new_untextured(&self.device, &self.queue)?;
                let pbr = source.pbr.as_ref().context("Missing imported PBR parameters")?;
                material.uniform.base_color = [source.diffuse[0], source.diffuse[1], source.diffuse[2], 1.0];
                material.uniform.transparency[3] = pbr.opacity.clamp(0.0, 1.0);
                material.uniform.color_adjustments[3] = if source.double_sided { 1.0 } else { 0.0 };
                if let Some(cutoff) = source.alpha_cutoff {
                    material.uniform.transparency[0] = 2.0;
                    material.uniform.transparency[1] = cutoff.clamp(0.0, 1.0);
                }
                material.uniform.properties[0] = pbr.metallic.clamp(0.0, 1.0);
                material.uniform.properties[1] = pbr.roughness.clamp(0.02, 1.0);
                material.uniform.emissive_color = [pbr.emissive[0], pbr.emissive[1], pbr.emissive[2], 1.0];
                material.uniform.options[1] = if pbr.emissive.iter().any(|value| *value > 0.0) { 1.0 } else { 0.0 };
                for (path, kind) in [(&source.diffuse_texture, GroupTextureKind::BaseColor), (&source.normal_texture, GroupTextureKind::Normal), (&source.roughness_texture, GroupTextureKind::Roughness), (&source.emissive_texture, GroupTextureKind::Emissive)] {
                    if path.is_empty() { continue; }
                    let texture = self.cached_material_texture(std::path::Path::new(path), matches!(kind, GroupTextureKind::BaseColor | GroupTextureKind::Emissive))?;
                    match kind {
                        GroupTextureKind::BaseColor => material.set_base_color_texture(&self.device, texture),
                        GroupTextureKind::Normal => material.set_normal_texture(&self.device, texture),
                        GroupTextureKind::Roughness => material.set_metallic_roughness_texture(&self.device, texture),
                        GroupTextureKind::Emissive => material.set_emissive_texture(&self.device, texture),
                    }
                }
                // USD emission without a texture is an independent radiance color.
                if source.emissive_texture.is_empty() && material.uniform.options[1] > 0.0 {
                    material.uniform.color_adjustments[2] = 1.0; // neutral white emission binding
                }
                material.upload(&self.queue);
                Ok(material_library::SceneMaterial {
                    id: material_library::MaterialId(self.next_material_id), name: source.name,
                    material, base_color_path: source.diffuse_texture, normal_path: source.normal_texture,
                    roughness_path: source.roughness_texture, emissive_path: source.emissive_texture,
                })
            })();
            match result {
                Ok(entry) => {
                    assignments.push(entry.id);
                    self.next_material_id += 1;
                    self.material_library.push(entry);
                }
                Err(error) => {
                    assignments.push(material_library::MaterialId(0));
                    self.obj_model.import_warnings.push(format!("Material import failed: {error:#}"));
                }
            }
        }
        for (index, mesh) in self.obj_model.meshes.iter().enumerate() {
            self.mesh_material_assignments[index] = assignments.get(mesh.material).copied().unwrap_or(material_library::MaterialId(0));
        }
        if let Some(first) = assignments.first() { self.selected_library_material = *first; }
        self.editor.status = format!("Imported USD: {} mesh groups, {} materials, {} notes", self.obj_model.meshes.len(), assignments.len(), self.obj_model.import_warnings.len());
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn apply_pending_model_materials(&mut self, context: &egui::Context) {
        let Some(root) = self.pending_model_material_root.take() else {
            return;
        };
        if self.obj_model.materials.iter().any(|material| material.pbr.is_some()) {
            self.apply_imported_pbr_materials();
            return;
        }
        for source in &mut self.obj_model.materials {
            if !source.opacity_texture.is_empty() {
                let base = (!source.diffuse_texture.is_empty()).then(|| root.join(&source.diffuse_texture));
                match material::bake_opacity_texture(base.as_deref(), &root.join(&source.opacity_texture)) {
                    Ok(path) => source.diffuse_texture = path.to_string_lossy().into_owned(),
                    Err(error) => self.obj_model.import_warnings.push(format!("Opacity map {}: {error:#}", source.name)),
                }
            }
        }
        for source in &self.obj_model.materials.clone() {
            if self
                .material_library
                .iter()
                .any(|entry| entry.name == source.name)
            {
                continue;
            }
            let id = material_library::MaterialId(self.next_material_id);
            self.next_material_id += 1;
            let mut entry = self.material_library[0].duplicate(
                &self.device,
                id,
                if source.name.trim().is_empty() {
                    format!("Imported Material {}", id.0)
                } else {
                    source.name.clone()
                },
            );
            entry.material.uniform.base_color =
                [source.diffuse[0], source.diffuse[1], source.diffuse[2], 1.0];
            entry.material.uniform.transparency = material::default_transparency();
            entry.material.uniform.transparency[3] = source.opacity;
            entry.base_color_path = (!source.diffuse_texture.is_empty())
                .then(|| root.join(&source.diffuse_texture).display().to_string())
                .unwrap_or_default();
            entry.normal_path = (!source.normal_texture.is_empty())
                .then(|| root.join(&source.normal_texture).display().to_string())
                .unwrap_or_default();
            entry.roughness_path = (!source.roughness_texture.is_empty())
                .then(|| root.join(&source.roughness_texture).display().to_string())
                .unwrap_or_default();
            entry.emissive_path = (!source.emissive_texture.is_empty())
                .then(|| root.join(&source.emissive_texture).display().to_string()).unwrap_or_default();
            for (path, kind) in [(&entry.base_color_path, GroupTextureKind::BaseColor),
                (&entry.normal_path, GroupTextureKind::Normal), (&entry.roughness_path, GroupTextureKind::Roughness)] {
                if path.is_empty() { continue; }
                match self.cached_material_texture(std::path::Path::new(path), matches!(kind, GroupTextureKind::BaseColor)) {
                    Ok(texture) => match kind {
                        GroupTextureKind::BaseColor => entry.material.set_base_color_texture(&self.device, texture),
                        GroupTextureKind::Normal => entry.material.set_normal_texture(&self.device, texture),
                        _ => entry.material.set_metallic_roughness_texture(&self.device, texture),
                    },
                    Err(error) => self.obj_model.import_warnings.push(format!("Material texture: {error:#}")),
                }
            }
            if !entry.emissive_path.is_empty() {
                match self.cached_material_texture(std::path::Path::new(&entry.emissive_path), true) {
                    Ok(texture) => {
                        entry.material.set_emissive_texture(&self.device, texture);
                        entry.material.uniform.options[1] = 1.0;
                    }
                    Err(error) => self.editor.status = format!("Could not load MTL emission: {error:#}"),
                }
            }
            self.material_library.push(entry);
        }
        for mesh in &self.obj_model.meshes {
            if let Some(source) = self.obj_model.materials.get(mesh.material) {
                self.geo_color_settings.entry(mesh.uv_tile).or_default().transparency[3] = source.opacity;
            }
        }
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
                    (!material.emissive_texture.is_empty()).then(|| {
                        (
                            tile,
                            root.join(material.emissive_texture),
                            GroupTextureKind::Emissive,
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
        let mut materials = materials;
        for material in &mut materials {
            if !material.dissolve_texture.is_empty() {
                let base = (!material.diffuse_texture.is_empty()).then(|| root.join(&material.diffuse_texture));
                match material::bake_opacity_texture(base.as_deref(), &root.join(&material.dissolve_texture)) {
                    Ok(path) => material.diffuse_texture = path.to_string_lossy().into_owned(),
                    Err(error) => {
                        self.editor.status = format!("Could not import MTL opacity map: {error:#}");
                        return;
                    }
                }
            }
        }
        let mut jobs = Vec::new();
        let mut solid_colors = Vec::new();
        for mesh in &self.obj_model.meshes {
            let Some(material) = materials.get(mesh.material) else {
                continue;
            };
            self.geo_color_settings.entry(mesh.uv_tile).or_default().transparency[3] =
                material.unknown_param.get("Tr").and_then(|v| v.parse::<f32>().ok())
                    .map_or(material.dissolve, |v| 1.0 - v).clamp(0.0, 1.0);
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
            if let Some(texture) = material.unknown_param.get("map_Ke") {
                jobs.push((
                    mesh.uv_tile,
                    root.join(texture),
                    GroupTextureKind::Emissive,
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

    fn rebuild_uv_space_cache(&mut self) {
        for mesh in &mut self.obj_model.meshes {
            mesh.ensure_uv_edges();
        }
        let mut spaces =
            BTreeMap::<(i32, i32), (Arc<str>, Vec<UvEdge>, HashSet<([u32; 2], [u32; 2])>)>::new();
        for (group_index, mesh) in self.obj_model.meshes.iter().enumerate() {
            if mesh.geometry_stats.uv_vertex_count == 0 {
                continue;
            }
            let group_name: Arc<str> = if mesh.name.trim().is_empty() {
                format!("Group {}", group_index + 1).into()
            } else {
                mesh.name.clone().into()
            };
            for triangle in mesh.uv_edges.chunks_exact(3) {
                let center = [
                    (triangle[0].0[0] + triangle[1].0[0] + triangle[2].0[0]) / 3.0,
                    (triangle[0].0[1] + triangle[1].0[1] + triangle[2].0[1]) / 3.0,
                ];
                let entry = spaces
                    .entry((center[0].floor() as i32, center[1].floor() as i32))
                    .or_insert_with(|| (group_name.clone(), Vec::new(), HashSet::new()));
                for &(start, end) in triangle {
                    let start_bits = [start[0].to_bits(), start[1].to_bits()];
                    let end_bits = [end[0].to_bits(), end[1].to_bits()];
                    let key = if start_bits <= end_bits {
                        (start_bits, end_bits)
                    } else {
                        (end_bits, start_bits)
                    };
                    if entry.2.insert(key) {
                        entry.1.push((start, end));
                    }
                }
            }
        }
        self.uv_space_cache = spaces
            .into_iter()
            .map(|(tile, (name, edges, _))| (tile, (name, Arc::from(edges))))
            .collect();
    }

    fn uv_map_window(&mut self, context: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        self.finish_pending_uv_texture(context);
        if !self.show_uv_map {
            return;
        }
        if self.uv_space_cache.is_empty() && !self.obj_model.meshes.is_empty() {
            self.rebuild_uv_space_cache();
        }
        // Arc-backed entries make this a shallow per-frame copy while allowing
        // the UI closure to freely mutate the rest of the editor state.
        let uv_spaces = self.uv_space_cache.clone();
        self.selected_uv_space = self
            .selected_uv_space
            .min(uv_spaces.len().saturating_sub(1));

        let mut open = self.show_uv_map;
        let mut texture_import_tile = None;
        let mut texture_import = None;
        let mut texture_reset = None;
        egui::Window::new("UV Map")
            .id(egui::Id::new("uv_map_window"))
            .open(&mut open)
            .default_pos(egui::pos2(260.0, 80.0))
            .default_width(340.0)
            .min_width(280.0)
            .min_width(240.0)
            .resizable(true)
            .constrain(true)
            .show(context, |ui| {
                egui::ComboBox::from_id_salt("uv_import_kind").selected_text(match self.uv_import_kind {
                    GroupTextureKind::BaseColor => "Import: Base color", GroupTextureKind::Normal => "Import: Normal",
                    GroupTextureKind::Roughness => "Import: Roughness", GroupTextureKind::Emissive => "Import: Emissive",
                }).show_ui(ui, |ui| {
                    for (kind, name) in [(GroupTextureKind::BaseColor, "Base color"), (GroupTextureKind::Normal, "Normal"), (GroupTextureKind::Roughness, "Roughness"), (GroupTextureKind::Emissive, "Emissive")] {
                        ui.selectable_value(&mut self.uv_import_kind, kind, name);
                    }
                });
                if let Some((tile, _)) = uv_spaces.get(self.selected_uv_space) {
                    egui::CollapsingHeader::new("Selected group color & emission").show(ui, |ui| {
                        self.group_color_ui(ui, *tile, &mut texture_import, &mut texture_reset);
                    });
                }


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
                let visible_tile_count = if self.show_all_uv_spaces {
                    uv_spaces.len().max(1)
                } else {
                    1
                };
                let per_tile_edge_limit = 75_000_usize.div_ceil(visible_tile_count).max(1);
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
                        let mut line_mesh = egui::Mesh::default();
                        // More edges than screen pixels cannot add visible
                        // detail, but each one expands to four UI vertices and
                        // six indices. Apply a screen-space LOD to keep egui's
                        // shared buffers well below the GPU's allocation limit.
                        let pixel_budget =
                            ((tile_rect.width() * tile_rect.height()) / 3.0).max(1.0) as usize;
                        let edge_budget = pixel_budget.min(per_tile_edge_limit).max(1);
                        let edge_step = edges.len().div_ceil(edge_budget).max(1);
                        for &(start, end) in edges.iter().step_by(edge_step) {
                            let start = uv_to_screen(start);
                            let end = uv_to_screen(end);
                            let delta = end - start;
                            let length = delta.length();
                            if length <= f32::EPSILON {
                                continue;
                            }
                            let normal = egui::vec2(-delta.y, delta.x) * (0.5 / length);
                            let first = line_mesh.vertices.len() as u32;
                            line_mesh.colored_vertex(start + normal, uv_color);
                            line_mesh.colored_vertex(start - normal, uv_color);
                            line_mesh.colored_vertex(end + normal, uv_color);
                            line_mesh.colored_vertex(end - normal, uv_color);
                            line_mesh.add_triangle(first, first + 1, first + 2);
                            line_mesh.add_triangle(first + 1, first + 3, first + 2);
                        }
                        if !line_mesh.indices.is_empty() {
                            painter.add(egui::Shape::mesh(line_mesh));
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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((tile, path, kind)) = texture_import { self.import_group_texture(context, tile, path, kind); }
        if let Some((tile, kind)) = texture_reset { self.reset_group_texture(tile, kind); }

    }

    fn material_browser_window(&mut self, context: &egui::Context) {
        if self.active_side_panel != Some(5) {
            self.dragged_material = None;
            return;
        }
        let missing_previews = self
            .material_library
            .iter()
            .filter(|entry| !self.material_previews.contains_key(&entry.id))
            .map(|entry| entry.id)
            .collect::<Vec<_>>();
        for id in missing_previews {
            let preview = material_preview::MaterialPreview::new(
                &self.device,
                &mut self.egui.renderer,
                &self.pbr_material.layout,
                &self.environment_lighting_layout,
            );
            self.material_previews.insert(id, preview);
        }
        self.mesh_material_assignments
            .resize(self.obj_model.meshes.len(), material_library::MaterialId(0));
        self.face_material_assignments
            .resize_with(self.obj_model.meshes.len(), Vec::new);
        self.selected_material_mesh = self
            .selected_material_mesh
            .min(self.obj_model.meshes.len().saturating_sub(1));

        let mut duplicate = None;
        let mut remove = None;
        let mut assign_faces = false;
        let mut clear_faces = false;
        let mut library_texture_import = None;
        #[cfg(not(target_arch = "wasm32"))]
        let mut save_user_material = None;
        #[cfg(not(target_arch = "wasm32"))]
        let mut import_new_texture = None;
        let material_panel_width = (context.content_rect().width() * 0.4).clamp(440.0, 600.0)
            .min((context.content_rect().width() - 180.0).max(220.0));
        egui::Window::new("Material Browser")
            .id(egui::Id::new("material_library_window_scrollable_v4"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
            .default_width(material_panel_width)
            .max_width(material_panel_width)
            .default_height((context.content_rect().height() - 24.0).max(120.0))
            .min_height(160.0)
            .max_height((context.content_rect().height() - 24.0).max(120.0))
            .resizable(true)
            .vscroll(true)
            .show(context, |ui| {
                ui.set_width(material_panel_width - 24.0);
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                ui.horizontal(|ui| {
                    ui.heading("Materials");
                    if ui.button("+ New").clicked() {
                        let id = material_library::MaterialId(self.next_material_id);
                        match material::PbrMaterial::new_untextured(&self.device, &self.queue) {
                            Ok(material) => {
                                self.next_material_id += 1;
                                self.material_library.push(material_library::SceneMaterial {
                                    id,
                                    name: format!("Material {}", id.0),
                                    material,
                                    base_color_path: String::new(),
                                    normal_path: String::new(),
                                    roughness_path: String::new(),
                                    emissive_path: String::new(),
                                });
                                self.selected_library_material = id;
                            }
                            Err(error) => {
                                self.editor.status =
                                    format!("Could not create material: {error:#}");
                            }
                        }
                    }
                });
                #[cfg(not(target_arch = "wasm32"))]
                ui.menu_button("Import Texture as Material…", |ui| {
                    for (kind, name) in [(GroupTextureKind::BaseColor, "Base Color…"), (GroupTextureKind::Normal, "Normal…"), (GroupTextureKind::Roughness, "Metal / Rough…"), (GroupTextureKind::Emissive, "Emissive…")] {
                        if ui.button(name).clicked() {
                            ui.close();
                            import_new_texture = pick_material_texture().map(|path| (path, kind));
                        }
                    }
                });
                ui.add(
                    egui::TextEdit::singleline(&mut self.material_browser_search)
                        .hint_text("Search materials…"),
                );
                ui.small("Drag a material card onto an object or mesh group below.");
                let search = self.material_browser_search.trim().to_ascii_lowercase();
                ui.horizontal_wrapped(|ui| {
                    for entry in &self.material_library {
                        if !search.is_empty() && !entry.name.to_ascii_lowercase().contains(&search)
                        {
                            continue;
                        }
                        let response = ui
                            .vertical(|ui| {
                                let (preview_rect, preview_response) = ui.allocate_exact_size(
                                    egui::vec2(92.0, 92.0),
                                    egui::Sense::click_and_drag(),
                                );
                                if let Some(preview) = self.material_previews.get(&entry.id) {
                                    ui.painter().image(
                                        preview.texture_id,
                                        preview_rect,
                                        egui::Rect::from_min_max(
                                            egui::Pos2::ZERO,
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::WHITE,
                                    );
                                }
                                let label_response = ui.add_sized(
                                    [92.0, 24.0],
                                    egui::Button::new(&entry.name)
                                        .selected(self.selected_library_material == entry.id)
                                        .sense(egui::Sense::click_and_drag()),
                                );
                                preview_response.union(label_response)
                            })
                            .inner;
                        if response.clicked() {
                            self.selected_library_material = entry.id;
                        }
                        if response.drag_started() {
                            self.dragged_material = Some(entry.id);
                        }
                    }
                });
                if ui.input(|input| !input.pointer.primary_down())
                    && !ui.rect_contains_pointer(ui.max_rect())
                {
                    self.dragged_material = None;
                }

                if let Some(entry) = self
                    .material_library
                    .iter_mut()
                    .find(|entry| entry.id == self.selected_library_material)
                {
                    ui.separator();
                    ui.heading("Selected Material");
                    ui.text_edit_singleline(&mut entry.name);
                    ui.horizontal(|ui| {
                        ui.label("Base Color");
                        ui.color_edit_button_rgba_unmultiplied(
                            &mut entry.material.uniform.base_color,
                        );
                    });
                    ui.add(
                        egui::Slider::new(&mut entry.material.uniform.properties[0], 0.0..=1.0)
                            .text("Metallic"),
                    );
                    ui.add(
                        egui::Slider::new(&mut entry.material.uniform.properties[1], 0.02..=1.0)
                            .text("Roughness"),
                    );
                    ui.add(
                        egui::Slider::new(&mut entry.material.uniform.properties[2], 0.0..=2.0)
                            .text("Normal Strength"),
                    );
                    material::color_controls(ui, ("library_material", entry.id.0), &mut entry.material.uniform);
                    egui::CollapsingHeader::new("Import Texture Maps").default_open(true).show(ui, |ui| {
                        for (label, path, kind) in [
                            (
                                "Base Color",
                                &mut entry.base_color_path,
                                GroupTextureKind::BaseColor,
                            ),
                            ("Normal", &mut entry.normal_path, GroupTextureKind::Normal),
                            ("Emissive", &mut entry.emissive_path, GroupTextureKind::Emissive),
                            (
                                "Metal / Rough",
                                &mut entry.roughness_path,
                                GroupTextureKind::Roughness,
                            ),
                        ] {
                            ui.horizontal(|ui| {
                                ui.label(label);
                                let response = ui.add(
                                    egui::TextEdit::singleline(path).desired_width((ui.available_width() - 58.0).max(60.0)).hint_text("Texture path"),
                                );
                                if let Some(dropped) = Self::explorer_drop(
                                    &response,
                                    &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                                ) {
                                    *path = dropped.display().to_string();
                                    library_texture_import = Some((entry.id, dropped, kind));
                                }
                                if response.lost_focus()
                                    && ui.input(|input| input.key_pressed(egui::Key::Enter))
                                {
                                    library_texture_import = Some((
                                        entry.id,
                                        std::path::PathBuf::from(path.trim()),
                                        kind,
                                    ));
                                }
                                #[cfg(not(target_arch = "wasm32"))]
                                if ui.small_button("Import…").clicked()
                                    && let Some(selected) = pick_material_texture()
                                {
                                    *path = selected.display().to_string();
                                    library_texture_import = Some((entry.id, selected, kind));
                                }
                            });
                        }
                    });
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("Save to User Library").on_hover_text("Save a reusable preset and copy its textures into your personal library").clicked() {
                        save_user_material = Some(entry.id);
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Duplicate").clicked() {
                            duplicate = Some(entry.id);
                        }
                        if entry.id.0 != 1 && ui.button("Delete").clicked() {
                            remove = Some(entry.id);
                        }
                    });
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    if ui.button("Open User Library in Explorer").clicked() {
                        self.explorer_open = true;
                        self.asset_explorer.user_library = true;
                        self.preferences.explorer_library = true;
                    }
                    if !self.user_library.message.is_empty() { ui.add(egui::Label::new(&self.user_library.message).wrap()); }
                }
                ui.separator();
                ui.heading("Assignments");
                let released = ui.input(|input| input.pointer.any_released());
                for (index, mesh) in self.obj_model.meshes.iter().enumerate() {
                    let assigned = self.mesh_material_assignments[index];
                    let assigned_name = self
                        .material_library
                        .iter()
                        .find(|entry| entry.id == assigned)
                        .map(|entry| entry.name.as_str())
                        .unwrap_or("Imported / Texture Group");
                    let response = ui.selectable_label(
                        self.selected_material_mesh == index,
                        format!("△ {}  ·  {assigned_name}", mesh.name),
                    );
                    if response.clicked() {
                        self.selected_material_mesh = index;
                        self.face_assign_first = 0;
                        self.face_assign_end = (mesh.num_elements / 3).max(1);
                    }
                    if response.hovered()
                        && released
                        && let Some(material) = self.dragged_material.take()
                    {
                        self.mesh_material_assignments[index] = material;
                        self.editor.status = format!("Assigned material to {}", mesh.name);
                    }
                }
                for object in self
                    .generated_objects
                    .iter()
                    .chain(std::iter::once(&self.ground_plane))
                    .filter(|object| object.generated)
                {
                    let assigned_name = self
                        .generated_material_assignments
                        .get(&object.scene_id)
                        .and_then(|id| self.material_library.iter().find(|entry| entry.id == *id))
                        .map(|entry| entry.name.as_str())
                        .unwrap_or("Object Material");
                    let response = ui.label(format!(
                        "◇ {} {}  ·  {assigned_name}",
                        object.kind.name(),
                        object.scene_id
                    ));
                    if response.hovered()
                        && released
                        && let Some(material) = self.dragged_material.take()
                    {
                        self.generated_material_assignments
                            .insert(object.scene_id, material);
                    }
                }

                if let Some(mesh) = self.obj_model.meshes.get(self.selected_material_mesh) {
                    let face_count = mesh.num_elements / 3;
                    ui.separator();
                    ui.strong(format!("Face assignment · {}", mesh.name));
                    ui.small(format!(
                        "Faces use zero-based ranges; this mesh has {face_count}."
                    ));
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.face_assign_first)
                                .range(0..=face_count)
                                .prefix("First "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.face_assign_end)
                                .range(0..=face_count)
                                .prefix("End "),
                        );
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Assign Selected Material").clicked() {
                            assign_faces = true;
                        }
                        if ui.button("Clear Face Overrides").clicked() {
                            clear_faces = true;
                        }
                    });
                    for assignment in &self.face_material_assignments[self.selected_material_mesh] {
                        let name = self
                            .material_library
                            .iter()
                            .find(|entry| entry.id == assignment.material)
                            .map(|entry| entry.name.as_str())
                            .unwrap_or("Missing Material");
                        ui.label(format!(
                            "Faces {}–{} → {name}",
                            assignment.first_face,
                            assignment.end_face.saturating_sub(1)
                        ));
                    }
                }
            });

        if context.input(|input| input.pointer.any_released()) {
            self.dragged_material = None;
        }

        if let Some(source) = duplicate
            && let Some(index) = self
                .material_library
                .iter()
                .position(|entry| entry.id == source)
        {
            let id = material_library::MaterialId(self.next_material_id);
            self.next_material_id += 1;
            let name = format!("{} Copy", self.material_library[index].name);
            let entry = self.material_library[index].duplicate(&self.device, id, name);
            self.material_library.push(entry);
            self.selected_library_material = id;
        }
        if let Some(id) = remove {
            self.material_library.retain(|entry| entry.id != id);
            self.material_previews.remove(&id);
            for assignment in &mut self.mesh_material_assignments {
                if *assignment == id {
                    *assignment = material_library::MaterialId(0);
                }
            }
            for assignments in &mut self.face_material_assignments {
                assignments.retain(|assignment| assignment.material != id);
            }
            self.generated_material_assignments
                .retain(|_, material| *material != id);
            self.selected_library_material = material_library::MaterialId(1);
        }
        if clear_faces && self.selected_material_mesh < self.face_material_assignments.len() {
            self.face_material_assignments[self.selected_material_mesh].clear();
        }
        if assign_faces
            && let Some(mesh) = self.obj_model.meshes.get(self.selected_material_mesh)
            && let Some(assignment) = (material_library::FaceMaterialAssignment {
                first_face: self.face_assign_first,
                end_face: self.face_assign_end,
                material: self.selected_library_material,
            })
            .normalized(mesh.num_elements / 3)
        {
            self.face_material_assignments[self.selected_material_mesh].push(assignment);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((id, path, kind)) = library_texture_import
            && !path.as_os_str().is_empty()
        {
            let srgb = matches!(kind, GroupTextureKind::BaseColor | GroupTextureKind::Emissive);
            match self.cached_material_texture(&path, srgb) {
                Ok(texture) => {
                    if let Some(entry) = self
                        .material_library
                        .iter_mut()
                        .find(|entry| entry.id == id)
                    {
                        match kind {
                            GroupTextureKind::Emissive => {
                                entry.material.set_emissive_texture(&self.device, texture);
                                entry.material.uniform.options[1] = 1.0;
                                entry.emissive_path = path.display().to_string();
                            }
                            GroupTextureKind::BaseColor => {
                                entry.material.set_base_color_texture(&self.device, texture);
                                entry.base_color_path = path.display().to_string();
                            }
                            GroupTextureKind::Normal => {
                                entry.material.set_normal_texture(&self.device, texture);
                                entry.normal_path = path.display().to_string();
                            }
                            GroupTextureKind::Roughness => {
                                entry
                                    .material
                                    .set_metallic_roughness_texture(&self.device, texture);
                                entry.roughness_path = path.display().to_string();
                            }
                        }
                    }
                }
                Err(error) => {
                    self.editor.status = format!("Could not load library texture: {error:#}");
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(id) = save_user_material { self.save_user_material(id); }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((path, kind)) = import_new_texture { self.import_texture_as_material(&path, kind); }
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
        let mut clear_imported = false;
        let mut frame_imported = false;
        let mut remove_generated = false;
        let mut remove_stored_generated = None;
        let mut select_stored_generated = None;
        let mut delete_light = None;

        egui::Window::new("Scene Outliner")
            .id(egui::Id::new("geo_tree_window_scrollable_v3"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
            .default_width(420.0)
            .default_height((context.content_rect().height() - 24.0).max(120.0))
            .min_height(160.0)
            .max_height((context.content_rect().height() - 24.0).max(120.0))
            .collapsible(true)
            .constrain(true)
            .vscroll(true)
            .resizable(true)
            .show(context, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 2.0);
                ui.spacing_mut().button_padding = egui::vec2(4.0, 1.0);
                ui.visuals_mut().selection.bg_fill = egui::Color32::from_rgb(74, 74, 74);
                ui.visuals_mut().selection.stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(245, 125, 35));
                ui.horizontal(|ui| {
                    ui.strong(egui::RichText::new("▤ Scene Collection").size(13.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(format!(
                            "{} obj · {} light",
                            self.instances.len() + self.generated_object_count(),
                            self.lighting.lights.len()
                        ));
                    });
                });
                ui.add(
                    egui::TextEdit::singleline(&mut self.outliner_search)
                        .hint_text("Search scene…"),
                );
                ui.horizontal_wrapped(|ui| {
                    if ui.small_button("Show All").clicked() {
                        self.geo_group_enabled.fill(true);
                        self.ground_plane.visible = true;
                        for object in &mut self.generated_objects {
                            object.visible = true;
                        }
                        for light in &mut self.lighting.lights {
                            light.viewport_enabled = true;
                        }
                        self.editor.show_grid = true;
                        self.lighting.touch();
                    }
                    if ui.small_button("Hide All").clicked() {
                        self.geo_group_enabled.fill(false);
                        self.ground_plane.visible = false;
                        for object in &mut self.generated_objects {
                            object.visible = false;
                        }
                        for light in &mut self.lighting.lights {
                            light.viewport_enabled = false;
                        }
                        self.editor.show_grid = false;
                        self.lighting.touch();
                    }
                    if !self.outliner_search.is_empty() && ui.small_button("Clear Search").clicked()
                    {
                        self.outliner_search.clear();
                    }
                });
                let filter = self.outliner_search.trim().to_ascii_lowercase();
                let matches =
                    |name: &str| filter.is_empty() || name.to_ascii_lowercase().contains(&filter);

                egui::CollapsingHeader::new(format!(
                    "▣ Collection ({})",
                    self.instances.len() + self.generated_object_count()
                ))
                .default_open(true)
                .show(ui, |ui| {
                    if !self.obj_model.meshes.is_empty() && matches(&self.editor.asset_name) {
                        let mut imported_visible =
                            self.geo_group_enabled.iter().any(|visible| *visible);
                        ui.horizontal(|ui| {
                            if Self::outliner_eye_toggle(ui, &mut imported_visible) {
                                self.geo_group_enabled.fill(imported_visible);
                            }
                            let selected = !self.ground_plane.selected
                                && self.lighting.selected_light.is_none()
                                && !self.instances.is_empty();
                            if ui
                                .selectable_label(selected, format!("◇ {}", self.editor.asset_name))
                                .clicked()
                            {
                                self.editor.selected_instance = self
                                    .editor
                                    .selected_instance
                                    .min(self.instances.len().saturating_sub(1));
                                self.ground_plane.selected = false;
                                self.lighting.selected_light = None;
                            }
                            if ui
                                .small_button("⌖")
                                .on_hover_text("Frame Selected")
                                .clicked()
                            {
                                frame_imported = true;
                            }
                            if ui
                                .small_button("×")
                                .on_hover_text("Clear Imported Model")
                                .clicked()
                            {
                                clear_imported = true;
                            }
                        });
                        ui.indent("outliner_imported_children", |ui| {
                            egui::CollapsingHeader::new(format!(
                                "◇ Objects ({})",
                                self.instances.len()
                            ))
                            .default_open(true)
                            .show(ui, |ui| {
                                for index in 0..self.instances.len() {
                                    if !matches(&format!("Instance {}", index + 1)) {
                                        continue;
                                    }
                                    if ui
                                        .selectable_label(
                                            self.editor.selected_instance == index
                                                && !self.ground_plane.selected
                                                && self.lighting.selected_light.is_none(),
                                            format!("◇ Instance {}", index + 1),
                                        )
                                        .clicked()
                                    {
                                        self.editor.selected_instance = index;
                                        self.ground_plane.selected = false;
                                        self.lighting.selected_light = None;
                                    }
                                }
                            });
                            egui::CollapsingHeader::new(format!("△ Mesh Data ({group_count})"))
                                .show(ui, |ui| {
                                    for index in 0..group_count {
                                        let mesh = &self.obj_model.meshes[index];
                                        let name = if mesh.name.trim().is_empty() {
                                            format!("Group {}", index + 1)
                                        } else {
                                            mesh.name.clone()
                                        };
                                        if !matches(&name) {
                                            continue;
                                        }
                                        ui.horizontal(|ui| {
                                            Self::outliner_eye_toggle(
                                                ui,
                                                &mut self.geo_group_enabled[index],
                                            );
                                            ui.label(
                                                egui::RichText::new(format!("△ {name}"))
                                                    .color(egui::Color32::from_rgb(108, 190, 103)),
                                            );
                                            ui.weak(format!(
                                                "{} tris",
                                                mesh.geometry_stats.triangle_count
                                            ));
                                        });
                                    }
                                });
                        });
                    }
                    for (index, object) in self.generated_objects.iter_mut().enumerate() {
                        if !matches(object.kind.name()) {
                            continue;
                        }
                        ui.horizontal(|ui| {
                            Self::outliner_eye_toggle(ui, &mut object.visible);
                            if ui
                                .selectable_label(
                                    false,
                                    format!("◇ {} {}", object.kind.name(), object.scene_id),
                                )
                                .clicked()
                            {
                                select_stored_generated = Some(index);
                            }
                            if ui
                                .small_button("×")
                                .on_hover_text("Delete Object")
                                .clicked()
                            {
                                remove_stored_generated = Some(index);
                            }
                        });
                    }
                    if self.ground_plane.generated && matches(self.ground_plane.kind.name()) {
                        ui.horizontal(|ui| {
                            Self::outliner_eye_toggle(ui, &mut self.ground_plane.visible);
                            if ui
                                .selectable_label(
                                    self.ground_plane.selected,
                                    format!(
                                        "◇ {} {}",
                                        self.ground_plane.kind.name(),
                                        self.ground_plane.scene_id
                                    ),
                                )
                                .clicked()
                            {
                                self.ground_plane.selected = true;
                                self.lighting.selected_light = None;
                            }
                            if ui
                                .small_button("×")
                                .on_hover_text("Delete Object")
                                .clicked()
                            {
                                remove_generated = true;
                            }
                        });
                    }
                });

                egui::CollapsingHeader::new(format!("▣ Lights ({})", self.lighting.lights.len()))
                    .default_open(true)
                    .show(ui, |ui| {
                        for light in &mut self.lighting.lights {
                            if !matches(&light.name) {
                                continue;
                            }
                            ui.horizontal(|ui| {
                                Self::outliner_eye_toggle(ui, &mut light.viewport_enabled);
                                if ui
                                    .selectable_label(
                                        self.lighting.selected_light == Some(light.id),
                                        format!("◉ {}", light.name),
                                    )
                                    .clicked()
                                {
                                    self.lighting.selected_light = Some(light.id);
                                    self.ground_plane.selected = false;
                                }
                                ui.checkbox(&mut light.enabled, "▣")
                                    .on_hover_text("Render Enabled");
                                if ui.small_button("×").on_hover_text("Delete Light").clicked() {
                                    delete_light = Some(light.id);
                                }
                            });
                        }
                    });

                egui::CollapsingHeader::new("▣ Scene Data")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            Self::outliner_eye_toggle(ui, &mut self.editor.show_grid);
                            ui.label("⊞ Grid");
                        });
                        ui.label(format!("▹ Camera · {}", self.current_view_name()));
                        if let Some(name) = &self.editor.hdri_name
                            && matches(name)
                        {
                            ui.horizontal(|ui| {
                                let mut environment_visible = !self.editor.hdri_image_disabled;
                                if Self::outliner_eye_toggle(ui, &mut environment_visible) {
                                    self.editor.hdri_image_disabled = !environment_visible;
                                }
                                ui.label(format!("◉ World · {name}"));
                            });
                        }
                    });

                ui.separator();
                ui.collapsing("▧ Mesh Materials & UDIM Data", |ui| {
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
                                    egui::TextEdit::singleline(texture_path)
                                        .hint_text("Texture path"),
                                );
                                if let Some(path) = Self::explorer_drop(
                                    &response,
                                    &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                                ) {
                                    *texture_path = path.display().to_string();
                                    texture_import =
                                        Some((tile, path, GroupTextureKind::BaseColor));
                                }
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
                                    texture_import =
                                        Some((tile, path, GroupTextureKind::BaseColor));
                                }
                                if ui.small_button("Reset").clicked() {
                                    texture_reset = Some((tile, GroupTextureKind::BaseColor));
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Normal");
                                let path_text = self.geo_normal_paths.entry(tile).or_default();
                                let response = ui.add(
                                    egui::TextEdit::singleline(path_text)
                                        .hint_text("Normal map path"),
                                );
                                if let Some(path) = Self::explorer_drop(
                                    &response,
                                    &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                                ) {
                                    *path_text = path.display().to_string();
                                    texture_import = Some((tile, path, GroupTextureKind::Normal));
                                }
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
                            self.group_color_ui(ui, tile, &mut texture_import, &mut texture_reset);
                            ui.horizontal(|ui| {
                                ui.label("Roughness");
                                let path_text = self.geo_roughness_paths.entry(tile).or_default();
                                let response = ui.add(
                                    egui::TextEdit::singleline(path_text)
                                        .hint_text("Roughness map path"),
                                );
                                if let Some(path) = Self::explorer_drop(
                                    &response,
                                    &["png", "jpg", "jpeg", "tga", "bmp", "exr"],
                                ) {
                                    *path_text = path.display().to_string();
                                    texture_import =
                                        Some((tile, path, GroupTextureKind::Roughness));
                                }
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
                                    texture_import =
                                        Some((tile, path, GroupTextureKind::Roughness));
                                }
                                if ui.small_button("Reset").clicked() {
                                    texture_reset = Some((tile, GroupTextureKind::Roughness));
                                }
                            });
                        });
                    }
                });
            });

        if frame_imported {
            self.frame_selected_instance();
        }
        if clear_imported {
            self.clear_imported_model();
        }
        if let Some(index) = remove_stored_generated {
            if index < self.generated_objects.len() {
                let removed = self.generated_objects.swap_remove(index);
                self.generated_material_assignments
                    .remove(&removed.scene_id);
                let target = Self::generated_target_index(removed.scene_id);
                for light in &mut self.lighting.lights {
                    if light.target_instance == Some(target) {
                        light.target_instance = None;
                    }
                }
                self.lighting.touch();
                self.editor.status = format!(
                    "Deleted generated {} {}",
                    removed.kind.name(),
                    removed.scene_id
                );
            }
        } else if let Some(index) = select_stored_generated {
            self.select_stored_generated(index);
        }
        if remove_generated {
            self.generated_material_assignments
                .remove(&self.ground_plane.scene_id);
            let target = Self::generated_target_index(self.ground_plane.scene_id);
            for light in &mut self.lighting.lights {
                if light.target_instance == Some(target) {
                    light.target_instance = None;
                }
            }
            self.ground_plane.generated = false;
            self.ground_plane.selected = false;
            self.lighting.touch();
            self.editor.status = "Generated geometry deleted from outliner".to_owned();
        }
        if let Some(light_id) = delete_light {
            self.lighting.lights.retain(|light| light.id != light_id);
            if self.lighting.selected_light == Some(light_id) {
                self.lighting.selected_light = None;
            }
            self.lighting.touch();
            self.editor.status = "Light deleted from outliner".to_owned();
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Some((tile, path, kind)) = texture_import {
            self.import_group_texture(context, tile, path, kind);
        }
        if let Some((tile, kind)) = texture_reset {
            self.reset_group_texture(tile, kind);
        }
    }

    fn viewport_settings_contents(&mut self, ui: &mut egui::Ui) {
        ui.small("Changes are saved automatically and kept across launches.");
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
        ui.heading("Transform Gizmo");
        if ui
            .checkbox(
                &mut self.gizmo_always_at_bottom,
                "Always snap gizmo to object bottom",
            )
            .changed()
        {
            self.editor.status = if self.gizmo_always_at_bottom {
                "Gizmo will stay at the selected object bottom".to_owned()
            } else {
                "Automatic bottom snapping disabled".to_owned()
            };
        }

        ui.separator();
        ui.heading("Generated Geometry");
        let topology_changed = ui
            .horizontal(|ui| {
                ui.label("Topology");
                let mut changed = ui
                    .selectable_value(
                        &mut self.ground_plane.triangulate_subdivision,
                        false,
                        "Quads",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.ground_plane.triangulate_subdivision,
                        true,
                        "Triangles",
                    )
                    .changed();
                changed
            })
            .inner;
        if topology_changed {
            let triangulate = self.ground_plane.triangulate_subdivision;
            self.ground_plane.rebuild_shape(&self.device);
            for object in &mut self.generated_objects {
                object.triangulate_subdivision = triangulate;
                object.rebuild_shape(&self.device);
            }
            self.editor.status = format!(
                "Generated geometry topology set to {}",
                if self.ground_plane.triangulate_subdivision {
                    "Triangles"
                } else {
                    "Quads"
                }
            );
        }
        ui.small("This setting applies to existing and future generated geometry.");

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
        self.capture_user_preferences();
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Reset settings to default:");
            if ui.button("Reset").clicked() {
                self.preferences = user_data::Preferences::default();
                self.apply_user_preferences();
                #[cfg(not(target_arch = "wasm32"))]
                self.preference_store.save(&self.preferences, true);
                self.editor.status = "Settings reset to defaults. User materials are unchanged.".into();
            }
        });
        ui.small("Reset affects preferences only. Your user library and scene materials are kept.");
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(error) = self.preference_store.error.clone() {
            ui.colored_label(ui.visuals().warn_fg_color, error);
            if ui.button("Retry saving settings").clicked() { self.preference_store.save(&self.preferences, true); }
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
        self.editor.hdri_path = path.display().to_string();
        self.editor.hdri_image_disabled = false;
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

        if let Some(path) = Self::explorer_drop(&drop_zone.response, &["hdr", "exr", "rat"]) {
            self.queue_hdri(path);
        }

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
                self.editor.hdri_path.clear();
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
            match self.cached_material_texture(&path, true) {
                Ok(texture) => {
                    self.pbr_material
                        .set_base_color_texture(&self.device, texture);
                    self.editor.base_color_path = path.display().to_string();
                    self.editor.status = format!("Loaded base color: {}", path.display());
                }
                Err(error) => self.editor.status = format!("Could not load texture: {error:#}"),
            }
        }
        if let Some(path) = self.editor.pending_normal.take() {
            match self.cached_material_texture(&path, false) {
                Ok(texture) => {
                    self.pbr_material.set_normal_texture(&self.device, texture);
                    shared_maps_changed = true;
                    self.editor.normal_path = path.display().to_string();
                    self.editor.status = format!("Loaded normal map: {}", path.display());
                }
                Err(error) => self.editor.status = format!("Could not load normal map: {error:#}"),
            }
        }
        if let Some(path) = self.editor.pending_metallic_roughness.take() {
            match self.cached_material_texture(&path, false) {
                Ok(texture) => {
                    self.pbr_material
                        .set_metallic_roughness_texture(&self.device, texture);
                    shared_maps_changed = true;
                    self.editor.metallic_roughness_path = path.display().to_string();
                    self.editor.status = format!("Loaded metallic/roughness: {}", path.display())
                }
                Err(error) => {
                    self.editor.status = format!("Could not load metallic/roughness map: {error:#}")
                }
            }
        }
        if let Some(path) = self.editor.pending_emissive.take() {
            match self.cached_material_texture(&path, true) {
                Ok(texture) => {
                    if self.pbr_material.uniform.color_adjustments[2] == 0.0 && self.pbr_material.uniform.options[1] == 0.0 {
                        self.pbr_material.uniform.options[1] = 1.0;
                    }
                    self.pbr_material.set_emissive_texture(&self.device, texture);
                    shared_maps_changed = true;
                    self.editor.emissive_path = path.display().to_string();
                    self.editor.status = format!("Loaded emissive: {}", path.display())
                }
                Err(error) => {
                    self.editor.status = format!("Could not load emissive map: {error:#}")
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
        if let Some(path) = self.editor.pending_asset.take() {
            let loading_fx_project = self.pending_project.as_ref()
                .and_then(|project| project.imported_model.as_ref())
                .is_some_and(|model| std::path::Path::new(&model.path) == path);
            let time_code = if loading_fx_project {
                self.pending_project.as_ref().and_then(|project| project.imported_model.as_ref()).and_then(|model| model.usd_time_code)
            } else {
                self.pending_project = None;
                (!self.editor.usd_use_stage_start).then_some(self.editor.usd_time_code)
            };
            let (sender, receiver) = std::sync::mpsc::channel();
            let worker_path = path.clone();
            std::thread::spawn(move || {
                let result = resources::prepare_model_from_path(&worker_path, time_code)
                    .map_err(|error| format!("{error:#}"));
                let _ = sender.send(result);
            });
            self.editor.status = format!("Loading {}…", path.display());
            self.pending_model_load = Some(PendingModelLoad { path, time_code, loading_fx_project, receiver });
            return;
        }
        let Some(pending) = &self.pending_model_load else { return; };
        let result = match pending.receiver.try_recv() {
            Ok(result) => result,
            Err(std::sync::mpsc::TryRecvError::Empty) => { self.window.request_redraw(); return; }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("Model import worker stopped unexpectedly".into()),
        };
        let pending = self.pending_model_load.take().unwrap();
        let path = pending.path;
        let loading_fx_project = pending.loading_fx_project;
        let loaded_model = match result.and_then(|prepared| resources::upload_prepared_model(prepared, &path.to_string_lossy(), &self.device).map_err(|error| format!("{error:#}"))) {
            Ok(model) if !model.meshes.is_empty() || model.imported_scene.is_some() => model,
            Ok(_) => {
                self.editor.status = "Imported model contains no surface meshes".to_owned();
                self.pending_project = None;
                return;
            }
            Err(error) => {
                self.editor.status = format!("Could not import model: {error}");
                self.pending_project = None;
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
            self.editor.hdri_path.clone(),
            self.editor.hdri_image_disabled,
            self.editor.hdri_intensity,
            self.editor.hdri_exposure,
            self.editor.hdri_rotation,
        );
        let mut default_editor = editor_ui::EditorUi::default();
        default_editor.usd_use_stage_start = pending.time_code.is_none();
        default_editor.usd_time_code = pending.time_code.unwrap_or(0.0);
        default_editor.pending_hdri = lighting_editor_state.0;
        default_editor.hdri_name = lighting_editor_state.1;
        default_editor.hdri_path = lighting_editor_state.2;
        default_editor.hdri_image_disabled = lighting_editor_state.3;
        default_editor.hdri_intensity = lighting_editor_state.4;
        default_editor.hdri_exposure = lighting_editor_state.5;
        default_editor.hdri_rotation = lighting_editor_state.6;
        default_editor.asset_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Model")
            .to_owned();
        default_editor.model_path_input = path.display().to_string();
        default_editor.status = format!("Imported model: {}", path.display());

        self.deconstruction = demos::Deconstruction::default();
        self.demo_buffers.clear();
        self.part_selection = Default::default();
        self.obj_model = loaded_model;
        self.mesh_material_assignments =
            vec![material_library::MaterialId(0); self.obj_model.meshes.len()];
        self.face_material_assignments = vec![Vec::new(); self.obj_model.meshes.len()];
        self.uv_space_cache.clear();
        self.pending_uv_textures.clear();
        // An FX project restores its explicit material assignments immediately
        // after this import. Do not also queue the model's MTL textures: that
        // decoded and uploaded the same images twice and could race the saved
        // project assignments.
        self.pending_model_material_root = (!loading_fx_project).then(|| {
            path.parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf()
        });
        self.pbr_material = default_material;
        self.material_graph = material_graph::MaterialGraphEditor::default();
        let imported_instance = Instance {
            position: cgmath::Vector3::new(0.0, 0.0, 0.0),
            rotation_degrees: cgmath::Vector3::new(0.0, 0.0, 0.0),
            scale: cgmath::Vector3::new(1.0, 1.0, 1.0),
            global_scale: 1.0,
        };
        self.instances = vec![imported_instance.clone()];
        self.initial_instances = vec![imported_instance];
        self.gizmo_at_bottom_by_instance = vec![false];
        default_editor.visible_instance_count = 1;
        self.editor = default_editor;
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
        self.geo_emissive_paths.clear();
        self.geo_normal_textures.clear();
        self.geo_roughness_textures.clear();
        self.geo_emissive_textures.clear();
        self.geo_material_bind_groups.clear();
        self.geo_material_uniform_buffers.clear();
        self.geo_texture_scales.clear();
        self.geo_texture_colors.clear();
        self.geo_normal_strengths.clear();
        self.geo_color_settings.clear();
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
        self.apply_user_preferences();
        self.frame_all_instances();
        self.window.request_redraw();
    }

    #[cfg(target_arch = "wasm32")]
    fn load_pending_asset(&mut self) {
        if self.editor.pending_asset.take().is_some() {
            self.editor.status =
                "Local model-path import is unavailable in the web build".to_owned();
        }
    }

    fn apply_fx_material_uniform(uniform: &mut material::MaterialUniform, saved: &FxMaterial) {
        uniform.transparency = saved.transparency;
        uniform.base_color = saved.base_color;
        uniform.color_adjustments = saved.color_adjustments;
        uniform.emissive_color = saved.emissive_color;
        uniform.properties = saved.properties;
        uniform.options = saved.options;
        uniform.inspection = saved.inspection;
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn cached_material_texture(
        &mut self,
        path: &std::path::Path,
        srgb: bool,
    ) -> anyhow::Result<Arc<texture::Texture>> {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let metadata = std::fs::metadata(path)?;
        let modified_ns = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos());
        let key = (canonical.clone(), srgb, metadata.len(), modified_ns);
        if let Some(texture) = self.material_texture_cache.get(&key) {
            return Ok(Arc::clone(texture));
        }
        // Discard older revisions of this same path without affecting other
        // cached assets. This keeps live texture replacement correct.
        self.material_texture_cache
            .retain(|(cached_path, cached_srgb, _, _), _| {
                cached_path != &canonical || *cached_srgb != srgb
            });
        let started = Instant::now();
        let texture = material::load_shared_texture(&self.device, &self.queue, path, srgb)?;
        log::debug!(
            "FX texture {:>7.3}s  {}",
            started.elapsed().as_secs_f64(),
            path.display()
        );
        self.material_texture_cache
            .insert(key, Arc::clone(&texture));
        Ok(texture)
    }

    fn apply_pending_fx_project(&mut self, context: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.pending_model_load.is_some() || self.editor.pending_asset.is_some() { return; }
        let Some(project) = self.pending_project.take() else {
            return;
        };

        if !project.material_library.is_empty() {
            self.material_library.clear();
            self.material_previews.clear();
            for saved in &project.material_library {
                let Ok(mut gpu_material) =
                    material::PbrMaterial::new_untextured(&self.device, &self.queue)
                else {
                    continue;
                };
                Self::apply_fx_material_uniform(&mut gpu_material.uniform, &saved.material);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if !saved.material.base_color_path.is_empty()
                        && let Ok(texture) = self.cached_material_texture(
                            std::path::Path::new(&saved.material.base_color_path),
                            true,
                        )
                    {
                        gpu_material.set_base_color_texture(&self.device, texture);
                    }
                    if !saved.material.normal_path.is_empty()
                        && let Ok(texture) = self.cached_material_texture(
                            std::path::Path::new(&saved.material.normal_path),
                            false,
                        )
                    {
                        gpu_material.set_normal_texture(&self.device, texture);
                    }
                    if !saved.material.roughness_path.is_empty()
                        && let Ok(texture) = self.cached_material_texture(
                            std::path::Path::new(&saved.material.roughness_path),
                            false,
                        )
                    {
                        gpu_material.set_metallic_roughness_texture(&self.device, texture);
                    }
                    if !saved.material.emissive_path.is_empty()
                        && let Ok(texture) = self.cached_material_texture(
                            std::path::Path::new(&saved.material.emissive_path),
                            true,
                        )
                    {
                        gpu_material.set_emissive_texture(&self.device, texture);
                    }
                }
                self.material_library.push(material_library::SceneMaterial {
                    id: material_library::MaterialId(saved.id),
                    name: saved.name.clone(),
                    material: gpu_material,
                    base_color_path: saved.material.base_color_path.clone(),
                    normal_path: saved.material.normal_path.clone(),
                    roughness_path: saved.material.roughness_path.clone(),
                    emissive_path: saved.material.emissive_path.clone(),
                });
            }
            self.next_material_id = self
                .material_library
                .iter()
                .map(|entry| entry.id.0)
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            self.selected_library_material = self
                .material_library
                .first()
                .map(|entry| entry.id)
                .unwrap_or(material_library::MaterialId(1));
        }
        self.mesh_material_assignments = project
            .mesh_material_assignments
            .iter()
            .map(|id| material_library::MaterialId(*id))
            .collect();
        self.mesh_material_assignments
            .resize(self.obj_model.meshes.len(), material_library::MaterialId(0));
        self.face_material_assignments = project.face_material_assignments.clone();
        self.face_material_assignments
            .resize_with(self.obj_model.meshes.len(), Vec::new);
        self.generated_material_assignments = project
            .generated_material_assignments
            .iter()
            .map(|(object, material)| (*object, material_library::MaterialId(*material)))
            .collect();

        if let Some(imported) = project.imported_model {
            if !self.obj_model.meshes.is_empty() || self.obj_model.imported_scene.is_some() {
                self.instances = imported
                    .instances
                    .into_iter()
                    .map(|transform| Instance {
                        position: transform.position.into(),
                        rotation_degrees: transform.rotation_degrees.into(),
                        scale: transform.scale.into(),
                        global_scale: transform.global_scale,
                    })
                    .collect();
                self.initial_instances = self.instances.clone();
                self.editor.visible_instance_count =
                    imported.visible_instance_count.min(self.instances.len());
                self.editor.selected_instance = imported
                    .selected_instance
                    .min(self.instances.len().saturating_sub(1));
                self.gizmo_at_bottom_by_instance = imported.gizmo_at_bottom;
                self.gizmo_at_bottom_by_instance
                    .resize(self.instances.len(), false);
                self.geo_group_enabled = imported.group_visibility;
                self.geo_group_enabled
                    .resize(self.obj_model.meshes.len(), true);
                self.geo_texture_enabled = imported.group_texture_enabled;
                self.geo_texture_enabled
                    .resize(self.obj_model.meshes.len(), true);
                #[cfg(not(target_arch = "wasm32"))]
                for assignment in imported.textures {
                    if let Some(colors) = assignment.colors {
                        self.geo_color_settings.insert(assignment.tile, colors);
                    }
                    self.rebuild_group_material_bind_group(assignment.tile);
                    self.geo_texture_scales
                        .insert(assignment.tile, assignment.scale.clamp(0.05, 100.0));
                    self.geo_texture_colors
                        .insert(assignment.tile, assignment.color);
                    self.geo_normal_strengths
                        .insert(assignment.tile, assignment.normal_strength.clamp(0.0, 2.0));
                    for (path, kind) in [
                        (assignment.base_color, GroupTextureKind::BaseColor),
                        (assignment.normal, GroupTextureKind::Normal),
                        (assignment.roughness, GroupTextureKind::Roughness),
                        (assignment.emissive, GroupTextureKind::Emissive),
                    ] {
                        if !path.is_empty() && !path.starts_with("MTL:") {
                            self.queue_group_texture(
                                context,
                                assignment.tile,
                                std::path::PathBuf::from(path),
                                kind,
                                false,
                            );
                        }
                    }
                }
                self.instance_buffer_dirty = true;
            }
        } else if !self.obj_model.meshes.is_empty() {
            self.clear_imported_model();
        }

        self.generated_objects.clear();
        let mut restored = Vec::new();
        for saved in project.generated_objects {
            let mut object =
                ground_plane::GroundPlane::new_instance_from(&self.device, &self.ground_plane);
            object.scene_id = saved.scene_id;
            object.generated = true;
            object.selected = saved.selected;
            object.kind = saved.kind;
            object.visible = saved.visible;
            object.snap_bottom_to_grid = saved.snap_bottom_to_grid;
            object.wireframe = saved.wireframe;
            object.hide_surface = saved.hide_surface;
            object.size = saved.size;
            object.height = saved.height;
            object.position_x = saved.position_x;
            object.position_y = saved.position_y;
            object.rotation_degrees = saved.rotation_degrees;
            object.shape_height = saved.shape_height;
            object.thickness = saved.thickness;
            object.subdivisions = saved.subdivisions;
            object.triangulate_subdivision = saved.triangles;
            object.uv_scale = saved.uv_scale;
            Self::apply_fx_material_uniform(&mut object.material.uniform, &saved.material);
            if !saved.material.base_color_path.is_empty() {
                let path = std::path::PathBuf::from(&saved.material.base_color_path);
                if let Ok(texture) = self.cached_material_texture(&path, true) {
                    object
                        .material
                        .set_base_color_texture(&self.device, texture);
                    object.base_color_path = saved.material.base_color_path.clone();
                }
            }
            if !saved.material.normal_path.is_empty() {
                let path = std::path::PathBuf::from(&saved.material.normal_path);
                if let Ok(texture) = self.cached_material_texture(&path, false) {
                    object.material.set_normal_texture(&self.device, texture);
                    object.normal_path = saved.material.normal_path.clone();
                }
            }
            if !saved.material.roughness_path.is_empty() {
                let path = std::path::PathBuf::from(&saved.material.roughness_path);
                if let Ok(texture) = self.cached_material_texture(&path, false) {
                    object
                        .material
                        .set_metallic_roughness_texture(&self.device, texture);
                    object.roughness_path = saved.material.roughness_path.clone();
                }
            }
            if !saved.material.emissive_path.is_empty() {
                let path = std::path::PathBuf::from(&saved.material.emissive_path);
                if let Ok(texture) = self.cached_material_texture(&path, true) {
                    object
                        .material
                        .set_emissive_texture(&self.device, texture);
                    object.emissive_path = saved.material.emissive_path.clone();
                }
            }
            let mesh_key = (
                object.kind,
                object.subdivisions,
                object.triangulate_subdivision,
            );
            if let Some(mesh) = self.generated_mesh_cache.get(&mesh_key) {
                object.use_shared_mesh(mesh);
            } else {
                object.rebuild_shape(&self.device);
                self.generated_mesh_cache
                    .insert(mesh_key, object.shared_mesh());
            }
            object.upload(&self.queue);
            restored.push(object);
        }
        let selected = restored.iter().position(|object| object.selected);
        if let Some(index) = selected.or_else(|| (!restored.is_empty()).then_some(0)) {
            self.ground_plane = restored.swap_remove(index);
            self.generated_objects = restored;
        } else if let Ok(object) = ground_plane::GroundPlane::new(&self.device, &self.queue) {
            self.ground_plane = object;
        }
        self.next_generated_id = self
            .generated_objects
            .iter()
            .chain(std::iter::once(&self.ground_plane))
            .map(|object| object.scene_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        Self::apply_fx_material_uniform(&mut self.pbr_material.uniform, &project.material);
        self.material_graph = material_graph::MaterialGraphEditor::default();
        self.material_graph.graph = project.material_graph;
        self.editor.base_color_path = project.material.base_color_path.clone();
        self.editor.normal_path = project.material.normal_path.clone();
        self.editor.metallic_roughness_path = project.material.roughness_path.clone();
        self.editor.emissive_path = project.material.emissive_path.clone();
        self.editor.pending_texture = (!project.material.base_color_path.is_empty())
            .then(|| std::path::PathBuf::from(project.material.base_color_path));
        self.editor.pending_normal = (!project.material.normal_path.is_empty())
            .then(|| std::path::PathBuf::from(project.material.normal_path));
        self.editor.pending_metallic_roughness = (!project.material.roughness_path.is_empty())
            .then(|| std::path::PathBuf::from(project.material.roughness_path));
        self.editor.pending_emissive = (!project.material.emissive_path.is_empty()).then(|| std::path::PathBuf::from(project.material.emissive_path));

        self.lighting.restore_project(
            project.lights.mode,
            project.lights.lights,
            project.lights.selected_light,
            project.lights.area_samples,
            project.lights.environment_samples,
        );
        self.editor.show_grid = project.grid.visible;
        self.editor.grid_size = project.grid.size;
        self.editor.grid_spacing = project.grid.spacing;
        self.editor.grid_color = project.grid.color;
        self.displayed_grid_spacing = 0.0;
        self.displayed_grid_extent = 0.0;
        self.grid.rebuild(
            &self.device,
            project.grid.size,
            project.grid.spacing,
            project.grid.color,
        );
        self.editor.background = project.viewport.background;
        self.background_color = wgpu::Color {
            r: project.viewport.background[0] as f64,
            g: project.viewport.background[1] as f64,
            b: project.viewport.background[2] as f64,
            a: project.viewport.background[3] as f64,
        };
        self.gizmo_always_at_bottom = project.viewport.gizmo_always_at_bottom;
        self.model_view_mode = if project.viewport.wireframe {
            ModelViewMode::Wireframe
        } else {
            ModelViewMode::Solid
        };
        self.show_points = project.viewport.show_points;
        self.point_size = project.viewport.point_size;
        self.point_color = project.viewport.point_color;
        self.show_normals = project.viewport.show_normals;
        self.normal_mode = match project.viewport.normal_mode {
            1 => NormalDisplayMode::Point,
            2 => NormalDisplayMode::Face,
            _ => NormalDisplayMode::Vertex,
        };
        self.normal_length = project.viewport.normal_length;
        self.normal_color = project.viewport.normal_color;
        self.show_uv_map = project.viewport.show_uv_map;
        self.show_uv_overlay = project.viewport.show_uv_overlay;
        self.show_uv_texture = project.viewport.show_uv_texture;
        self.show_uv_lines = project.viewport.show_uv_lines;
        self.camera.eye = project.viewport.camera_eye.into();
        self.camera.target = project.viewport.camera_target.into();
        self.camera.up = project.viewport.camera_up.into();
        self.camera.fovy = project.viewport.camera_fovy;
        self.camera.projection_mode = if project.viewport.orthographic {
            ProjectionMode::Orthographic
        } else {
            ProjectionMode::Perspective
        };
        self.camera.ortho_scale = project.viewport.ortho_scale;

        let environment_unchanged = self.environment.is_some()
            && self.editor.hdri_path == project.environment.path
            && !project.environment.path.is_empty();
        self.editor.hdri_path = project.environment.path.clone();
        self.editor.hdri_intensity = project.environment.intensity;
        self.editor.hdri_exposure = project.environment.exposure;
        self.editor.hdri_rotation = project.environment.rotation;
        if !project.environment.path.is_empty() && !environment_unchanged {
            self.queue_hdri(std::path::PathBuf::from(project.environment.path));
        } else {
            self.environment = None;
            let (texture, bind_group) = environment::fallback_lighting_bind_group(
                &self.device,
                &self.queue,
                &self.environment_lighting_layout,
            );
            self._fallback_environment_texture = texture;
            self.environment_lighting_bind_group = bind_group;
            self.editor.hdri_name = None;
        }
        self.deconstruction = project.deconstruction;
        self.deconstruction.sanitize();
        self.editor.hdri_image_disabled = project.environment.disabled;
        self.apply_user_preferences();
        self.undo_stack.clear();
        self.undo_last_snapshot = None;
        self.undo_transaction_active = false;
        self.editor.status = "FX project loaded".to_owned();
        if let Some(started) = self.fx_load_started.take() {
            let elapsed = started.elapsed();
            self.editor.status = format!("FX project loaded in {:.2}s", elapsed.as_secs_f64());
            log::info!("FX project loaded in {:.3}s", elapsed.as_secs_f64());
        }
        self.window.request_redraw();
    }

    fn surface_draws(&self) -> Vec<model::SurfaceDraw<'_>> {
        let mut draws = Vec::new();
        let forward = (self.camera.target - self.camera.eye).normalize();
        let depth = |point: cgmath::Vector4<f32>| {
            (cgmath::Point3::new(point.x, point.y, point.z) - self.camera.eye).dot(forward)
        };
        for object in self
            .generated_objects
            .iter()
            .chain(std::iter::once(&self.ground_plane))
        {
            let material = self
                .generated_material_assignments
                .get(&object.scene_id)
                .and_then(|id| self.material_library.iter().find(|entry| entry.id == *id))
                .map(|entry| &entry.material)
                .unwrap_or(&object.material);
            let center = object.model_matrix() * cgmath::Vector4::new(0.0, 0.0, 0.0, 1.0);
            if let Some(draw) = object.surface_draw(
                &material.bind_group,
                depth(center),
                material.needs_alpha_blend(),
            ) {
                draws.push(draw);
            }
        }
        for (index, mesh) in self.obj_model.meshes.iter().enumerate() {
            if !self.geo_group_enabled.get(index).copied().unwrap_or(true) {
                continue;
            }
            let fallback = self
                .geo_material_bind_groups
                .get(&mesh.uv_tile)
                .filter(|_| self.geo_texture_enabled.get(index).copied().unwrap_or(true))
                .unwrap_or(&self.pbr_material.bind_group);
            let mut fallback_uniform = self.pbr_material.uniform;
            let mut fallback_blend = self.pbr_material.needs_alpha_blend();
            if self.geo_material_bind_groups.contains_key(&mesh.uv_tile)
                && self.geo_texture_enabled.get(index).copied().unwrap_or(true)
            {
                fallback_uniform.base_color = self
                    .geo_texture_colors
                    .get(&mesh.uv_tile)
                    .copied()
                    .unwrap_or(fallback_uniform.base_color);
                self.apply_group_colors(mesh.uv_tile, &mut fallback_uniform);
                let has_alpha = self
                    .uv_space_textures
                    .get(&mesh.uv_tile)
                    .map_or(false, |texture| texture._viewport_texture.has_transparency);
                // Group bindings use the global base texture if no group map is set.
                fallback_blend = fallback_uniform.needs_alpha_blend(has_alpha)
                    || (self.uv_space_textures.get(&mesh.uv_tile).is_none()
                        && fallback_uniform
                            .needs_alpha_blend(self.pbr_material.base_texture_has_alpha()));
            }
            let base = self
                .mesh_material_assignments
                .get(index)
                .copied()
                .filter(|id| id.0 != 0);
            let overrides = self
                .face_material_assignments
                .get(index)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let center = cgmath::Vector4::new(
                (mesh.bounds_min[0] + mesh.bounds_max[0]) * 0.5,
                (mesh.bounds_min[1] + mesh.bounds_max[1]) * 0.5,
                (mesh.bounds_min[2] + mesh.bounds_max[2]) * 0.5,
                1.0,
            );
            for (first, end, id) in material_library::resolved_face_ranges(mesh.num_elements / 3, base, overrides) {
                let assigned_material = id
                    .and_then(|id| self.material_library.iter().find(|entry| entry.id == id))
                    .map(|entry| &entry.material);
                let material = assigned_material.map(|m| &m.bind_group).unwrap_or(fallback);
                let blend = assigned_material.map_or(fallback_blend, |m| m.needs_alpha_blend());
                for (instance_index, instance) in self
                    .instances
                    .iter()
                    .take(self.editor.visible_instance_count)
                    .enumerate()
                {
                    let matrix: cgmath::Matrix4<f32> =
                        self.demo_instance_raw(instance, index).model.into();
                    draws.push(model::SurfaceDraw {
                        vertices: &mesh.vertex_buffer,
                        indices: &mesh.index_buffer,
                        instances: self.demo_buffer(index),
                        material,
                        index_range: first * 3..end * 3,
                        instance_range: instance_index as u32..instance_index as u32 + 1,
                        depth: depth(matrix * center),
                        blend,
                    });
                }
            }
        }
        // Opaque surfaces write depth first; fractional-alpha surfaces blend
        // back to front without writing depth. Demo offsets and instances count.
        draws.sort_by(|a, b| b.depth.total_cmp(&a.depth));
        draws
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
        self.apply_pending_fx_project(&context);
        self.load_pending_hdri();
        self.load_pending_texture();
        self.update_demo_buffers();
        self.update_camera_clip_planes();
        let pointer_down = context.input(|input| input.pointer.primary_down());
        #[cfg(not(target_arch = "wasm32"))]
        if !pointer_down { self.preference_store.save(&self.preferences, false); }
        self.record_undo_state(pointer_down);

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
        self.update_grid_camera();
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
        for (tile, buffer) in &self.geo_material_uniform_buffers {
            let mut uniform = self.pbr_material.uniform;
            uniform.inspection[2] = self
                .geo_texture_scales
                .get(tile)
                .copied()
                .unwrap_or(1.0)
                .clamp(0.05, 100.0);
            uniform.base_color = self
                .geo_texture_colors
                .get(tile)
                .copied()
                .unwrap_or(self.pbr_material.uniform.base_color);
            uniform.properties[2] = self
                .geo_normal_strengths
                .get(tile)
                .copied()
                .unwrap_or(self.pbr_material.uniform.properties[2])
                .clamp(0.0, 2.0);
            self.apply_group_colors(*tile, &mut uniform);
            self.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
        }
        self.shadow_renderer
            .prepare(&self.queue, &self.lighting, &self.camera);
        self.lighting_gpu.upload(
            &self.queue,
            &self.lighting,
            &self.camera,
            &self.shadow_renderer,
        );
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

        // Inspection overlays can dwarf the surface mesh. Upload their GPU
        // buffers only if the saved project actually displays them.
        if self.show_points {
            for mesh in &mut self.obj_model.meshes {
                mesh.ensure_point_markers_uploaded(&self.device);
            }
        }
        if self.show_normals {
            for mesh in &mut self.obj_model.meshes {
                mesh.ensure_normal_markers_uploaded(&self.device);
            }
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
        if self.active_side_panel == Some(5) {
            for entry in &self.material_library {
                if let Some(preview) = self.material_previews.get(&entry.id) {
                    preview.render(
                        &mut encoder,
                        &entry.material.bind_group,
                        &self.environment_lighting_bind_group,
                    );
                }
            }
        }

        let surface_draws = self.surface_draws();
        self.shadow_renderer.render(&mut encoder, &surface_draws);

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
                self.grid
                    .draw(&mut render_pass, &self.grid_camera_bind_group);
            }
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(2, &self.environment_lighting_bind_group, &[]);
            render_pass.set_bind_group(3, &self.lighting_gpu.bind_group, &[]);
            render_pass.set_pipeline(&self.render_pipeline);
            for draw in &surface_draws { draw.draw(&mut render_pass, 0); }
            render_pass.set_pipeline(&self.transparent_pipeline);
            for draw in surface_draws.iter().filter(|draw| draw.blend) { draw.draw(&mut render_pass, 0); }
            self.draw_part_highlight(&mut render_pass);
            // Draw generated-object wire overlays after both surface passes.
            for object in self.generated_objects.iter().chain(std::iter::once(&self.ground_plane)) {
                object.draw(&mut render_pass, &self.render_pipeline,
                    self.wireframe_pipeline.as_ref(), &self.quad_line_pipeline,
                    &self.camera_bind_group, &self.environment_lighting_bind_group,
                    &self.lighting_gpu.bind_group, None, false);
            }
            use model::DrawModel;

            if self.model_view_mode == ModelViewMode::Wireframe {
                if let Some(wireframe_pipeline) = self.wireframe_pipeline.as_ref() {
                    render_pass.set_pipeline(wireframe_pipeline);
                    render_pass.set_bind_group(0, &self.wireframe_material.bind_group, &[]);
                    for (mesh_index, mesh) in self.obj_model.meshes.iter().enumerate() {
                        render_pass.set_vertex_buffer(1, self.demo_buffer(mesh_index).slice(..));
                        if !self
                            .geo_group_enabled
                            .get(mesh_index)
                            .copied()
                            .unwrap_or(true)
                        {
                            continue;
                        }
                        render_pass.draw_mesh_instanced(
                            mesh,
                            0..self.editor.visible_instance_count as u32,
                        );
                    }
                }
            }

            if self.show_points {
                render_pass.set_pipeline(&self.point_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &self.point_bind_group, &[]);
                let stride = std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress;
                for (mesh_index, mesh) in self.obj_model.meshes.iter().enumerate() {
                    render_pass.set_vertex_buffer(1, self.demo_buffer(mesh_index).slice(..));
                    if !self
                        .geo_group_enabled
                        .get(mesh_index)
                        .copied()
                        .unwrap_or(true)
                    {
                        continue;
                    }
                    let Some(buffer) = mesh.point_marker_buffer.as_ref() else {
                        continue;
                    };
                    render_pass.set_vertex_buffer(0, buffer.slice(..));
                    for instance in 0..self.editor.visible_instance_count as wgpu::BufferAddress {
                        let start = instance * stride;
                        render_pass.set_vertex_buffer(
                            1,
                            self.demo_buffer(mesh_index).slice(start..start + stride),
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
                    render_pass.set_vertex_buffer(1, self.demo_buffer(mesh_index).slice(..));
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
                            (mesh.vertex_normal_buffer.as_ref(), mesh.vertex_normal_count)
                        }
                        NormalDisplayMode::Point => {
                            (mesh.point_normal_buffer.as_ref(), mesh.point_normal_count)
                        }
                        NormalDisplayMode::Face => {
                            (mesh.face_normal_buffer.as_ref(), mesh.face_normal_count)
                        }
                    };
                    let Some(buffer) = buffer else {
                        continue;
                    };
                    render_pass.set_vertex_buffer(0, buffer.slice(..));
                    for instance in 0..self.editor.visible_instance_count as wgpu::BufferAddress {
                        let start = instance * stride;
                        render_pass.set_vertex_buffer(
                            1,
                            self.demo_buffer(mesh_index).slice(start..start + stride),
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
    #[cfg(not(target_arch = "wasm32"))]
    startup_project: Option<std::path::PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
fn pick_material_texture() -> Option<std::path::PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Texture image", &["png", "jpg", "jpeg", "hdr", "exr"])
        .pick_file()
}

#[cfg(not(target_arch = "wasm32"))]
fn pick_ground_texture() -> Option<std::path::PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Texture image", &["png", "jpg", "jpeg", "hdr", "exr"])
        .pick_file()
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(not(target_arch = "wasm32"))]
            startup_project: std::env::args_os().nth(1).map(std::path::PathBuf::from),
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
            let mut state = pollster::block_on(State::new(window)).unwrap();
            if let Some(path) = self.startup_project.take()
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("fx"))
            {
                state.queue_fx_project(path);
            }
            self.state = Some(state);
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

        if let WindowEvent::ModifiersChanged(modifiers) = &event {
            #[cfg(target_os = "macos")]
            {
                state.undo_modifier_held = modifiers.state().super_key();
            }
            #[cfg(not(target_os = "macos"))]
            {
                state.undo_modifier_held = modifiers.state().control_key();
            }
        }

        // Tab belongs exclusively to the viewport generation menu. Intercept it
        // before egui receives the event so it cannot advance widget focus.
        let tab_pressed = matches!(
            &event,
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::Tab),
                    state: ElementState::Pressed,
                    repeat: false,
                    ..
                },
                ..
            }
        );
        if tab_pressed {
            if !state.egui.context.is_pointer_over_egui() {
                if let Some(position) = state.egui.context.pointer_hover_pos() {
                    state.generate_popup_position = position;
                }
                state.generate_popup_open = true;
            }
            state.window.request_redraw();
            return;
        }

        let undo_pressed = state.undo_modifier_held
            && matches!(
                &event,
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyZ),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                    ..
                }
            );
        if undo_pressed {
            state.undo_last_action();
            return;
        }

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
                state.undo_modifier_held = false;
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

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyG),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } if !state.egui.context.egui_wants_keyboard_input()
                && !state.houdini_navigation.view_mode_active()
                && !state.houdini_navigation.control_held
                && !state.undo_modifier_held =>
            {
                state.editor.show_grid = !state.editor.show_grid;
                state.window.request_redraw();
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::KeyT),
                    state: ElementState::Pressed, repeat: false, ..
                }, ..
            } if !state.egui.context.egui_wants_keyboard_input()
                && !state.houdini_navigation.view_mode_active()
                && !state.houdini_navigation.control_held
                && !state.undo_modifier_held =>
            {
                state.toggle_part_selection();
                state.window.request_redraw();
            }

            // MouseINPUT //
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                if button_state == ElementState::Pressed {
                    if button == MouseButton::Left && !state.houdini_navigation.view_mode_active() {
                        if state.part_selection.active {
                            state.select_part_at_cursor();
                        } else {
                            state.select_instance_at_cursor();
                        }
                    }
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

#[cfg(test)]
mod fx_project_tests {
    use super::*;

    #[test]
    fn generated_light_targets_decode_without_underflow() {
        for target in [0, 1, GENERATED_SHAPE_LIGHT_TARGET_BASE - 1] {
            assert_eq!(State::generated_target_id(target), None);
        }
        assert_eq!(
            State::generated_target_id(GENERATED_SHAPE_LIGHT_TARGET_BASE),
            Some(0)
        );
        assert_eq!(
            State::generated_target_id(State::generated_target_index(42)),
            Some(42)
        );
        assert_eq!(
            State::generated_target_id(usize::MAX),
            Some((usize::MAX - GENERATED_SHAPE_LIGHT_TARGET_BASE) as u64)
        );
    }

    #[test]
    fn old_materials_and_group_assignments_keep_compatible_defaults() {
        let saved: FxMaterial = serde_json::from_value(serde_json::json!({
            "base_color": [1.0, 1.0, 1.0, 1.0], "properties": [0.0, 0.5, 1.0, 1.0],
            "options": [0.0, 2.0, 1.0, 0.0], "inspection": [0.0, 0.0, 0.0, 0.0],
            "base_color_path": "", "normal_path": "", "roughness_path": ""
        }))
        .unwrap();
        assert_eq!(saved.transparency, material::default_transparency());
        assert_eq!(saved.color_adjustments, [0.0; 4]);
        assert_eq!(saved.emissive_color, [1.0; 4]);
        assert_eq!(saved.options[1], 2.0);
        assert!(saved.emissive_path.is_empty());
        let group: FxGroupTextures = serde_json::from_value(serde_json::json!({
            "tile": [0, 0], "base_color": "", "normal": "", "roughness": ""
        }))
        .unwrap();
        assert!(
            group.colors.is_none(),
            "Legacy groups inherit the scene color controls"
        );
        assert!(group.emissive.is_empty());
    }

    #[test]
    fn group_emission_round_trip_keeps_zero_intensity_and_hue() {
        let group = FxGroupTextures {
            tile: (2, 0),
            base_color: String::new(),
            normal: String::new(),
            roughness: String::new(),
            emissive: "emission.png".into(),
            scale: 1.0,
            color: [1.0; 4],
            normal_strength: 1.0,
            colors: Some(GroupColorSettings {
                transparency: [2.0, 0.3, 0.0, 0.4],
                double_sided: true,
                hue: 45.0,
                emission_hue: -120.0,
                intensity: 0.0,
                tint: [0.2, 0.5, 1.0, 1.0],
            }),
        };
        let restored: FxGroupTextures =
            serde_json::from_str(&serde_json::to_string(&group).unwrap()).unwrap();
        assert_eq!(restored.emissive, "emission.png");
        let settings = restored.colors.unwrap();
        assert_eq!(settings.intensity, 0.0);
        assert_eq!(settings.transparency, [2.0, 0.3, 0.0, 0.4]);
        assert!(settings.double_sided);
        assert_eq!(settings.hue, 45.0);
        assert_eq!(settings.emission_hue, -120.0);
        assert_eq!(settings.tint, [0.2, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn fx_project_schema_round_trips() {
        let material = FxMaterial {
            transparency: material::default_transparency(),
            color_adjustments: [30.0, 120.0, 1.0, 0.0],
            emissive_color: [0.8, 0.2, 0.5, 1.0],            base_color: [0.2, 0.3, 0.4, 1.0],
            properties: [0.1, 0.5, 1.0, 1.0],
            options: [0.0; 4],
            inspection: [0.0; 4],
            base_color_path: "albedo.png".to_owned(),
            normal_path: String::new(),
            roughness_path: String::new(),
            emissive_path: "emission.png".into(),
        };
        let project = FxProject {
            deconstruction: demos::Deconstruction::default(),
            format: "FX Scene Project".to_owned(),
            version: FX_PROJECT_VERSION,
            imported_model: None,
            generated_objects: vec![FxGeneratedObject {
                scene_id: 7,
                selected: true,
                kind: ground_plane::BasicShape::Cube,
                visible: true,
                snap_bottom_to_grid: false,
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
                triangles: false,
                uv_scale: 1.0,
                material,
            }],
            material: FxMaterial {
            transparency: material::default_transparency(),
            color_adjustments: [0.0; 4],
            emissive_color: [1.0; 4],                base_color: [1.0; 4],
                properties: [0.0, 0.5, 1.0, 1.0],
                options: [0.0; 4],
                inspection: [0.0; 4],
                base_color_path: String::new(),
                normal_path: String::new(),
                roughness_path: String::new(),
                emissive_path: String::new(),
            },
            material_graph: material_graph::MaterialGraph::default(),
            lights: FxLighting {
                mode: lighting::ViewportLightingMode::DefaultLight,
                lights: Vec::new(),
                selected_light: None,
                area_samples: 16,
                environment_samples: 64,
            },
            grid: FxGrid {
                visible: true,
                size: 20.0,
                spacing: 1.0,
                color: [0.5; 4],
            },
            viewport: FxViewport {
                background: [0.0, 0.0, 0.0, 1.0],
                gizmo_always_at_bottom: false,
                wireframe: false,
                show_points: false,
                point_size: 7.0,
                point_color: [0.0, 0.8, 0.72, 1.0],
                show_normals: false,
                normal_mode: 0,
                normal_length: 0.01,
                normal_color: [0.0, 0.8, 0.72, 1.0],
                show_uv_map: false,
                show_uv_overlay: false,
                show_uv_texture: true,
                show_uv_lines: true,
                camera_eye: [8.0, -8.0, 6.0],
                camera_target: [0.0; 3],
                camera_up: [0.0, 0.0, 1.0],
                camera_fovy: 45.0,
                orthographic: false,
                ortho_scale: 10.0,
            },
            environment: FxEnvironment {
                path: String::new(),
                disabled: false,
                intensity: 1.0,
                exposure: 0.0,
                rotation: 0.0,
            },
            material_library: Vec::new(),
            mesh_material_assignments: Vec::new(),
            face_material_assignments: Vec::new(),
            generated_material_assignments: BTreeMap::new(),
        };
        let json = serde_json::to_string(&project).unwrap();
        let restored: FxProject = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.version, FX_PROJECT_VERSION);
        assert_eq!(restored.generated_objects[0].scene_id, 7);
        assert_eq!(
            restored.generated_objects[0].kind,
            ground_plane::BasicShape::Cube
        );
        assert_eq!(
            restored.generated_objects[0].material.base_color_path,
            "albedo.png"
        );
        assert_eq!(restored.generated_objects[0].material.emissive_path, "emission.png");
        assert_eq!(restored.generated_objects[0].material.color_adjustments, [30.0, 120.0, 1.0, 0.0]);
        assert_eq!(restored.generated_objects[0].material.emissive_color, [0.8, 0.2, 0.5, 1.0]);
        assert!(restored.material_library.is_empty());
        assert!(restored.face_material_assignments.is_empty());
        let mut legacy = serde_json::to_value(&project).unwrap();
        legacy.as_object_mut().unwrap().remove("deconstruction");
        let legacy: FxProject = serde_json::from_value(legacy).unwrap();
        assert_eq!(legacy.deconstruction, demos::Deconstruction::default());
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
