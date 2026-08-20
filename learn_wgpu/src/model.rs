use std::ops::Range;

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointMarkerVertex {
    pub center: [f32; 3],
    pub corner: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NormalMarkerVertex {
    pub origin: [f32; 3],
    pub normal: [f32; 3],
    /// Position along the arrow (0..1) and signed screen-space half-width.
    pub shape: [f32; 2],
}

impl NormalMarkerVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

impl PointMarkerVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

impl Vertex for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<MaterialSource>,
}

#[derive(Clone, Debug)]
pub struct MaterialSource {
    pub name: String,
    pub diffuse: [f32; 3],
    pub diffuse_texture: String,
    pub normal_texture: String,
    pub roughness_texture: String,
}

pub struct Mesh {
    #[allow(dead_code)]
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub point_marker_buffer: wgpu::Buffer,
    pub point_marker_count: u32,
    pub vertex_normal_buffer: wgpu::Buffer,
    pub vertex_normal_count: u32,
    pub point_normal_buffer: wgpu::Buffer,
    pub point_normal_count: u32,
    pub face_normal_buffer: wgpu::Buffer,
    pub face_normal_count: u32,
    pub uv_edges: Vec<([f32; 2], [f32; 2])>,
    pub uv_tile: (i32, i32),
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub geometry_stats: GeometryStats,
    #[allow(dead_code)]
    pub material: usize,
}

#[derive(Debug, Clone, Default)]
pub struct GeometryStats {
    /// Welded position count (UV and normal seam duplicates count once).
    pub point_count: usize,
    /// GPU vertex count (seam duplicates are included).
    pub vertex_count: usize,
    pub index_count: usize,
    pub triangle_count: usize,
    pub edge_count: usize,
    pub boundary_edge_count: usize,
    pub non_manifold_edge_count: usize,
    pub degenerate_triangle_count: usize,
    pub uv_vertex_count: usize,
    pub uv_seam_vertex_count: usize,
    pub authored_normals: bool,
    pub surface_area: f64,
    pub enclosed_volume: Option<f64>,
}

pub trait DrawModel<'a> {
    fn draw_mesh_instanced(&mut self, mesh: &'a Mesh, instances: Range<u32>);
}
impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh_instanced(&mut self, mesh: &'b Mesh, instances: Range<u32>) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }
}
