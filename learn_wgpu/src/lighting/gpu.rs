use cgmath::{Deg, InnerSpace, Matrix3, Vector3};
use wgpu::util::DeviceExt;

use super::{LightingManager, SceneLight};

pub const MAX_GPU_LIGHTS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    position_type: [f32; 4],
    direction_range: [f32; 4],
    color_power: [f32; 4],
    shape: [f32; 4],
    spot: [f32; 4],
    contribution: [f32; 4],
    shadow: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightingUniform {
    counts: [u32; 4],
    default_settings: [f32; 4],
    default_direction: [f32; 4],
    default_color: [f32; 4],
}

pub struct LightingGpu {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    light_buffer: wgpu::Buffer,
}

impl LightingGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform = LightingUniform {
            counts: [0, 0, 16, 64],
            default_settings: [0.08, 1.0, 1.0, 0.0],
            default_direction: [0.25, -0.35, -1.0, 0.0],
            default_color: [1.0, 0.97, 0.92, 3.0],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lighting uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let empty = [GpuLight::zeroed(); MAX_GPU_LIGHTS];
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene lights"),
            contents: bytemuck::cast_slice(&empty),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            layout,
            bind_group,
            uniform_buffer,
            light_buffer,
        }
    }

    pub fn upload(&self, queue: &wgpu::Queue, manager: &LightingManager, _camera: &crate::Camera) {
        let active = manager
            .active_direct_lights()
            .take(manager.max_lights.min(MAX_GPU_LIGHTS))
            .map(GpuLight::from_scene)
            .collect::<Vec<_>>();
        if !active.is_empty() {
            queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&active));
        }
        let uniform = LightingUniform {
            counts: [
                manager.effective_gpu_mode(),
                active.len() as u32,
                manager.area_samples,
                manager.environment_samples,
            ],
            default_settings: [
                0.08,
                1.0,
                1.0,
                manager.active_environment().is_some() as u32 as f32,
            ],
            // World-locked studio key: the fallback must not orbit with the
            // camera, otherwise material comparisons are not repeatable.
            default_direction: [0.35, -0.45, -0.82, 0.0],
            default_color: [1.0, 0.97, 0.92, 3.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

impl GpuLight {
    fn zeroed() -> Self {
        Self {
            position_type: [0.0; 4],
            direction_range: [0.0; 4],
            color_power: [0.0; 4],
            shape: [0.0; 4],
            spot: [0.0; 4],
            contribution: [0.0; 4],
            shadow: [0.0; 4],
        }
    }

    fn from_scene(light: &SceneLight) -> Self {
        let rotation = Matrix3::from_angle_z(Deg(light.rotation_degrees[2]))
            * Matrix3::from_angle_y(Deg(light.rotation_degrees[1]))
            * Matrix3::from_angle_x(Deg(light.rotation_degrees[0]));
        let direction = (rotation * Vector3::new(0.0, 0.0, -1.0)).normalize();
        let color = if light.use_temperature {
            kelvin_to_rgb(light.temperature_kelvin)
        } else {
            light.color
        };
        Self {
            position_type: [
                light.position[0],
                light.position[1],
                light.position[2],
                light.kind.gpu_type() as f32,
            ],
            direction_range: [direction.x, direction.y, direction.z, 100_000.0],
            color_power: [color[0], color[1], color[2], light.power()],
            shape: [
                light.radius.max(0.001),
                light.size[0].max(0.001),
                light.size[1].max(0.001),
                light.spread.clamp(0.0, 1.0),
            ],
            spot: [
                light.inner_angle_degrees.to_radians().cos(),
                light.outer_angle_degrees.to_radians().cos(),
                light.rolloff.max(0.001),
                light.single_sided as u32 as f32,
            ],
            contribution: [
                light.diffuse_contribution,
                light.specular_contribution,
                0.0,
                1.0,
            ],
            shadow: [
                light.shadows_enabled as u32 as f32,
                light.shadow_softness,
                -1.0,
                0.0,
            ],
        }
    }
}

fn kelvin_to_rgb(kelvin: f32) -> [f32; 3] {
    let temperature = (kelvin.clamp(1000.0, 40_000.0) / 100.0).max(1.0);
    let red = if temperature <= 66.0 {
        255.0
    } else {
        329.698_73 * (temperature - 60.0).powf(-0.133_204_76)
    };
    let green = if temperature <= 66.0 {
        99.470_8 * temperature.ln() - 161.119_57
    } else {
        288.122_16 * (temperature - 60.0).powf(-0.075_514_846)
    };
    let blue = if temperature >= 66.0 {
        255.0
    } else if temperature <= 19.0 {
        0.0
    } else {
        138.517_73 * (temperature - 10.0).ln() - 305.044_8
    };
    [
        red.clamp(0.0, 255.0) / 255.0,
        green.clamp(0.0, 255.0) / 255.0,
        blue.clamp(0.0, 255.0) / 255.0,
    ]
}
