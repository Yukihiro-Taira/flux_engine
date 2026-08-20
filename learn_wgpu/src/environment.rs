use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use cgmath::SquareMatrix;
use image::GenericImageView;
use wgpu::util::DeviceExt;

use crate::Camera;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EnvironmentUniform {
    inverse_view_projection: [[f32; 4]; 4],
    camera_position: [f32; 4],
    // intensity, exposure in stops, rotation in radians, unused
    settings: [f32; 4],
}

pub struct Environment {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    // Keep the GPU texture alive for as long as its view is bound.
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl Environment {
    pub fn from_path(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        path: &Path,
    ) -> Result<Self> {
        const MAX_ENVIRONMENT_EDGE: u32 = 4096;
        let maximum_edge = MAX_ENVIRONMENT_EDGE.min(device.limits().max_texture_dimension_2d);
        let image = load_environment_image(path, maximum_edge)?;
        let original_dimensions = image.dimensions();
        anyhow::ensure!(
            original_dimensions.0 > 0 && original_dimensions.1 > 0,
            "HDRI is empty"
        );

        // An 8K RGBA32F environment occupies 512 MiB on both the CPU and GPU.
        // Consume the decoded buffer to avoid cloning it, then reduce oversized
        // maps for predictable interactive viewport memory use.
        let longest_edge = original_dimensions.0.max(original_dimensions.1);
        let rgba = image.into_rgba32f();
        let rgba = if longest_edge > maximum_edge {
            let scale = maximum_edge as f64 / longest_edge as f64;
            let width = (original_dimensions.0 as f64 * scale).round().max(1.0) as u32;
            let height = (original_dimensions.1 as f64 * scale).round().max(1.0) as u32;
            image::imageops::resize(&rgba, width, height, image::imageops::FilterType::Triangle)
        } else {
            rgba
        };
        let dimensions = rgba.dimensions();
        let mip_level_count = 32 - dimensions.0.max(dimensions.1).leading_zeros();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDRI environment texture"),
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for mip_level in 0..mip_level_count {
            let mip_width = (dimensions.0 >> mip_level).max(1);
            let mip_height = (dimensions.1 >> mip_level).max(1);
            let mip = (mip_level > 0).then(|| {
                image::imageops::resize(
                    &rgba,
                    mip_width,
                    mip_height,
                    image::imageops::FilterType::Triangle,
                )
            });
            let pixels = mip.as_ref().map_or(rgba.as_raw(), |image| image.as_raw());
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(pixels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16 * mip_width),
                    rows_per_image: Some(mip_height),
                },
                wgpu::Extent3d {
                    width: mip_width,
                    height: mip_height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("HDRI environment sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_max_clamp: mip_level_count.saturating_sub(1) as f32,
            ..Default::default()
        });

        let uniform = EnvironmentUniform {
            inverse_view_projection: cgmath::Matrix4::identity().into(),
            camera_position: [0.0; 4],
            settings: [1.0, 0.0, 0.0, 0.0],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HDRI environment uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HDRI environment bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HDRI environment bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HDRI environment shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("environment.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HDRI environment pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HDRI environment pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group,
            uniform_buffer,
            _texture: texture,
            view,
            sampler,
        })
    }

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        camera: &Camera,
        intensity: f32,
        exposure: f32,
        rotation_degrees: f32,
    ) {
        let inverse = camera
            .build_view_projection_matrix()
            .invert()
            .unwrap_or_else(cgmath::Matrix4::identity);
        let uniform = EnvironmentUniform {
            inverse_view_projection: inverse.into(),
            camera_position: [camera.eye.x, camera.eye.y, camera.eye.z, 1.0],
            settings: [intensity, exposure, rotation_degrees.to_radians(), 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn create_lighting_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HDRI PBR lighting bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

pub fn lighting_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("HDRI PBR lighting layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    })
}

pub fn fallback_lighting_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Fallback environment"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        // Neutral studio illumination keeps both dielectric and metallic
        // materials readable before the user loads a custom HDRI.
        bytemuck::cast_slice(&[0.6_f32, 0.6, 0.6, 1.0]),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Fallback environment bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    (texture, bind_group)
}

fn load_environment_image(path: &Path, maximum_edge: u32) -> Result<image::DynamicImage> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default();

    if extension.eq_ignore_ascii_case("rat") {
        return load_rat(path, maximum_edge);
    }
    if extension.eq_ignore_ascii_case("exr") {
        return load_exr_scaled(path, maximum_edge);
    }

    image::open(path).with_context(|| format!("failed to decode {}", path.display()))
}

struct DownsampledExr {
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

fn load_exr_scaled(path: &Path, maximum_edge: u32) -> Result<image::DynamicImage> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        move |resolution, _channels| {
            let source_width = resolution.width();
            let source_height = resolution.height();
            let longest_edge = source_width.max(source_height);
            let scale = (maximum_edge as f64 / longest_edge as f64).min(1.0);
            let width = (source_width as f64 * scale).round().max(1.0) as usize;
            let height = (source_height as f64 * scale).round().max(1.0) as usize;
            DownsampledExr {
                source_width,
                source_height,
                width,
                height,
                pixels: vec![0.0; width * height * 4],
            }
        },
        |output, position, (red, green, blue, alpha): (f32, f32, f32, f32)| {
            let x = position.x() * output.width / output.source_width;
            let y = position.y() * output.height / output.source_height;
            let index = (y * output.width + x) * 4;
            output.pixels[index..index + 4].copy_from_slice(&[red, green, blue, alpha]);
        },
    )
    .with_context(|| format!("failed to decode {}", path.display()))?;

    let output = image.layer_data.channel_data.pixels;
    let buffer = image::ImageBuffer::<image::Rgba<f32>, Vec<f32>>::from_raw(
        output.width as u32,
        output.height as u32,
        output.pixels,
    )
    .context("EXR decoder returned an invalid pixel buffer")?;
    Ok(image::DynamicImage::ImageRgba32F(buffer))
}

#[cfg(target_arch = "wasm32")]
fn load_rat(_path: &Path, _maximum_edge: u32) -> Result<image::DynamicImage> {
    anyhow::bail!("Houdini RAT conversion is only available in the desktop build")
}

#[cfg(not(target_arch = "wasm32"))]
fn load_rat(path: &Path, maximum_edge: u32) -> Result<image::DynamicImage> {
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    let converter = find_iconvert().context(
        "Houdini iconvert was not found; install Houdini or set HFS so .rat files can be converted",
    )?;
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary_exr = std::env::temp_dir().join(format!(
        "learn_wgpu_rat_{}_{}.exr",
        std::process::id(),
        unique
    ));

    let conversion = Command::new(&converter)
        .arg(path)
        .arg(&temporary_exr)
        .output()
        .with_context(|| format!("failed to start {}", converter.display()))?;

    if !conversion.status.success() {
        let message = String::from_utf8_lossy(&conversion.stderr);
        let _ = std::fs::remove_file(&temporary_exr);
        anyhow::bail!(
            "Houdini iconvert could not read the RAT file: {}",
            message.trim()
        );
    }

    let decoded = load_exr_scaled(&temporary_exr, maximum_edge)
        .with_context(|| format!("failed to decode converted RAT file {}", path.display()));
    let _ = std::fs::remove_file(&temporary_exr);
    decoded
}

#[cfg(not(target_arch = "wasm32"))]
fn find_iconvert() -> Option<PathBuf> {
    use std::process::Command;

    if Command::new("iconvert").arg("--help").output().is_ok() {
        return Some(PathBuf::from("iconvert"));
    }

    if let Some(path) = std::env::var_os("HFS") {
        let candidate = PathBuf::from(path).join("bin").join("iconvert");
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    #[cfg(target_os = "macos")]
    {
        let houdini_root = Path::new("/Applications/Houdini");
        if let Ok(installations) = std::fs::read_dir(houdini_root) {
            let mut candidates = installations
                .filter_map(Result::ok)
                .map(|entry| {
                    entry.path().join(
                        "Frameworks/Houdini.framework/Versions/Current/Resources/bin/iconvert",
                    )
                })
                .filter(|candidate| candidate.is_file())
                .collect::<Vec<_>>();
            candidates.sort();
            if let Some(candidate) = candidates.pop() {
                return Some(candidate);
            }
        }
    }

    None
}
