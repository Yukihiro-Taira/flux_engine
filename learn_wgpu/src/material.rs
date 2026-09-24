use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::texture::Texture;

#[cfg(not(target_arch = "wasm32"))]
const MATERIAL_TEXTURE_CACHE_VERSION: u32 = 1;
const VIEWPORT_TEXTURE_MAX_EDGE: u32 = 2048;

#[cfg(not(target_arch = "wasm32"))]
#[derive(serde::Serialize, serde::Deserialize)]
struct MaterialTextureCache {
    version: u32,
    source_len: u64,
    source_modified_ns: u128,
    maximum_edge: u32,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub base_color: [f32; 4],
    // metallic, roughness, normal strength, environment strength
    pub properties: [f32; 4],
    // environment rotation radians, emissive strength, use textures, padding
    pub options: [f32; 4],
    // UV overlay enabled, reserved inspection controls
    pub inspection: [f32; 4],
    // UDIM minimum U/V, atlas columns/rows
    pub udim: [f32; 4],
    // 128-bit row-major assignment mask for atlas tiles.
    pub udim_mask: [u32; 4],
}

pub struct PbrMaterial {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform: MaterialUniform,
    base_color: Arc<Texture>,
    normal: Arc<Texture>,
    metallic_roughness: Arc<Texture>,
}

impl PbrMaterial {
    pub fn new_instance_from(device: &wgpu::Device, template: &Self) -> Self {
        let uniform = template.uniform;
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PBR material instance uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = template.layout.clone();
        let base_color = Arc::clone(&template.base_color);
        let normal = Arc::clone(&template.normal);
        let metallic_roughness = Arc::clone(&template.metallic_roughness);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR material instance bind group"),
            layout: &layout,
            entries: &[
                texture_binding(0, &base_color),
                sampler_binding(1, &base_color),
                texture_binding(2, &normal),
                sampler_binding(3, &normal),
                texture_binding(4, &metallic_roughness),
                sampler_binding(5, &metallic_roughness),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            layout,
            bind_group,
            uniform_buffer,
            uniform,
            base_color,
            normal,
            metallic_roughness,
        }
    }

    pub fn new_untextured(device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<Self> {
        // The shader binding must always contain a valid texture. A single
        // neutral white texel fulfills that GPU requirement without applying
        // an image texture to the material.
        let neutral = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            1,
            1,
            image::Rgba([255, 255, 255, 255]),
        ));
        Self::new(device, queue, &neutral)
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base_image: &image::DynamicImage,
    ) -> anyhow::Result<Self> {
        let base_rgba = base_image.to_rgba8();
        let base_color = Arc::new(Texture::from_rgba8(
            device,
            queue,
            base_rgba.as_raw(),
            base_rgba.width(),
            base_rgba.height(),
            true,
            "PBR base color",
        )?);
        let normal = Arc::new(Texture::from_rgba8(
            device,
            queue,
            &[128, 128, 255, 255],
            1,
            1,
            false,
            "PBR flat normal",
        )?);
        // R = metallic, G = roughness.
        let metallic_roughness = Arc::new(Texture::from_rgba8(
            device,
            queue,
            // White metallic channel lets the graph's metallic factor control
            // the default material. A loaded map is multiplied by that factor.
            &[255, 255, 0, 255],
            1,
            1,
            false,
            "PBR metallic roughness",
        )?);
        let uniform = MaterialUniform {
            base_color: [1.0; 4],
            properties: [0.0, 0.5, 1.0, 1.0],
            options: [0.0, 0.0, 1.0, 0.0],
            inspection: [0.0; 4],
            udim: [0.0, 0.0, 1.0, 1.0],
            udim_mask: [0; 4],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PBR material uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PBR material layout"),
            entries: &[
                texture_entry(0),
                sampler_entry(1),
                texture_entry(2),
                sampler_entry(3),
                texture_entry(4),
                sampler_entry(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
            label: Some("PBR material bind group"),
            layout: &layout,
            entries: &[
                texture_binding(0, &base_color),
                sampler_binding(1, &base_color),
                texture_binding(2, &normal),
                sampler_binding(3, &normal),
                texture_binding(4, &metallic_roughness),
                sampler_binding(5, &metallic_roughness),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            layout,
            bind_group,
            uniform_buffer,
            uniform,
            base_color,
            normal,
            metallic_roughness,
        })
    }

    pub fn upload(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform));
    }

    pub fn bind_group_with_overrides_and_uniform(
        &self,
        device: &wgpu::Device,
        base_color: Option<&Texture>,
        normal: Option<&Texture>,
        metallic_roughness: Option<&Texture>,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let base_color = base_color.unwrap_or(&self.base_color);
        let normal = normal.unwrap_or(&self.normal);
        let metallic_roughness = metallic_roughness.unwrap_or(&self.metallic_roughness);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR per-group texture-set bind group"),
            layout: &self.layout,
            entries: &[
                texture_binding(0, base_color),
                sampler_binding(1, base_color),
                texture_binding(2, normal),
                sampler_binding(3, normal),
                texture_binding(4, metallic_roughness),
                sampler_binding(5, metallic_roughness),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn set_base_color_texture(&mut self, device: &wgpu::Device, texture: Arc<Texture>) {
        self.base_color = texture;
        self.rebuild_bind_group(device);
    }

    pub fn set_normal_texture(&mut self, device: &wgpu::Device, texture: Arc<Texture>) {
        self.normal = texture;
        self.rebuild_bind_group(device);
    }

    pub fn set_metallic_roughness_texture(&mut self, device: &wgpu::Device, texture: Arc<Texture>) {
        self.metallic_roughness = texture;
        self.rebuild_bind_group(device);
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR material bind group"),
            layout: &self.layout,
            entries: &[
                texture_binding(0, &self.base_color),
                sampler_binding(1, &self.base_color),
                texture_binding(2, &self.normal),
                sampler_binding(3, &self.normal),
                texture_binding(4, &self.metallic_roughness),
                sampler_binding(5, &self.metallic_roughness),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
    }
}

pub fn load_shared_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &std::path::Path,
    srgb: bool,
) -> anyhow::Result<Arc<Texture>> {
    let (rgba, width, height) = load_viewport_rgba(path)?;
    Ok(Arc::new(Texture::from_rgba8(
        device,
        queue,
        &rgba,
        width,
        height,
        srgb,
        if srgb {
            "Shared PBR base color"
        } else {
            "Shared PBR linear map"
        },
    )?))
}

#[cfg(target_arch = "wasm32")]
fn load_viewport_rgba(path: &std::path::Path) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    decode_viewport_rgba(path)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_viewport_rgba(path: &std::path::Path) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let metadata = std::fs::metadata(path)?;
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map_or_else(
            || "fxtexcache".to_owned(),
            |value| format!("{value}.fxtexcache"),
        );
    let cache_path = path.with_extension(extension);
    if let Ok(file) = std::fs::File::open(&cache_path)
        && let Ok(cache) =
            bincode::deserialize_from::<_, MaterialTextureCache>(std::io::BufReader::new(file))
        && cache.version == MATERIAL_TEXTURE_CACHE_VERSION
        && cache.source_len == metadata.len()
        && cache.source_modified_ns == modified_ns
        && cache.maximum_edge == VIEWPORT_TEXTURE_MAX_EDGE
        && cache.rgba.len() == cache.width as usize * cache.height as usize * 4
    {
        return Ok((cache.rgba, cache.width, cache.height));
    }

    let (rgba, width, height) = decode_viewport_rgba(path)?;
    let cache = MaterialTextureCache {
        version: MATERIAL_TEXTURE_CACHE_VERSION,
        source_len: metadata.len(),
        source_modified_ns: modified_ns,
        maximum_edge: VIEWPORT_TEXTURE_MAX_EDGE,
        width,
        height,
        rgba,
    };
    let temporary = cache_path.with_extension("fxtexcache.tmp");
    if let Ok(file) = std::fs::File::create(&temporary) {
        let mut writer = std::io::BufWriter::new(file);
        if bincode::serialize_into(&mut writer, &cache).is_ok()
            && std::io::Write::flush(&mut writer).is_ok()
        {
            let _ = std::fs::remove_file(&cache_path);
            let _ = std::fs::rename(&temporary, &cache_path);
        } else {
            let _ = std::fs::remove_file(&temporary);
        }
    }
    Ok((cache.rgba, width, height))
}

fn decode_viewport_rgba(path: &std::path::Path) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let rgba = image::open(path)?.to_rgba8();
    let longest = rgba.width().max(rgba.height());
    let rgba = if longest > VIEWPORT_TEXTURE_MAX_EDGE {
        let scale = VIEWPORT_TEXTURE_MAX_EDGE as f64 / longest as f64;
        let width = (rgba.width() as f64 * scale).round().max(1.0) as u32;
        let height = (rgba.height() as f64 * scale).round().max(1.0) as u32;
        image::imageops::resize(&rgba, width, height, image::imageops::FilterType::Triangle)
    } else {
        rgba
    };
    let (width, height) = rgba.dimensions();
    Ok((rgba.into_raw(), width, height))
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn texture_binding<'a>(binding: u32, texture: &'a Texture) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(&texture.view),
    }
}

fn sampler_binding<'a>(binding: u32, texture: &'a Texture) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(&texture.sampler),
    }
}
