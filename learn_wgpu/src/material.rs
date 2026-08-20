use wgpu::util::DeviceExt;

use crate::texture::Texture;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
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
    base_color: Texture,
    normal: Texture,
    metallic_roughness: Texture,
}

impl PbrMaterial {
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
        let base_color = Texture::from_rgba8(
            device,
            queue,
            base_rgba.as_raw(),
            base_rgba.width(),
            base_rgba.height(),
            true,
            "PBR base color",
        )?;
        let normal = Texture::from_rgba8(
            device,
            queue,
            &[128, 128, 255, 255],
            1,
            1,
            false,
            "PBR flat normal",
        )?;
        // R = metallic, G = roughness.
        let metallic_roughness = Texture::from_rgba8(
            device,
            queue,
            // White metallic channel lets the graph's metallic factor control
            // the default material. A loaded map is multiplied by that factor.
            &[255, 255, 0, 255],
            1,
            1,
            false,
            "PBR metallic roughness",
        )?;
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

    pub fn bind_group_with_overrides(
        &self,
        device: &wgpu::Device,
        base_color: Option<&Texture>,
        normal: Option<&Texture>,
        metallic_roughness: Option<&Texture>,
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
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn load_base_color(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        let image = image::open(path)?;
        let rgba = image.to_rgba8();
        self.base_color = Texture::from_rgba8(
            device,
            queue,
            rgba.as_raw(),
            rgba.width(),
            rgba.height(),
            true,
            "PBR base color",
        )?;
        self.rebuild_bind_group(device);
        Ok(())
    }

    pub fn load_normal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        self.normal = load_linear_texture(device, queue, path, "PBR normal map")?;
        self.rebuild_bind_group(device);
        Ok(())
    }

    pub fn load_metallic_roughness(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        self.metallic_roughness =
            load_linear_texture(device, queue, path, "PBR metallic roughness")?;
        self.rebuild_bind_group(device);
        Ok(())
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

fn load_linear_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &std::path::Path,
    label: &str,
) -> anyhow::Result<Texture> {
    let image = image::open(path)?;
    let rgba = image.to_rgba8();
    Texture::from_rgba8(
        device,
        queue,
        rgba.as_raw(),
        rgba.width(),
        rgba.height(),
        false,
        label,
    )
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
