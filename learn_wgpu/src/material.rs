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
    // Base hue (degrees), emission hue, emission map enabled, double sided.
    pub color_adjustments: [f32; 4],
    pub emissive_color: [f32; 4],
    // Alpha mode, cutoff, ignore texture alpha, opacity multiplier.
    pub transparency: [f32; 4],
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

impl MaterialUniform {
    pub fn needs_alpha_blend(&self, texture_has_alpha: bool) -> bool {
        let mode = self.transparency[0] as u32;
        matches!(mode, 0 | 3)
            && (self.base_color[3] * self.transparency[3] < 1.0
                || (self.options[2] > 0.5 && self.transparency[2] < 0.5 && texture_has_alpha))
    }
}

pub struct PbrMaterial {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform: MaterialUniform,
    base_color: Arc<Texture>,
    normal: Arc<Texture>,
    metallic_roughness: Arc<Texture>,
    emissive: Arc<Texture>,
}

impl PbrMaterial {
    pub fn base_texture_has_alpha(&self) -> bool {
        self.base_color.has_transparency
    }

    pub fn needs_alpha_blend(&self) -> bool {
        self.uniform
            .needs_alpha_blend(self.base_color.has_transparency)
    }

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
        let emissive = Arc::clone(&template.emissive);
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
                texture_binding(7, &emissive),
                sampler_binding(8, &emissive),
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
            emissive,
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
        let emissive = Arc::new(Texture::from_rgba8(
            device,
            queue,
            &[255; 4],
            1,
            1,
            true,
            "Neutral emission",
        )?);
        let uniform = MaterialUniform {
            color_adjustments: [0.0; 4],
            emissive_color: [1.0; 4],
            transparency: default_transparency(),
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
                texture_entry(7),
                sampler_entry(8),
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
                texture_binding(7, &emissive),
                sampler_binding(8, &emissive),
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
            emissive,
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
        emissive: Option<&Texture>,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let emissive = emissive.unwrap_or(&self.emissive);
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
                texture_binding(7, emissive),
                sampler_binding(8, emissive),
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

    pub fn set_emissive_texture(&mut self, device: &wgpu::Device, texture: Arc<Texture>) {
        self.emissive = texture;
        self.uniform.color_adjustments[2] = 1.0;
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
                texture_binding(7, &self.emissive),
                sampler_binding(8, &self.emissive),
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

pub fn default_transparency() -> [f32; 4] {
    [0.0, 0.5, 0.0, 1.0]
}

/// Shared color controls keep all material editors consistent.
pub fn color_controls(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    uniform: &mut MaterialUniform,
) -> egui::Id {
    ui.push_id(("material_color_controls", id_salt), |ui| {
    let transparency = ui.collapsing("Transparency", |ui| {
        let modes = ["Automatic", "Opaque", "Cutout", "Alpha blend", "Dithered"];
        let mut mode = (uniform.transparency[0] as usize).min(modes.len() - 1);
        egui::ComboBox::from_id_salt("alpha_mode").selected_text(modes[mode]).show_ui(ui, |ui| {
            for (index, label) in modes.iter().enumerate() {
                ui.selectable_value(&mut mode, index, *label);
            }
        });
        uniform.transparency[0] = mode as f32;
        ui.add_enabled(mode != 1, egui::Slider::new(&mut uniform.transparency[3], 0.0..=1.0).text("Opacity"));
        let mut texture_alpha = uniform.transparency[2] < 0.5;
        ui.checkbox(&mut texture_alpha, "Use base texture alpha");
        uniform.transparency[2] = if texture_alpha { 0.0 } else { 1.0 };
        if mode == 2 {
            ui.add(egui::Slider::new(&mut uniform.transparency[1], 0.0..=1.0).text("Alpha cutoff"));
        }
        let mut double_sided = uniform.color_adjustments[3] > 0.5;
        ui.checkbox(&mut double_sided, "Double sided");
        uniform.color_adjustments[3] = if double_sided { 1.0 } else { 0.0 };
        if ui.small_button("Reset transparency").clicked() {
            uniform.transparency = default_transparency();
            uniform.color_adjustments[3] = 0.0;
        }
        ui.add(egui::Label::new(egui::RichText::new("Opacity multiplies material and texture alpha. Cutout suits foliage; dithered avoids overlap sorting artifacts.").small()).wrap());
    });
    ui.add(
        egui::Slider::new(&mut uniform.color_adjustments[0], -180.0..=180.0).text("Color hue °"),
    );
    ui.add(egui::Slider::new(&mut uniform.options[1], 0.0..=20.0).text("Emissive intensity"));
    ui.horizontal(|ui| {
        ui.label("Emission tint");
        ui.color_edit_button_rgb((&mut uniform.emissive_color[..3]).try_into().unwrap());
    });
    ui.add(
        egui::Slider::new(&mut uniform.color_adjustments[1], -180.0..=180.0).text("Emission hue °"),
    );
    transparency.header_response.id
    }).inner
}

#[cfg(test)]
mod transparency_tests {
    #[test]
    fn repeated_material_editors_have_independent_transparency_state() {
        let context = egui::Context::default();
        let mut material: super::MaterialUniform = bytemuck::Zeroable::zeroed();
        material.transparency = super::default_transparency();
        let _ = context.run_ui(Default::default(), |ui| {
            let first = super::color_controls(ui, ("group", 1), &mut material);
            let second = super::color_controls(ui, ("group", 2), &mut material);
            assert_ne!(first, second);
            let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
                &context, first, false,
            );
            state.set_open(true);
            state.store(&context);
            assert!(
                !egui::collapsing_header::CollapsingState::load_with_default_open(
                    &context, second, false,
                )
                .is_open()
            );
        });
    }

    #[test]
    fn blend_pass_tracks_material_and_texture_alpha() {
        let mut uniform: super::MaterialUniform = bytemuck::Zeroable::zeroed();
        uniform.base_color = [1.0; 4];
        uniform.options[2] = 1.0;
        uniform.transparency = super::default_transparency();
        assert!(!uniform.needs_alpha_blend(false));
        assert!(uniform.needs_alpha_blend(true));
        uniform.transparency[2] = 1.0;
        assert!(!uniform.needs_alpha_blend(true));
        uniform.transparency[3] = 0.4;
        assert!(uniform.needs_alpha_blend(false));
        for mode in [1.0, 2.0, 4.0] {
            uniform.transparency[0] = mode;
            assert!(!uniform.needs_alpha_blend(true));
        }
        uniform.transparency = super::default_transparency();
        uniform.options[2] = 0.0;
        assert!(!uniform.needs_alpha_blend(true));
        uniform.base_color[3] = 0.0;
        assert!(uniform.needs_alpha_blend(false));
    }

    #[test]
    fn opacity_map_multiplies_existing_alpha_without_changing_rgb() {
        let mut base =
            image::RgbaImage::from_raw(2, 1, vec![20, 30, 40, 128, 50, 60, 70, 255]).unwrap();
        let alpha =
            image::RgbaImage::from_raw(2, 1, vec![0, 0, 0, 255, 128, 128, 128, 255]).unwrap();
        super::apply_opacity_map(&mut base, &alpha);
        assert_eq!(base.into_raw(), vec![20, 30, 40, 0, 50, 60, 70, 128]);
    }

    #[test]
    fn all_material_shaders_validate_with_transparency() {
        for shader in [
            include_str!("shader.wgsl"),
            include_str!("material_preview.wgsl"),
            include_str!("lighting/shadow.wgsl"),
        ] {
            let source = format!("{}\n{}", include_str!("material_alpha.wgsl"), shader);
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}

/// Fold a conventional grayscale opacity map into base alpha so every existing
/// texture import, library save, and shader uses the same RGBA representation.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn bake_opacity_texture(
    base: Option<&std::path::Path>,
    opacity: &std::path::Path,
) -> anyhow::Result<std::path::PathBuf> {
    use std::hash::{Hash, Hasher};
    let (alpha, width, height) = decode_viewport_rgba(opacity)?;
    let alpha = image::RgbaImage::from_raw(width, height, alpha).unwrap();
    let mut base = if let Some(path) = base {
        let (pixels, w, h) = decode_viewport_rgba(path)?;
        image::RgbaImage::from_raw(w, h, pixels).unwrap()
    } else {
        image::RgbaImage::from_pixel(width, height, image::Rgba([255; 4]))
    };
    apply_opacity_map(&mut base, &alpha);
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    base.dimensions().hash(&mut hash);
    base.as_raw().hash(&mut hash);
    let root = crate::user_data::data_dir()?.join("material-assets");
    std::fs::create_dir_all(&root)?;
    let path = root.join(format!("alpha-{:016x}.png", hash.finish()));
    if !path.exists() {
        base.save(&path)?;
    }
    Ok(path)
}

fn apply_opacity_map(base: &mut image::RgbaImage, alpha: &image::RgbaImage) {
    let alpha = image::imageops::resize(
        alpha,
        base.width(),
        base.height(),
        image::imageops::FilterType::Triangle,
    );
    for (pixel, opacity) in base.pixels_mut().zip(alpha.pixels()) {
        pixel[3] = ((u16::from(pixel[3]) * u16::from(opacity[0]) + 127) / 255) as u8;
    }
}
