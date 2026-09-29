use super::*;
use crate::model::{ImportedPbr, MaterialSource};
use image::{Rgba, RgbaImage, imageops::FilterType};
use openusd::ar::Resolver;
use sha2::{Digest, Sha256};

pub(super) fn asset_bytes(asset: &sdf::AssetPath) -> Result<Vec<u8>> {
    let path = asset.resolved_path().context("Unresolved texture asset")?;
    Ok(openusd::ar::DefaultResolver::new()
        .open_asset(&openusd::ar::ResolvedPath::new(path))?
        .read_all()?)
}
struct Texture {
    image: RgbaImage,
    uv: String,
    transform: [f32; 5],
}
fn connected(p: &Prim, name: &str, time: f64, shader_type: &str) -> Result<Option<(Prim, String)>> {
    let mut attribute = p.attribute(name);
    let mut visited = HashSet::new();
    loop {
        let connections = attribute.connections()?;
        let Some(path) = connections.first() else {
            return Ok(None);
        };
        ensure!(
            visited.insert(path.to_string()),
            "Cyclic material connection"
        );
        let source = p
            .stage()
            .prim(path.parent().context("Invalid material connection")?)?;
        let output = path
            .as_str()
            .rsplit_once('.')
            .context("Missing connection output")?
            .1
            .to_string();
        if token(&source, "info:id", time, "")? == shader_type {
            return Ok(Some((source, output)));
        }
        attribute = source.attribute(output.as_str());
    }
}
impl Importer<'_> {
    pub(super) fn persist(&self, data: &[u8], extension: &str) -> Result<String> {
        ensure!(
            extension.chars().all(|c| c.is_ascii_alphanumeric()),
            "Invalid asset extension"
        );
        std::fs::create_dir_all(self.cache)?;
        let dest = self
            .cache
            .join(format!("{:x}.{extension}", Sha256::digest(data)));
        if !dest.exists() {
            let temp = self.cache.join(format!(
                "{:x}.{}.{}.tmp",
                Sha256::digest(data),
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos()
            ));
            std::fs::write(&temp, data)?;
            if let Err(e) = std::fs::rename(&temp, &dest) {
                let _ = std::fs::remove_file(&temp);
                if !dest.exists() {
                    return Err(e.into());
                }
            }
        }
        Ok(std::fs::canonicalize(dest)?.to_string_lossy().into_owned())
    }
    fn save_image(&self, image: RgbaImage) -> Result<String> {
        let mut bytes = std::io::Cursor::new(vec![]);
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, image::ImageOutputFormat::Png)?;
        self.persist(bytes.get_ref(), "png")
    }
    fn texture(
        &mut self,
        p: &Prim,
        name: &str,
        color: bool,
        normal: bool,
    ) -> Result<Option<Texture>> {
        let Some((shader, output)) =
            connected(p, &format!("inputs:{name}"), self.time, "UsdUVTexture")?
        else {
            if !p
                .attribute(format!("inputs:{name}"))
                .connections()?
                .is_empty()
            {
                self.warn(format!(
                    "{}: unsupported texture network for {name}",
                    p.path()
                ));
            }
            return Ok(None);
        };
        let Some(sdf::Value::AssetPath(asset)) = value(&shader, "inputs:file", self.time)? else {
            bail!("Missing texture asset")
        };
        let image = image::load_from_memory(&asset_bytes(&asset)?)?;
        let mut image = image.thumbnail(2048, 2048).to_rgba8();
        let v4 = |name: &str, default: [f32; 4]| -> Result<[f32; 4]> {
            Ok(match value(&shader, name, self.time)? {
                Some(sdf::Value::Vec4f(v)) => v.into(),
                _ => default,
            })
        };
        let scale = v4("inputs:scale", [1.; 4])?;
        let bias = v4("inputs:bias", [0.; 4])?;
        let space = token(&shader, "inputs:sourceColorSpace", self.time, "auto")?;
        let srgb = space == "sRGB" || (space == "auto" && color);
        let channel = match output.rsplit(':').next().unwrap_or("") {
            "r" => Some(0),
            "g" => Some(1),
            "b" => Some(2),
            "a" => Some(3),
            _ => None,
        };
        let remap = normal && (scale[..3] != [1.; 3] || bias[..3] != [0.; 3]);
        let tables: [[u8; 256]; 4] = std::array::from_fn(|out| {
            std::array::from_fn(|byte| {
                let source = if out < 3 { channel.unwrap_or(out) } else { out };
                let mut v = byte as f32 / 255.;
                if srgb && source < 3 {
                    v = if v <= 0.04045 {
                        v / 12.92
                    } else {
                        ((v + 0.055) / 1.055).powf(2.4)
                    }
                }
                v = v * scale[source] + bias[source];
                if out < 3 {
                    if remap {
                        v = v * 0.5 + 0.5
                    } else if color && !normal {
                        v = if v <= 0.0031308 {
                            v * 12.92
                        } else {
                            1.055 * v.max(0.).powf(1. / 2.4) - 0.055
                        }
                    }
                }
                (v.clamp(0., 1.) * 255.).round() as u8
            })
        });
        for pixel in image.pixels_mut() {
            let source = *pixel;
            for out in 0..4 {
                pixel[out] = tables[out]
                    [source[if out < 3 { channel.unwrap_or(out) } else { out }] as usize];
            }
        }
        let mut uv = "st".to_string();
        let mut transform = [1., 1., 0., 0., 0.];
        if let Some((reader, _)) =
            connected(&shader, "inputs:st", self.time, "UsdPrimvarReader_float2")?
        {
            uv = token(&reader, "inputs:varname", self.time, "st")?;
        }
        if let Some((mapping, _)) = connected(&shader, "inputs:st", self.time, "UsdTransform2d")? {
            let v2 = |name: &str, default: [f32; 2]| -> Result<[f32; 2]> {
                Ok(match value(&mapping, name, self.time)? {
                    Some(sdf::Value::Vec2f(v)) => v.into(),
                    _ => default,
                })
            };
            let scale = v2("inputs:scale", [1.; 2])?;
            let offset = v2("inputs:translation", [0.; 2])?;
            transform = [
                scale[0],
                scale[1],
                offset[0],
                offset[1],
                scalar(&mapping, "inputs:rotation", self.time, 0.)?,
            ];
            if let Some((reader, _)) =
                connected(&mapping, "inputs:in", self.time, "UsdPrimvarReader_float2")?
            {
                uv = token(&reader, "inputs:varname", self.time, "st")?;
            }
        }
        for axis in ["wrapS", "wrapT"] {
            let wrap = token(&shader, &format!("inputs:{axis}"), self.time, "repeat")?;
            if !matches!(wrap.as_str(), "repeat" | "useMetadata") {
                self.warn(format!(
                    "{}: {wrap} wrapping is approximated by repeat",
                    shader.path()
                ));
            }
        }
        Ok(Some(Texture {
            image,
            uv,
            transform,
        }))
    }
    pub(super) fn material(&mut self, p: &Prim) -> Result<usize> {
        let mut bound = None;
        let mut current = Some(p.clone());
        while let Some(prim) = current {
            if let Some(api) =
                openusd_schemas::shade::MaterialBindingAPI::get(self.stage, prim.path().clone())?
            {
                if let Some(path) = api.compute_bound_material("")? {
                    bound = Some(path);
                    break;
                }
            }
            if let Some(path) = prim.relationship("material:binding").targets()?.first() {
                bound = Some(path.clone());
                break;
            }
            current = parent(&prim)?;
        }
        let surface = if p.type_name()?.is_some_and(|t| t.as_str() == "GeomSubset") {
            parent(p)?.unwrap_or(p.clone())
        } else {
            p.clone()
        };
        let double_sided = boolean(&surface, "doubleSided", self.time, false)?;
        let mut key = bound
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("display:{}", p.path()));
        if double_sided {
            key.push_str(" · Double sided");
        }
        if let Some(&id) = self.material_ids.get(&key) {
            return Ok(id);
        }
        let mut record = MaterialSource {
            name: key.clone(),
            diffuse: [0.8; 3],
            opacity: 1.,
            opacity_texture: String::new(),
            alpha_cutoff: None,
            double_sided,
            diffuse_texture: String::new(),
            normal_texture: String::new(),
            roughness_texture: String::new(),
            emissive_texture: String::new(),
            pbr: None,
        };
        let mut pbr = ImportedPbr {
            metallic: 0.,
            roughness: 0.5,
            opacity: 1.,
            emissive: [0.; 3],
        };
        let mut uv = ("st".to_string(), [1., 1., 0., 0., 0.]);
        let shader = bound
            .map(|path| self.stage.prim(path))
            .transpose()?
            .map(|material| connected(&material, "outputs:surface", self.time, "UsdPreviewSurface"))
            .transpose()?
            .flatten();
        if let Some((shader, _)) = shader {
            record.diffuse = vector(&shader, "inputs:diffuseColor", self.time, [0.18; 3])?;
            let cutoff = scalar(&shader, "inputs:opacityThreshold", self.time, 0.)?;
            if cutoff > 0. {
                record.alpha_cutoff = Some(cutoff);
            }
            pbr = ImportedPbr {
                metallic: scalar(&shader, "inputs:metallic", self.time, 0.)?,
                roughness: scalar(&shader, "inputs:roughness", self.time, 0.5)?,
                opacity: scalar(&shader, "inputs:opacity", self.time, 1.)?,
                emissive: vector(&shader, "inputs:emissiveColor", self.time, [0.; 3])?,
            };
            let mut maps = HashMap::new();
            let mut has_uv = false;
            for name in [
                "diffuseColor",
                "normal",
                "emissiveColor",
                "metallic",
                "roughness",
                "opacity",
            ] {
                match self.texture(
                    &shader,
                    name,
                    matches!(name, "diffuseColor" | "emissiveColor"),
                    name == "normal",
                ) {
                    Ok(Some(texture)) => {
                        let settings = (texture.uv, texture.transform);
                        if has_uv && uv != settings {
                            self.warn(format!(
                                "{key}: multiple UV sets/transforms; viewport uses {}",
                                uv.0
                            ));
                        } else {
                            uv = settings;
                            has_uv = true;
                        }
                        maps.insert(name, texture.image);
                    }
                    Ok(None) => {}
                    Err(e) => self.warn(format!("{}: {name} texture: {e}", shader.path())),
                }
            }
            if let Some(image) = maps.remove("normal") {
                record.normal_texture = self.save_image(image)?;
            }
            if let Some(image) = maps.remove("emissiveColor") {
                record.emissive_texture = self.save_image(image)?;
                pbr.emissive = [1.; 3];
            }
            let metal = maps.remove("metallic");
            let rough = maps.remove("roughness");
            if metal.is_some() || rough.is_some() {
                let (w, h) = metal
                    .iter()
                    .chain(rough.iter())
                    .map(|i| i.dimensions())
                    .max_by_key(|&(w, h)| u64::from(w) * u64::from(h))
                    .unwrap();
                let metal = metal.map(|i| image::imageops::resize(&i, w, h, FilterType::Triangle));
                let rough = rough.map(|i| image::imageops::resize(&i, w, h, FilterType::Triangle));
                let packed = RgbaImage::from_fn(w, h, |x, y| {
                    Rgba([
                        metal
                            .as_ref()
                            .map_or((pbr.metallic.clamp(0., 1.) * 255.).round() as u8, |i| {
                                i.get_pixel(x, y)[0]
                            }),
                        rough
                            .as_ref()
                            .map_or((pbr.roughness.clamp(0., 1.) * 255.).round() as u8, |i| {
                                i.get_pixel(x, y)[0]
                            }),
                        0,
                        255,
                    ])
                });
                record.roughness_texture = self.save_image(packed)?;
                pbr.metallic = 1.;
                pbr.roughness = 1.;
            }
            let mut diffuse = maps.remove("diffuseColor");
            if diffuse.is_some() {
                record.diffuse = [1.; 3];
            }
            if let Some(alpha) = maps.remove("opacity") {
                let base = diffuse.get_or_insert_with(|| {
                    RgbaImage::from_pixel(alpha.width(), alpha.height(), Rgba([255; 4]))
                });
                let alpha = image::imageops::resize(
                    &alpha,
                    base.width(),
                    base.height(),
                    FilterType::Triangle,
                );
                for (p, a) in base.pixels_mut().zip(alpha.pixels()) {
                    p[3] = a[0];
                }
                pbr.opacity = 1.;
            }
            if let Some(image) = diffuse {
                record.diffuse_texture = self.save_image(image)?;
            }
        } else {
            if let Some(color) = vectors(&surface, "primvars:displayColor", self.time)?.first() {
                record.diffuse = *color;
            }
            if let Some(sdf::Value::FloatVec(opacity)) =
                value(&surface, "primvars:displayOpacity", self.time)?
            {
                pbr.opacity = opacity.first().copied().unwrap_or(1.);
            }
        }
        record.opacity = pbr.opacity;
        record.pbr = Some(pbr);
        let id = self.scene.materials.len();
        self.scene.materials.push(record);
        self.uv_settings.push(uv);
        self.material_ids.insert(key, id);
        Ok(id)
    }
}
