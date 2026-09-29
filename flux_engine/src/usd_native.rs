use super::*;
use anyhow::{Context, Result, bail, ensure};
use openusd::{
    gf, sdf,
    usd::{Prim, PrimPredicate, SchemaBase, SchemaKind, Stage, TimeCode},
};
use openusd_schemas::geom::{Imageable, Xformable};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
};
#[path = "usd_geometry.rs"]
mod geometry;
#[path = "usd_material.rs"]
mod material;
#[cfg(test)]
#[path = "usd_tests.rs"]
mod tests;

type Matrix = gf::Matrix4d;
type Transforms = HashMap<String, [[f32; 4]; 4]>;
struct View(Prim);
impl SchemaBase for View {
    const KIND: SchemaKind = SchemaKind::AbstractBase;
    fn prim(&self) -> &Prim {
        &self.0
    }
}
impl Imageable for View {}
impl Xformable for View {}

fn value(p: &Prim, name: &str, time: f64) -> Result<Option<sdf::Value>> {
    Ok(p.attribute(name).get_at(TimeCode::from(time))?)
}
fn number(v: Option<sdf::Value>, default: f64) -> f64 {
    match v {
        Some(sdf::Value::Double(v)) => v,
        Some(sdf::Value::Float(v)) => v as f64,
        Some(sdf::Value::Int(v)) => v as f64,
        _ => default,
    }
}
fn scalar(p: &Prim, name: &str, time: f64, default: f32) -> Result<f32> {
    Ok(number(value(p, name, time)?, default as f64) as f32)
}
fn token(p: &Prim, name: &str, time: f64, default: &str) -> Result<String> {
    Ok(match value(p, name, time)? {
        Some(sdf::Value::Token(v)) => v.to_string(),
        Some(sdf::Value::String(v)) => v,
        _ => default.into(),
    })
}
fn boolean(p: &Prim, name: &str, time: f64, default: bool) -> Result<bool> {
    Ok(match value(p, name, time)? {
        Some(sdf::Value::Bool(v)) => v,
        _ => default,
    })
}
fn vector(p: &Prim, name: &str, time: f64, default: [f32; 3]) -> Result<[f32; 3]> {
    Ok(match value(p, name, time)? {
        Some(sdf::Value::Vec3f(v)) => v.into(),
        Some(sdf::Value::Vec3d(v)) => [v.x as f32, v.y as f32, v.z as f32],
        _ => default,
    })
}
fn vectors(p: &Prim, name: &str, time: f64) -> Result<Vec<[f32; 3]>> {
    Ok(match value(p, name, time)? {
        Some(sdf::Value::Vec3fVec(v)) => v.into_iter().map(Into::into).collect(),
        Some(sdf::Value::Vec3dVec(v)) => v
            .into_iter()
            .map(|v| [v.x as f32, v.y as f32, v.z as f32])
            .collect(),
        None => vec![],
        _ => bail!("{}: invalid {name} vector array", p.path()),
    })
}
fn ints(p: &Prim, name: &str, time: f64) -> Result<Vec<i32>> {
    Ok(match value(p, name, time)? {
        Some(sdf::Value::IntVec(v)) => v,
        None => vec![],
        _ => bail!("{}: invalid {name} integer array", p.path()),
    })
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = v.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-20);
    v.map(|v| v / n)
}
fn parent(p: &Prim) -> Result<Option<Prim>> {
    p.path()
        .parent()
        .filter(|p| p.as_str() != "/")
        .map(|path| p.stage().prim(path).map_err(Into::into))
        .transpose()
}
fn world(p: &Prim, time: f64) -> Result<Matrix> {
    let view = View(p.clone());
    let local = view.local_to_parent_transform(time)?;
    Ok(if view.resets_xform_stack()? {
        local
    } else if let Some(parent) = parent(p)? {
        local * world(&parent, time)?
    } else {
        local
    })
}
fn visible(p: &Prim, time: f64) -> Result<bool> {
    let mut prim = Some(p.clone());
    while let Some(p) = prim {
        if token(&p, "visibility", time, "")? == "invisible"
            || token(&p, "purpose", time, "")? == "guide"
        {
            return Ok(false);
        }
        prim = parent(&p)?;
    }
    Ok(true)
}
fn paths(stage: &Stage) -> Result<Vec<sdf::Path>> {
    let mut paths = vec![];
    stage.traverse(PrimPredicate::DEFAULT_PROXIES, |p| paths.push(p.clone()))?;
    Ok(paths)
}
fn open(path: &Path) -> Result<Stage> {
    let path = std::fs::canonicalize(path).context("USD file does not exist")?;
    Stage::open(path.to_str().context("USD path is not valid UTF-8")?).map_err(Into::into)
}

pub fn load(path: &Path, time: Option<f64>) -> Result<UsdScene> {
    let stage = open(path)?;
    let cache = crate::user_data::data_dir()?.join("usd-assets");
    evaluate(&stage, &cache, time, false, true)
}
fn evaluate(
    stage: &Stage,
    cache: &Path,
    time: Option<f64>,
    local: bool,
    require_objects: bool,
) -> Result<UsdScene> {
    ensure!(
        time.is_none_or(f64::is_finite),
        "USD time code must be finite"
    );
    let paths = paths(stage)?;
    let mut start = f64::INFINITY;
    let mut end = f64::NEG_INFINITY;
    for path in &paths {
        for attr in stage.prim(path.clone())?.authored_attributes()? {
            if let Some(samples) = attr.time_samples()? {
                for (t, _) in samples.iter() {
                    start = start.min(*t);
                    end = end.max(*t);
                }
            }
        }
    }
    let animated = start.is_finite() && end > start;
    if !start.is_finite() {
        start = 0.;
        end = 0.;
    }
    if stage.stage_metadata("startTimeCode")?.is_some() {
        start = stage.start_time_code();
    }
    if stage.stage_metadata("endTimeCode")?.is_some() {
        end = stage.end_time_code();
    }
    let time = time.unwrap_or(start);
    let scale = number(stage.stage_metadata("metersPerUnit")?, 0.01) as f32;
    ensure!(
        scale.is_finite() && scale > 0.,
        "metersPerUnit must be positive and finite"
    );
    let z_up =
        matches!(stage.stage_metadata("upAxis")?,Some(sdf::Value::Token(v)) if v.as_str()=="Z");
    let mut importer = Importer {
        stage,
        cache,
        time,
        scale,
        z_up,
        local,
        prototypes: HashSet::new(),
        material_ids: HashMap::new(),
        uv_settings: vec![],
        vertices: 0,
        scene: UsdScene {
            meshes: vec![],
            materials: vec![],
            warnings: stage
                .composition_errors()
                .iter()
                .map(|e| format!("Composition: {e:?}"))
                .collect(),
            time_code: time,
            scene: ImportedScene {
                animation: Some(AnimationClip {
                    start,
                    end,
                    rate: stage.time_codes_per_second(),
                    animated,
                    name: "USD stage".into(),
                }),
                ..Default::default()
            },
        },
    };
    for path in &paths {
        let p = stage.prim(path.clone())?;
        if p.type_name()?
            .is_some_and(|t| t.as_str() == "PointInstancer")
        {
            for target in p.relationship("prototypes").targets()? {
                importer.prototypes.insert(target.to_string());
            }
        }
    }
    for path in paths {
        if importer
            .prototypes
            .iter()
            .any(|root| path.as_str() == root || path.as_str().starts_with(&format!("{root}/")))
        {
            continue;
        }
        let p = stage.prim(path)?;
        if visible(&p, time)? {
            importer.emit(&p, None, None, 0)?;
        }
    }
    ensure!(
        !require_objects
            || !importer.scene.meshes.is_empty()
            || !importer.scene.scene.cameras.is_empty()
            || !importer.scene.scene.lights.is_empty(),
        "USD stage contains no supported scene objects"
    );
    importer.scene.warnings.sort();
    importer.scene.warnings.dedup();
    Ok(importer.scene)
}
struct Importer<'a> {
    stage: &'a Stage,
    cache: &'a Path,
    time: f64,
    scale: f32,
    z_up: bool,
    local: bool,
    prototypes: HashSet<String>,
    material_ids: HashMap<String, usize>,
    uv_settings: Vec<(String, [f32; 5])>,
    vertices: usize,
    scene: UsdScene,
}
impl Importer<'_> {
    fn warn(&mut self, text: impl Into<String>) {
        self.scene.warnings.push(text.into());
    }
    fn viewport(&self, v: [f32; 3], position: bool) -> [f32; 3] {
        let v = if self.z_up { v } else { [v[0], -v[2], v[1]] };
        v.map(|v| v * if position { self.scale } else { 1. })
    }
    fn mesh_vector(&self, v: [f32; 3], position: bool) -> [f32; 3] {
        let v = if self.z_up { [v[0], v[2], -v[1]] } else { v };
        v.map(|v| v * if position { self.scale } else { 1. })
    }
    fn emit(
        &mut self,
        p: &Prim,
        transform: Option<Matrix>,
        label: Option<String>,
        depth: usize,
    ) -> Result<()> {
        let kind = p.type_name()?.map(|s| s.to_string()).unwrap_or_default();
        let matrix = transform.unwrap_or(world(p, self.time)?);
        match kind.as_str() {
            "Mesh" | "Cube" | "Sphere" | "Cylinder" | "Cone" | "Capsule" | "Plane" => {
                self.mesh(p, &kind, matrix, label)?
            }
            "PointInstancer" => self.instancer(p, matrix, label, depth)?,
            "Camera" => self.camera(p, matrix)?,
            "DistantLight" | "SphereLight" | "RectLight" | "DiskLight" | "CylinderLight"
            | "DomeLight" => self.light(p, &kind, matrix)?,
            "BasisCurves" | "NurbsCurves" | "Volume" | "Points" => {
                self.warn(format!("{}: unsupported geometry {kind}", p.path()))
            }
            _ => {}
        }
        Ok(())
    }
    fn camera(&mut self, p: &Prim, m: Matrix) -> Result<()> {
        let eye = self.viewport(m.transform_point([0., 0., 0.].into()).into(), true);
        let direction =
            normalize(self.viewport(m.transform_vec([0., 0., -1.].into()).into(), false));
        let up = normalize(self.viewport(m.transform_vec([0., 1., 0.].into()).into(), false));
        let aperture = scalar(p, "verticalAperture", self.time, 15.2908)?;
        let focal = scalar(p, "focalLength", self.time, 50.)?;
        self.scene.scene.cameras.push(ImportedCamera {
            name: p.path().to_string(),
            eye,
            target: std::array::from_fn(|i| eye[i] + direction[i]),
            up,
            fovy: (2. * (aperture / (2. * focal.max(1e-6))).atan()).to_degrees(),
            orthographic: token(p, "projection", self.time, "")? == "orthographic",
            ortho_scale: aperture * 0.1 * self.scale,
        });
        Ok(())
    }
    fn light(&mut self, p: &Prim, kind: &str, m: Matrix) -> Result<()> {
        let n = |name: &str, default| scalar(p, &format!("inputs:{name}"), self.time, default);
        let axes: [[f32; 3]; 3] = std::array::from_fn(|j| {
            self.viewport(
                m.transform_vec(
                    std::array::from_fn::<_, 3, _>(|i| if i == j { 1. } else { 0. }).into(),
                )
                .into(),
                false,
            )
        });
        let lengths = axes.map(|a| a.iter().map(|v| v * v).sum::<f32>().sqrt());
        let [x, y, z] = axes.map(normalize);
        let pitch = (-x[2]).clamp(-1., 1.).asin();
        let roll = if pitch.cos().abs() > 1e-6 {
            y[2].atan2(z[2])
        } else {
            0.
        };
        let yaw = if pitch.cos().abs() > 1e-6 {
            x[1].atan2(x[0])
        } else {
            (-y[0]).atan2(y[1])
        };
        let mut mapped = match kind {
            "DistantLight" => "directional",
            "SphereLight" => "sphere",
            "RectLight" => "rectangle",
            "DiskLight" => "disk",
            "CylinderLight" => "tube",
            _ => "environment",
        };
        if kind == "SphereLight" && boolean(p, "inputs:treatAsPoint", self.time, false)? {
            mapped = "point";
        }
        let cone_angle = n("shaping:cone:angle", 180.)?;
        if cone_angle < 180. {
            mapped = "spot";
        }
        let mut light = ImportedLight {
            name: p.path().to_string(),
            kind: mapped.into(),
            position: self.viewport(m.transform_point([0., 0., 0.].into()).into(), true),
            rotation_degrees: [roll, pitch, yaw].map(f32::to_degrees),
            color: vector(p, "inputs:color", self.time, [1.; 3])?,
            intensity: n("intensity", 1.)?,
            exposure: n("exposure", 0.)?,
            temperature: n("colorTemperature", 6500.)?,
            use_temperature: boolean(p, "inputs:enableColorTemperature", self.time, false)?,
            radius: n("radius", 0.5)? * self.scale * lengths.into_iter().fold(0., f32::max),
            size: [
                n("width", n("length", 1.)?)? * self.scale * lengths[0],
                n("height", 1.)? * self.scale * lengths[1],
            ],
            cone_angle,
            cone_softness: n("shaping:cone:softness", 0.)?,
            diffuse: n("diffuse", 1.)?,
            specular: n("specular", 1.)?,
            environment_path: String::new(),
        };
        if kind == "DomeLight" {
            if let Some(sdf::Value::AssetPath(asset)) = value(p, "inputs:texture:file", self.time)?
            {
                match material::asset_bytes(&asset).and_then(|data| {
                    self.persist(
                        &data,
                        Path::new(asset.asset_path())
                            .extension()
                            .and_then(|s| s.to_str())
                            .unwrap_or("hdr"),
                    )
                }) {
                    Ok(path) => light.environment_path = path,
                    Err(e) => self.warn(format!("{}: environment texture: {e}", p.path())),
                }
            }
        }
        self.scene.scene.lights.push(light);
        Ok(())
    }
}

pub(crate) struct AnimationSession {
    stage: Stage,
    cache: PathBuf,
    transform_only: bool,
    sent_geometry: bool,
}
impl AnimationSession {
    pub fn new(path: &Path) -> Result<Self> {
        let stage = open(path)?;
        let mut transform_only = true;
        for path in paths(&stage)? {
            let p = stage.prim(path)?;
            if p.type_name()?.is_some_and(|t| {
                matches!(
                    t.as_str(),
                    "PointInstancer" | "SkelRoot" | "Skeleton" | "SkelAnimation"
                )
            }) {
                transform_only = false;
            }
            for attr in p.authored_attributes()? {
                if attr.time_samples()?.is_some_and(|s| !s.is_empty())
                    && !attr
                        .path()
                        .as_str()
                        .rsplit_once('.')
                        .map_or("", |(_, name)| name)
                        .starts_with("xformOp:")
                {
                    transform_only = false;
                }
            }
        }
        Ok(Self {
            stage,
            cache: crate::user_data::data_dir()?.join("usd-assets"),
            transform_only,
            sent_geometry: false,
        })
    }
    pub fn sample(&mut self, time: f64) -> Result<AnimationSample> {
        ensure!(time.is_finite(), "USD time code must be finite");
        if !self.transform_only {
            return Ok(AnimationSample::Geometry(
                evaluate(&self.stage, &self.cache, Some(time), false, false)?,
                None,
            ));
        }
        let scale = number(self.stage.stage_metadata("metersPerUnit")?, 0.01);
        let z_up = matches!(self.stage.stage_metadata("upAxis")?,Some(sdf::Value::Token(v)) if v.as_str()=="Z");
        // Local mesh preparation converts Y-up vertices to the editor's Z-up coordinates.
        let basis = if z_up {
            Matrix::scale([scale; 3])
        } else {
            Matrix::scale([scale; 3]) * Matrix::rotation_x(std::f64::consts::FRAC_PI_2)
        };
        let inverse = basis.inverse().context("Invalid USD units")?;
        let mut transforms = Transforms::new();
        for path in paths(&self.stage)? {
            let p = self.stage.prim(path)?;
            let m = inverse * world(&p, time)? * basis;
            transforms.insert(
                p.path().to_string(),
                std::array::from_fn(|i| std::array::from_fn(|j| m.0[i * 4 + j] as f32)),
            );
        }
        if self.sent_geometry {
            Ok(AnimationSample::Transforms(transforms))
        } else {
            let scene = evaluate(&self.stage, &self.cache, Some(time), true, false)?;
            self.sent_geometry = true;
            Ok(AnimationSample::Geometry(scene, Some(transforms)))
        }
    }
}
