//! Native Rust USD decoding, composition, and scene conversion.
use serde::Deserialize;
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
#[path = "usd_native.rs"]
mod native;

pub const MODEL_EXTENSIONS: &[&str] = &["obj", "fbx", "usd", "usda", "usdc", "usdz"];

pub fn is_usd(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()).is_some_and(|s| {
        matches!(
            s.to_ascii_lowercase().as_str(),
            "usd" | "usda" | "usdc" | "usdz"
        )
    })
}

pub fn supported_model(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()).is_some_and(|s| {
        MODEL_EXTENSIONS
            .iter()
            .any(|extension| s.eq_ignore_ascii_case(extension))
    })
}

#[derive(Deserialize)]
pub struct UsdScene {
    pub meshes: Vec<UsdMesh>,
    pub materials: Vec<crate::model::MaterialSource>,
    pub warnings: Vec<String>,
    pub time_code: f64,
    pub scene: ImportedScene,
}

#[derive(Deserialize)]
pub struct UsdMesh {
    pub name: String,
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub texcoords: Vec<f32>,
    pub indices: Vec<u32>,
    pub material: usize,
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::load;

#[derive(Clone, Default, serde::Serialize, Deserialize)]
pub struct ImportedScene {
    #[serde(default)]
    pub animation: Option<AnimationClip>,
    pub lights: Vec<ImportedLight>,
    pub cameras: Vec<ImportedCamera>,
}

#[derive(Clone, serde::Serialize, Deserialize)]
pub struct ImportedCamera {
    pub name: String,
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fovy: f32,
    pub orthographic: bool,
    pub ortho_scale: f32,
}

#[derive(Clone, serde::Serialize, Deserialize)]
pub struct ImportedLight {
    pub name: String,
    pub kind: String,
    pub position: [f32; 3],
    pub rotation_degrees: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub exposure: f32,
    pub temperature: f32,
    pub use_temperature: bool,
    pub radius: f32,
    pub size: [f32; 2],
    pub cone_angle: f32,
    pub cone_softness: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub environment_path: String,
}

#[derive(Clone, Debug, serde::Serialize, Deserialize)]
pub struct AnimationClip {
    pub start: f64,
    pub end: f64,
    pub rate: f64,
    pub animated: bool,
    pub name: String,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::AnimationSession;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum AnimationSample {
    Transforms(std::collections::HashMap<String, [[f32; 4]; 4]>),
    Geometry(
        UsdScene,
        Option<std::collections::HashMap<String, [[f32; 4]; 4]>>,
    ),
}
