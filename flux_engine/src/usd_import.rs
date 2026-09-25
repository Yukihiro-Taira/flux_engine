//! OpenUSD is the composition/decoding backend; no USD syntax is parsed ad hoc.
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

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
fn runtime_python() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("LEARN_WGPU_USD_PYTHON").filter(|p| !p.is_empty()) {
        return Ok(path.into());
    }
    let python = if cfg!(windows) {
        "Scripts/python.exe"
    } else {
        "bin/python"
    };
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join(".usd-runtime"));
    }
    if let Ok(executable) = std::env::current_exe() {
        for parent in executable.ancestors().skip(1).take(4) {
            roots.push(parent.join(".usd-runtime"));
            roots.push(parent.join("usd-runtime"));
        }
    }
    if let Ok(data) = crate::user_data::data_dir() {
        roots.push(data.join("usd-runtime"));
    }
    for root in roots {
        let candidate = root.join(python);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    // An explicitly configured PATH interpreter can also carry official pxr bindings.
    Ok(PathBuf::from(if cfg!(windows) {
        "python"
    } else {
        "python3"
    }))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load(path: &Path, time_code: Option<f64>) -> Result<UsdScene> {
    anyhow::ensure!(
        time_code.is_none_or(f64::is_finite),
        "USD time code must be finite"
    );
    let source = std::fs::canonicalize(path).context("USD file does not exist")?;
    let cache = crate::user_data::data_dir()?.join("usd-assets");
    load_with_runtime(&runtime_python()?, &source, &cache, time_code)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_with_runtime(
    python: &Path,
    source: &Path,
    cache: &Path,
    time_code: Option<f64>,
) -> Result<UsdScene> {
    let mut child = Command::new(python)
        // The script is compiled into the application, never read from the asset folder.
        .arg("-I").arg("-").arg(source).arg(cache)
        .arg(time_code.map_or_else(|| "default".into(), |time| time.to_string()))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().with_context(|| format!("Could not start OpenUSD runtime {}. Run python3 scripts/setup_usd.py, or set LEARN_WGPU_USD_PYTHON to a Python with usd-core and Pillow installed", python.display()))?;
    let write_result = child
        .stdin
        .take()
        .context("Missing USD worker input")?
        .write_all(include_bytes!("../scripts/usd_import.py"));
    let output = child
        .wait_with_output()
        .context("OpenUSD worker stopped unexpectedly")?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        if stderr.contains("No module named") {
            bail!(
                "OpenUSD runtime is not installed. Run python3 scripts/setup_usd.py, or set LEARN_WGPU_USD_PYTHON to a Python with usd-core and Pillow. {stderr}"
            );
        }
        bail!(
            "OpenUSD could not import {}: {}",
            source.display(),
            stderr.trim()
        );
    }
    write_result.context("Could not send importer to OpenUSD")?;
    let scene: UsdScene =
        serde_json::from_slice(&output.stdout).context("Invalid OpenUSD mesh response")?;
    anyhow::ensure!(
        !scene.meshes.is_empty()
            || !scene.scene.lights.is_empty()
            || !scene.scene.cameras.is_empty(),
        "USD stage contains no supported scene objects"
    );
    for mesh in &scene.meshes {
        anyhow::ensure!(
            mesh.material < scene.materials.len(),
            "USD mesh has an invalid material binding"
        );
    }
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the OpenUSD runtime; run scripts/setup_usd.py first"]
    fn official_openusd_runtime_passes_the_rust_interchange_boundary() {
        let root = std::env::temp_dir().join(format!(
            "wgpu-usd-rust-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("fixture.usda");
        std::fs::write(
            &source,
            r#"#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Cube "Box" {
    double size = 2
}
def Camera "View" {
    float focalLength = 35
}
def RectLight "Key" {
    float inputs:intensity = 10
}
"#,
        )
        .unwrap();
        let scene = load_with_runtime(
            &runtime_python().unwrap(),
            &source,
            &root.join("assets"),
            None,
        )
        .unwrap();
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.meshes[0].indices.len(), 36);
        assert!(scene.materials[0].pbr.is_some());
        assert_eq!(scene.scene.cameras.len(), 1);
        assert_eq!(scene.scene.lights[0].kind, "rectangle");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn all_usd_extensions_are_recognized_case_insensitively() {
        for extension in ["usd", "usda", "usdc", "usdz", "USDZ", "UsDc"] {
            let path = PathBuf::from(format!("asset.{extension}"));
            assert!(is_usd(&path));
            assert!(supported_model(&path));
        }
        assert!(!is_usd(Path::new("asset.obj")));
        assert!(!supported_model(Path::new("asset.txt")));
    }
}

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
pub(crate) struct AnimationSession {
    child: std::process::Child,
    output: std::io::BufReader<std::process::ChildStdout>,
}
#[cfg(not(target_arch = "wasm32"))]
impl AnimationSession {
    pub fn new(path: &Path) -> Result<Self> {
        let cache = crate::user_data::data_dir()?.join("usd-assets");
        let mut child = Command::new(runtime_python()?).arg("-I").arg("-c")
            .arg(include_str!("../scripts/usd_import.py")).arg(path).arg(cache).arg("--serve")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn()?;
        let output = std::io::BufReader::new(child.stdout.take().context("Missing USD output")?);
        Ok(Self {child, output})
    }
    pub fn sample(&mut self, time: f64) -> Result<AnimationSample> {
        use std::io::BufRead;
        writeln!(self.child.stdin.as_mut().context("Missing USD input")?, "{}", serde_json::json!({"time":time,"transforms":true}))?;
        let mut response = String::new();
        self.output.read_line(&mut response)?;
        let value: serde_json::Value = serde_json::from_str(&response).context("USD animation worker stopped")?;
        if let Some(error) = value.get("error") { bail!("USD animation: {error}"); }
        let transforms=value.get("transforms").map(|t|serde_json::from_value(t.clone())).transpose()?;
        if value.get("meshes").is_some() {
            Ok(AnimationSample::Geometry(serde_json::from_value(value)?,transforms))
        } else { Ok(AnimationSample::Transforms(transforms.context("Missing animated transforms")?)) }
    }

}
#[cfg(not(target_arch = "wasm32"))]
impl Drop for AnimationSession {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum AnimationSample {
    Transforms(std::collections::HashMap<String, [[f32;4];4]>),
    Geometry(UsdScene, Option<std::collections::HashMap<String, [[f32;4];4]>>),
}
