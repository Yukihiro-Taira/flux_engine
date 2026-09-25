//! Personal settings and material presets live outside projects and version control.
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Component, Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Preferences {
    pub background: [f32; 4],
    pub gizmo_at_bottom: bool,
    pub triangles: bool,
    pub point_size: f32,
    pub point_color: [f32; 4],
    pub show_grid: bool,
    pub grid_size: f32,
    pub grid_spacing: f32,
    pub grid_color: [f32; 4],
    pub explorer_grid: bool,
    pub explorer_hidden: bool,
    pub explorer_library: bool,
}

impl Default for Preferences {
    fn default() -> Self {
        let editor = crate::editor_ui::EditorUi::default();
        Self {
            background: editor.background,
            gizmo_at_bottom: false,
            triangles: false,
            point_size: 7.0,
            point_color: [0.0, 0.8, 0.72, 1.0],
            show_grid: editor.show_grid,
            grid_size: editor.grid_size,
            grid_spacing: editor.grid_spacing,
            grid_color: editor.grid_color,
            explorer_grid: true,
            explorer_hidden: false,
            explorer_library: false,
        }
    }
}

impl Preferences {
    fn sanitize(&mut self) {
        self.point_size = self.point_size.clamp(2.0, 24.0);
        self.grid_size = self.grid_size.clamp(1.0, 10_000.0);
        self.grid_spacing = self.grid_spacing.clamp(0.1, 10.0);
        for color in [
            &mut self.background,
            &mut self.point_color,
            &mut self.grid_color,
        ] {
            for channel in color {
                *channel = channel.clamp(0.0, 1.0);
            }
        }
    }
}

pub fn data_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("LEARN_WGPU_USER_DATA_DIR").filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    #[cfg(target_os = "macos")]
    let base =
        std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Application Support"));
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".local/share")));
    Ok(base
        .context("No user data directory is available")?
        .join("learn_wgpu"))
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(value)?;
    let parent = path.parent().context("Missing destination directory")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".write-{}-{}.tmp", std::process::id(), unique_id()));
    let result = (|| -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.with_context(|| format!("Could not save {}", path.display()))
}

fn unique_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    format!(
        "{:x}-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

pub struct PreferenceStore {
    pub saved: Preferences,
    pub error: Option<String>,
    root: Option<PathBuf>,
    failed: Option<Preferences>,
}

impl PreferenceStore {
    pub fn load() -> Self {
        Self::load_at(data_dir())
    }

    fn load_at(root: Result<PathBuf>) -> Self {
        let mut store = Self {
            saved: Preferences::default(),
            error: None,
            root: None,
            failed: None,
        };
        match root {
            Err(error) => store.error = Some(error.to_string()),
            Ok(root) => {
                match fs::read(root.join("settings.json")) {
                    Ok(bytes) => match serde_json::from_slice::<Preferences>(&bytes) {
                        Ok(mut settings) => {
                            settings.sanitize();
                            store.saved = settings;
                        }
                        Err(error) => {
                            store.error = Some(format!("Settings could not be read: {error}"))
                        }
                    },
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        store.error = Some(format!("Settings could not be read: {error}"))
                    }
                }
                store.root = Some(root);
            }
        }
        store
    }

    pub fn save(&mut self, settings: &Preferences, force: bool) {
        if !force && (self.saved == *settings || self.failed.as_ref() == Some(settings)) {
            return;
        }
        let result = self
            .root
            .as_ref()
            .context("No user data directory is available")
            .and_then(|root| atomic_json(&root.join("settings.json"), settings));
        match result {
            Ok(()) => {
                self.saved = settings.clone();
                self.error = None;
                self.failed = None;
            }
            Err(error) => {
                self.error = Some(format!("Settings are not saved: {error:#}"));
                self.failed = Some(settings.clone());
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct UserMaterial {
    pub id: String,
    pub name: String,
    pub material: crate::FxMaterial,
}

#[derive(Default, Serialize, Deserialize)]
struct Catalog {
    #[serde(default)]
    materials: Vec<UserMaterial>,
}

pub struct UserLibrary {
    pub materials: Vec<UserMaterial>,
    pub message: String,
    root: Option<PathBuf>,
    search: String,
    read_error: bool,
}

impl UserLibrary {
    pub fn load() -> Self {
        Self::load_at(data_dir())
    }

    fn load_at(root: Result<PathBuf>) -> Self {
        let mut library = Self {
            materials: Vec::new(),
            message: String::new(),
            root: None,
            search: String::new(),
            read_error: false,
        };
        match root {
            Err(error) => {
                library.message = error.to_string();
                library.read_error = true;
            }
            Ok(root) => {
                match fs::read(root.join("materials.json")) {
                    Ok(bytes) => match serde_json::from_slice::<Catalog>(&bytes) {
                        Ok(catalog) => library.materials = catalog.materials,
                        Err(error) => {
                            library.message = format!("Library could not be read: {error}");
                            library.read_error = true;
                        }
                    },
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        library.message = format!("Library could not be read: {error}");
                        library.read_error = true;
                    }
                }
                library.root = Some(root);
            }
        }
        library
    }

    pub fn save_material(&mut self, name: &str, mut material: crate::FxMaterial) -> Result<()> {
        if self.read_error {
            bail!(
                "Resolve the library read error before adding presets; the existing library has been preserved"
            );
        }
        let root = self
            .root
            .as_ref()
            .context("No user data directory is available")?;
        let id = unique_id();
        let relative = PathBuf::from("materials").join(&id);
        let directory = root.join(&relative);
        fs::create_dir_all(&directory)?;
        // Each preset owns its textures, so moving/deleting the original import is safe.
        let result = (|| -> Result<UserMaterial> {
            for (slot, source) in [
                ("base_color", &mut material.base_color_path),
                ("normal", &mut material.normal_path),
                ("metal_rough", &mut material.roughness_path),
                ("emissive", &mut material.emissive_path),
            ] {
                if source.trim().is_empty() {
                    continue;
                }
                let path = Path::new(source);
                let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("png");
                let filename = format!("{slot}.{extension}");
                fs::copy(path, directory.join(&filename))
                    .with_context(|| format!("Could not copy {}", path.display()))?;
                *source = relative.join(filename).to_string_lossy().into_owned();
            }
            Ok(UserMaterial {
                id,
                name: if name.trim().is_empty() {
                    "Untitled material".into()
                } else {
                    name.trim().into()
                },
                material,
            })
        })();
        let entry = match result {
            Ok(entry) => entry,
            Err(error) => {
                let _ = fs::remove_dir_all(&directory);
                return Err(error);
            }
        };
        let mut materials = self.materials.clone();
        materials.push(entry);
        if let Err(error) = atomic_json(
            &root.join("materials.json"),
            &Catalog {
                materials: materials.clone(),
            },
        ) {
            let _ = fs::remove_dir_all(&directory);
            return Err(error);
        }
        self.materials = materials;
        self.message = format!("Saved “{}” to your library", name.trim());
        Ok(())
    }

    pub fn resolve(&self, id: &str) -> Result<UserMaterial> {
        let root = self
            .root
            .as_ref()
            .context("No user data directory is available")?;
        let mut entry = self
            .materials
            .iter()
            .find(|entry| entry.id == id)
            .context("Material is no longer in the library")?
            .clone();
        for path in [
            &mut entry.material.base_color_path,
            &mut entry.material.normal_path,
            &mut entry.material.roughness_path,
            &mut entry.material.emissive_path,
        ] {
            if path.is_empty() {
                continue;
            }
            let relative = Path::new(path);
            if relative
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            {
                bail!("Invalid texture path in material preset");
            }
            *path = root.join(relative).to_string_lossy().into_owned();
        }
        Ok(entry)
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<String> {
        let mut load = None;
        ui.heading("User Material Library");
        ui.label("Reusable materials with local copies of their textures.");
        ui.add(
            egui::TextEdit::singleline(&mut self.search)
                .hint_text("Search saved materials…")
                .desired_width(f32::INFINITY),
        );
        if !self.message.is_empty() {
            ui.add(egui::Label::new(&self.message).wrap());
        }
        if self.read_error && ui.button("Reload library").clicked() {
            *self = Self::load();
        }
        let search = self.search.trim().to_lowercase();
        let entries: Vec<_> = self
            .materials
            .iter()
            .filter(|entry| entry.name.to_lowercase().contains(&search))
            .collect();
        ui.weak(format!("{} saved materials", entries.len()));
        if entries.is_empty() {
            ui.add_space(16.0);
            ui.strong(if search.is_empty() {
                "Build your material collection"
            } else {
                "No matching materials"
            });
            ui.add(egui::Label::new("In Materials, import texture maps and choose Save to User Library. Add a saved material to any scene from here.").wrap());
        }
        for entry in entries {
            ui.push_id(&entry.id, |ui| {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let color = entry
                            .material
                            .base_color
                            .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8);
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(36.0, 36.0), egui::Sense::hover());
                        ui.painter().rect_filled(
                            rect,
                            6.0,
                            egui::Color32::from_rgba_unmultiplied(
                                color[0], color[1], color[2], color[3],
                            ),
                        );
                        ui.vertical(|ui| {
                            ui.set_max_width((ui.available_width() - 100.0).max(60.0));
                            ui.add(
                                egui::Label::new(egui::RichText::new(&entry.name).strong())
                                    .truncate(),
                            )
                            .on_hover_text(&entry.name);
                            let maps = [
                                &entry.material.base_color_path,
                                &entry.material.normal_path,
                                &entry.material.roughness_path,
                                &entry.material.emissive_path,
                            ]
                            .iter()
                            .filter(|s| !s.is_empty())
                            .count();
                            ui.small(format!("{maps} texture maps"));
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Add to Scene").clicked() {
                                load = Some(entry.id.clone());
                            }
                        });
                    });
                });
            });
        }
        load
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sandbox(PathBuf);
    impl Sandbox {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("learn-wgpu-user-data-test-{}", unique_id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn material() -> crate::FxMaterial {
        crate::FxMaterial {
            transparency: crate::material::default_transparency(),
            color_adjustments: [0.0; 4],
            emissive_color: [1.0; 4],
            base_color: [0.2, 0.4, 0.6, 1.0],
            properties: [0.3, 0.7, 1.0, 0.0],
            options: [0.0; 4],
            inspection: [0.0; 4],
            base_color_path: String::new(),
            normal_path: String::new(),
            roughness_path: String::new(),
            emissive_path: String::new(),
        }
    }

    #[test]
    fn material_library_keeps_owned_textures_after_originals_are_removed() {
        let sandbox = Sandbox::new();
        let root = sandbox.0.join("user-data");
        let mut saved = material();
        for (name, path) in [
            ("color.png", &mut saved.base_color_path),
            ("normal.png", &mut saved.normal_path),
            ("rough.png", &mut saved.roughness_path),
            ("emission.png", &mut saved.emissive_path),
        ] {
            let source = sandbox.0.join(name);
            fs::write(&source, name.as_bytes()).unwrap();
            *path = source.display().to_string();
        }
        let mut library = UserLibrary::load_at(Ok(root.clone()));
        saved.options[1] = 3.5;
        saved.transparency = [2.0, 0.4, 0.0, 0.65];
        saved.color_adjustments = [45.0, -90.0, 1.0, 0.0];
        saved.emissive_color = [1.0, 0.2, 0.5, 1.0];
        library.save_material("Paint", saved).unwrap();
        let id = library.materials[0].id.clone();
        for name in ["color.png", "normal.png", "rough.png", "emission.png"] {
            fs::remove_file(sandbox.0.join(name)).unwrap();
        }
        let reopened = UserLibrary::load_at(Ok(root));
        let preset = reopened.resolve(&id).unwrap();
        assert_eq!(preset.name, "Paint");
        assert_eq!(preset.material.options[1], 3.5);
        assert_eq!(preset.material.transparency, [2.0, 0.4, 0.0, 0.65]);
        assert_eq!(preset.material.color_adjustments, [45.0, -90.0, 1.0, 0.0]);
        assert_eq!(preset.material.emissive_color, [1.0, 0.2, 0.5, 1.0]);
        assert_eq!(
            fs::read(preset.material.emissive_path).unwrap(),
            b"emission.png"
        );
        assert_eq!(preset.material.properties, [0.3, 0.7, 1.0, 0.0]);
        assert_eq!(
            fs::read(preset.material.base_color_path).unwrap(),
            b"color.png"
        );
        assert_eq!(
            fs::read(preset.material.normal_path).unwrap(),
            b"normal.png"
        );
        assert_eq!(
            fs::read(preset.material.roughness_path).unwrap(),
            b"rough.png"
        );
    }

    #[test]
    fn failed_copy_and_corrupt_catalog_preserve_existing_library() {
        let sandbox = Sandbox::new();
        let mut library = UserLibrary::load_at(Ok(sandbox.0.clone()));
        library.save_material("Working", material()).unwrap();
        let manifest = fs::read(sandbox.0.join("materials.json")).unwrap();
        let mut invalid = material();
        invalid.normal_path = sandbox.0.join("missing.png").display().to_string();
        assert!(library.save_material("Broken", invalid).is_err());
        assert_eq!(library.materials.len(), 1);
        assert_eq!(
            fs::read(sandbox.0.join("materials.json")).unwrap(),
            manifest
        );
        assert_eq!(
            fs::read_dir(sandbox.0.join("materials")).unwrap().count(),
            1
        );
        fs::write(sandbox.0.join("materials.json"), b"corrupt catalog").unwrap();
        let mut reopened = UserLibrary::load_at(Ok(sandbox.0.clone()));
        assert!(reopened.save_material("Replacement", material()).is_err());
        assert_eq!(
            fs::read(sandbox.0.join("materials.json")).unwrap(),
            b"corrupt catalog"
        );
    }

    #[test]
    fn preferences_survive_restart_and_reset_preserves_materials() {
        let sandbox = Sandbox::new();
        let root = sandbox.0.clone();
        let mut library = UserLibrary::load_at(Ok(root.clone()));
        library.save_material("Keep me", material()).unwrap();
        let mut store = PreferenceStore::load_at(Ok(root.clone()));
        let settings = Preferences {
            show_grid: false,
            grid_spacing: 2.5,
            gizmo_at_bottom: true,
            background: [0.1, 0.2, 0.3, 1.0],
            point_size: 12.0,
            explorer_library: true,
            explorer_grid: false,
            triangles: true,
            ..Preferences::default()
        };
        store.save(&settings, false);
        assert!(store.error.is_none());
        let mut reopened = PreferenceStore::load_at(Ok(root.clone()));
        assert_eq!(reopened.saved, settings);
        reopened.save(&Preferences::default(), true);
        assert_eq!(
            PreferenceStore::load_at(Ok(root.clone())).saved,
            Preferences::default()
        );
        assert_eq!(UserLibrary::load_at(Ok(root)).materials.len(), 1);
    }

    #[test]
    fn older_settings_get_defaults_and_bad_values_are_bounded() {
        let sandbox = Sandbox::new();
        fs::write(
            sandbox.0.join("settings.json"),
            br#"{"grid_spacing":0,"point_size":1000}"#,
        )
        .unwrap();
        let store = PreferenceStore::load_at(Ok(sandbox.0.clone()));
        assert_eq!(store.saved.grid_spacing, 0.1);
        assert_eq!(store.saved.point_size, 24.0);
        assert!(store.saved.explorer_grid);
        fs::write(sandbox.0.join("settings.json"), b"corrupt").unwrap();
        let mut store = PreferenceStore::load_at(Ok(sandbox.0.clone()));
        assert!(store.error.is_some());
        store.save(&Preferences::default(), false);
        assert_eq!(
            fs::read(sandbox.0.join("settings.json")).unwrap(),
            b"corrupt"
        );
    }

    #[test]
    fn preset_paths_cannot_escape_library_storage() {
        let sandbox = Sandbox::new();
        let mut library = UserLibrary::load_at(Ok(sandbox.0.clone()));
        let mut saved = material();
        saved.base_color_path = "../outside.png".into();
        library.materials.push(UserMaterial {
            id: "invalid".into(),
            name: "Invalid".into(),
            material: saved,
        });
        assert!(library.resolve("invalid").is_err());
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn library_view_stays_bounded_with_long_material_names() {
        let context = egui::Context::default();
        let mut explorer = crate::asset_explorer::AssetExplorer::default();
        explorer.user_library = true;
        let mut library = UserLibrary {
            materials: Vec::new(),
            message: String::new(),
            root: None,
            search: String::new(),
            read_error: false,
        };
        library.materials.push(UserMaterial {
            id: "preview".into(),
            name: "A very long reusable material name 世界 ".repeat(20),
            material: crate::FxMaterial {
            transparency: crate::material::default_transparency(),
                color_adjustments: [0.0; 4],
                emissive_color: [1.0; 4],
                base_color: [1.0; 4],
                properties: [0.0; 4],
                options: [0.0; 4],
                inspection: [0.0; 4],
                base_color_path: String::new(),
                normal_path: String::new(),
                roughness_path: String::new(),
                emissive_path: String::new(),
            },
        });
        for _ in 0..30 {
            let _ = context.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(480.0, 600.0),
                    )),
                    ..Default::default()
                },
                |ui| {
                    explorer.show_with_library(ui.ctx(), &mut true, |ui| {
                        library.ui(ui);
                    })
                },
            );
            let rect = context
                .memory(|m| m.area_rect(egui::Id::new(("workspace_panel_v1", "Asset Explorer"))))
                .unwrap();
            assert!(
                rect.width() <= 480.0,
                "Library expanded outside the screen: {rect:?}"
            );
        }
    }
}
