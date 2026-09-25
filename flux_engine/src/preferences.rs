use crate::State;

impl State {
    pub(super) fn capture_user_preferences(&mut self) {
        self.preferences.background = self.editor.background;
        self.preferences.gizmo_at_bottom = self.gizmo_always_at_bottom;
        self.preferences.triangles = self.ground_plane.triangulate_subdivision;
        self.preferences.point_size = self.point_size;
        self.preferences.point_color = self.point_color;
        self.preferences.show_grid = self.editor.show_grid;
        self.preferences.grid_size = self.editor.grid_size;
        self.preferences.grid_spacing = self.editor.grid_spacing;
        self.preferences.grid_color = self.editor.grid_color;
    }

    pub(super) fn apply_user_preferences(&mut self) {
        let settings = &self.preferences;
        self.editor.background = settings.background;
        self.background_color = wgpu::Color {
            r: settings.background[0] as f64,
            g: settings.background[1] as f64,
            b: settings.background[2] as f64,
            a: settings.background[3] as f64,
        };
        self.gizmo_always_at_bottom = settings.gizmo_at_bottom;
        self.point_size = settings.point_size;
        self.point_color = settings.point_color;
        self.editor.show_grid = settings.show_grid;
        self.editor.grid_size = settings.grid_size;
        self.editor.grid_spacing = settings.grid_spacing;
        self.editor.grid_color = settings.grid_color;
        self.displayed_grid_spacing = 0.0;
        self.displayed_grid_extent = 0.0;
        self.grid.rebuild(
            &self.device,
            settings.grid_size,
            settings.grid_spacing,
            settings.grid_color,
        );
        for object in self
            .generated_objects
            .iter_mut()
            .chain(std::iter::once(&mut self.ground_plane))
        {
            if object.triangulate_subdivision != settings.triangles {
                object.triangulate_subdivision = settings.triangles;
                object.rebuild_shape(&self.device);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.asset_explorer.grid = settings.explorer_grid;
            self.asset_explorer.show_hidden = settings.explorer_hidden;
            self.asset_explorer.user_library = settings.explorer_library;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn import_texture_as_material(
        &mut self,
        path: &std::path::Path,
        kind: crate::GroupTextureKind,
    ) {
        let result = (|| -> anyhow::Result<crate::material_library::SceneMaterial> {
            use crate::GroupTextureKind;
            let texture = self.cached_material_texture(
                path,
                matches!(
                    kind,
                    GroupTextureKind::BaseColor | GroupTextureKind::Emissive
                ),
            )?;
            let mut material =
                crate::material::PbrMaterial::new_untextured(&self.device, &self.queue)?;
            match kind {
                GroupTextureKind::BaseColor => {
                    material.set_base_color_texture(&self.device, texture)
                }
                GroupTextureKind::Normal => material.set_normal_texture(&self.device, texture),
                GroupTextureKind::Roughness => {
                    material.set_metallic_roughness_texture(&self.device, texture)
                }
                GroupTextureKind::Emissive => {
                    material.set_emissive_texture(&self.device, texture);
                    material.uniform.options[1] = 1.0;
                }
            }
            let slot_path = |slot| {
                if slot == kind {
                    path.to_string_lossy().into_owned()
                } else {
                    String::new()
                }
            };
            Ok(crate::material_library::SceneMaterial {
                id: crate::material_library::MaterialId(self.next_material_id),
                name: path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                material,
                base_color_path: slot_path(GroupTextureKind::BaseColor),
                normal_path: slot_path(GroupTextureKind::Normal),
                roughness_path: slot_path(GroupTextureKind::Roughness),
                emissive_path: slot_path(GroupTextureKind::Emissive),
            })
        })();
        match result {
            Ok(entry) => {
                self.editor.status = format!(
                    "Imported {}. Save to User Library to reuse it in other scenes.",
                    entry.name
                );
                self.selected_library_material = entry.id;
                self.next_material_id += 1;
                self.material_library.push(entry);
            }
            Err(error) => self.editor.status = format!("Could not import texture: {error:#}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn save_user_material(&mut self, id: crate::material_library::MaterialId) {
        let Some(entry) = self.material_library.iter().find(|entry| entry.id == id) else {
            return;
        };
        let uniform = &entry.material.uniform;
        let material = crate::FxMaterial {
            transparency: uniform.transparency,
            color_adjustments: uniform.color_adjustments,
            emissive_color: uniform.emissive_color,
            base_color: uniform.base_color,
            properties: uniform.properties,
            options: uniform.options,
            inspection: uniform.inspection,
            base_color_path: entry.base_color_path.clone(),
            normal_path: entry.normal_path.clone(),
            roughness_path: entry.roughness_path.clone(),
            emissive_path: entry.emissive_path.clone(),
        };
        match self.user_library.save_material(&entry.name, material) {
            Ok(()) => self.editor.status = self.user_library.message.clone(),
            Err(error) => {
                self.user_library.message = format!("Material was not saved: {error:#}");
                self.editor.status = self.user_library.message.clone();
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn add_user_material_to_scene(&mut self, id: &str) {
        let result = (|| -> anyhow::Result<crate::material_library::SceneMaterial> {
            let preset = self.user_library.resolve(id)?;
            let saved = preset.material;
            let mut material =
                crate::material::PbrMaterial::new_untextured(&self.device, &self.queue)?;
            Self::apply_fx_material_uniform(&mut material.uniform, &saved);
            if !saved.base_color_path.is_empty() {
                let texture = self
                    .cached_material_texture(std::path::Path::new(&saved.base_color_path), true)?;
                material.set_base_color_texture(&self.device, texture);
            }
            if !saved.normal_path.is_empty() {
                let texture =
                    self.cached_material_texture(std::path::Path::new(&saved.normal_path), false)?;
                material.set_normal_texture(&self.device, texture);
            }
            if !saved.roughness_path.is_empty() {
                let texture = self
                    .cached_material_texture(std::path::Path::new(&saved.roughness_path), false)?;
                material.set_metallic_roughness_texture(&self.device, texture);
            }
            if !saved.emissive_path.is_empty() {
                let texture =
                    self.cached_material_texture(std::path::Path::new(&saved.emissive_path), true)?;
                material.set_emissive_texture(&self.device, texture);
            }
            Ok(crate::material_library::SceneMaterial {
                id: crate::material_library::MaterialId(self.next_material_id),
                name: preset.name,
                material,
                base_color_path: saved.base_color_path,
                normal_path: saved.normal_path,
                roughness_path: saved.roughness_path,
                emissive_path: saved.emissive_path,
            })
        })();
        match result {
            Ok(entry) => {
                self.user_library.message = format!("Added “{}” to this scene", entry.name);
                self.editor.status = self.user_library.message.clone();
                self.selected_library_material = entry.id;
                self.next_material_id += 1;
                self.material_library.push(entry);
                self.active_side_panel = Some(5);
            }
            Err(error) => {
                self.user_library.message = format!("Could not add material: {error:#}");
                self.editor.status = self.user_library.message.clone();
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for State {
    fn drop(&mut self) {
        self.preference_store.save(&self.preferences, false);
    }
}
