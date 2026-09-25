use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct EditorUi {
    pub usd_use_stage_start: bool,
    pub usd_time_code: f64,
    pub selected_instance: usize,
    pub visible_instance_count: usize,
    pub texture_enabled: bool,
    pub background: [f32; 4],
    pub pending_asset: Option<PathBuf>,
    pub model_path_input: String,
    pub pending_texture: Option<PathBuf>,
    pub pending_normal: Option<PathBuf>,
    pub pending_metallic_roughness: Option<PathBuf>,
    pub pending_emissive: Option<PathBuf>,
    pub mtl_path_input: String,
    pub asset_name: String,
    pub texture_name: String,
    pub base_color_path: String,
    pub normal_path: String,
    pub metallic_roughness_path: String,
    pub emissive_path: String,
    pub pending_hdri: Option<PathBuf>,
    pub hdri_name: Option<String>,
    pub hdri_path: String,
    pub hdri_image_disabled: bool,
    pub hdri_intensity: f32,
    pub hdri_exposure: f32,
    pub hdri_rotation: f32,

    pub show_grid: bool,
    pub grid_size: f32,
    pub grid_spacing: f32,
    pub grid_color: [f32; 4],

    pub status: String,
}

impl Default for EditorUi {
    fn default() -> Self {
        Self {
            usd_use_stage_start: true,
            usd_time_code: 0.0,
            selected_instance: 0,
            visible_instance_count: 0,
            texture_enabled: false,
            background: [0.0, 0.0, 0.0, 1.0],
            pending_asset: None,
            model_path_input: String::new(),
            pending_texture: None,
            pending_normal: None,
            pending_metallic_roughness: None,
            pending_emissive: None,
            mtl_path_input: String::new(),
            asset_name: "No model loaded".into(),
            texture_name: "None".into(),
            base_color_path: String::new(),
            normal_path: String::new(),
            metallic_roughness_path: String::new(),
            emissive_path: String::new(),
            pending_hdri: None,
            hdri_name: None,
            hdri_path: String::new(),
            hdri_image_disabled: false,
            hdri_intensity: 1.0,
            hdri_exposure: 0.0,
            hdri_rotation: 0.0,
            show_grid: true,
            grid_size: 20.0,
            // One world unit is one meter.
            grid_spacing: 1.0,
            grid_color: [0.55, 0.58, 0.62, 0.15],
            status: "Ready".into(),
        }
    }
}

/// Fixed-height catalog row; its name and thumbnail share one click/drag target.
pub(crate) fn material_catalog_row(
    ui: &mut egui::Ui,
    id: u64,
    name: &str,
    selected: bool,
    preview: Option<egui::TextureId>,
) -> egui::Response {
    ui.push_id(id, |ui| {
        let (rect, mut response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 52.0),
            egui::Sense::click_and_drag(),
        );
        if ui.is_rect_visible(rect) {
            let fill = if selected {
                ui.visuals().selection.bg_fill
            } else if response.hovered() {
                ui.visuals().widgets.hovered.bg_fill
            } else {
                ui.visuals().faint_bg_color
            };
            ui.painter().rect_filled(rect, 5.0, fill);
            let thumbnail =
                egui::Rect::from_min_size(rect.min + egui::vec2(4.0, 4.0), egui::vec2(44.0, 44.0));
            if let Some(texture) = preview {
                ui.painter().image(
                    texture,
                    thumbnail,
                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            } else {
                ui.painter()
                    .rect_filled(thumbnail, 4.0, ui.visuals().widgets.inactive.bg_fill);
            }
            let label = egui::Rect::from_min_max(
                rect.min + egui::vec2(58.0, 8.0),
                rect.max - egui::vec2(8.0, 8.0),
            );
            response = response.union(
                ui.put(
                    label,
                    egui::Label::new(name)
                        .truncate()
                        .halign(egui::Align::Min)
                        .sense(egui::Sense::click_and_drag()),
                ),
            );
        }
        response.on_hover_text(name)
    })
    .inner
}

#[cfg(test)]
mod material_catalog_tests {
    use super::*;
    #[test]
    fn thumbnail_and_long_name_both_select_without_expanding_the_row() {
        for x in [24.0, 130.0] {
            let ctx = egui::Context::default();
            let name = "Long material name ".repeat(100);
            let frame = |events| {
                let mut response = None;
                let _ = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(420.0, 200.0),
                        )),
                        events,
                        ..Default::default()
                    },
                    |ui| {
                        response = Some(material_catalog_row(ui, 1, &name, false, None));
                    },
                );
                response.unwrap()
            };
            frame(vec![]);
            let initial = frame(vec![]);
            assert!(initial.rect.width() <= 420.0);
            assert_eq!(initial.rect.height(), 52.0);
            let pos = initial.rect.min + egui::vec2(x, 26.0);
            frame(vec![egui::Event::PointerMoved(pos)]);
            frame(vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            }]);
            let released = frame(vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }]);
            assert!(
                released.clicked(),
                "row should select when clicked at x={x}"
            );
        }
    }
}
