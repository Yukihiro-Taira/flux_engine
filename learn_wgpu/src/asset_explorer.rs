//! Local asset browser. All file-system reads happen on navigation or refresh.
use egui::{Align2, Color32, FontId, Sense, Stroke, StrokeKind, Vec2};
use std::path::{Path, PathBuf};

struct Entry {
    path: PathBuf,
    name: String,
    directory: bool,
    size: Option<u64>,
}

pub struct AssetExplorer {
    path: PathBuf,
    history: Vec<PathBuf>,
    history_index: usize,
    search: String,
    selected: Option<PathBuf>,
    pub grid: bool,
    pub show_hidden: bool,
    pub user_library: bool,
    entries: Vec<Entry>,
    error: Option<String>,
    loaded: bool,
}

impl Default for AssetExplorer {
    fn default() -> Self {
        let path = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            history: vec![path.clone()],
            path,
            history_index: 0,
            search: String::new(),
            selected: None,
            user_library: false,
            grid: true,
            show_hidden: false,
            entries: Vec::new(),
            error: None,
            loaded: false,
        }
    }
}

impl AssetExplorer {
    fn reload(&mut self) {
        self.loaded = true;
        self.entries.clear();
        self.error = None;
        match std::fs::read_dir(&self.path) {
            Err(error) => self.error = Some(format!("Cannot read this folder: {error}")),
            Ok(entries) => {
                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            // Follow directory symlinks so shortcuts remain navigable.
                            let metadata = std::fs::metadata(entry.path()).ok();
                            self.entries.push(Entry {
                                name: entry.file_name().to_string_lossy().into_owned(),
                                path: entry.path(),
                                directory: metadata.as_ref().is_some_and(|m| m.is_dir()),
                                size: metadata.filter(|m| m.is_file()).map(|m| m.len()),
                            });
                        }
                        Err(error) => {
                            self.error = Some(format!("Some entries could not be read: {error}"))
                        }
                    }
                }
                self.entries
                    .sort_by_cached_key(|e| (!e.directory, e.name.to_lowercase(), e.name.clone()));
            }
        }
        if self
            .selected
            .as_ref()
            .is_some_and(|path| !self.entries.iter().any(|e| &e.path == path))
        {
            self.selected = None;
        }
    }

    fn navigate(&mut self, path: PathBuf) {
        if self.path == path {
            return;
        }
        self.history.truncate(self.history_index + 1);
        self.history.push(path.clone());
        self.history_index = self.history.len() - 1;
        self.set_path(path);
    }

    fn set_path(&mut self, path: PathBuf) {
        self.path = path;
        self.search.clear();
        self.selected = None;
        self.reload();
    }

    fn history_step(&mut self, forward: bool) {
        self.history_index = if forward {
            (self.history_index + 1).min(self.history.len() - 1)
        } else {
            self.history_index.saturating_sub(1)
        };
        self.set_path(self.history[self.history_index].clone());
    }

    #[cfg(test)]
    pub fn show(&mut self, context: &egui::Context, open: &mut bool) {
        self.show_with_library(context, open, |_| {});
    }

    pub fn show_with_library(
        &mut self,
        context: &egui::Context,
        open: &mut bool,
        mut library_ui: impl FnMut(&mut egui::Ui),
    ) {
        if !*open {
            self.loaded = false;
            return;
        }
        if !self.loaded {
            self.reload();
        }
        let mut navigate = None;
        let mut history = None;
        let mut refresh = false;
        let favorites = favorites();
        let screen = context.content_rect().size();
        egui::Window::new("Asset Explorer")
            .id(egui::Id::new("asset_explorer_polished"))
            .open(open)
            .default_pos(egui::pos2(24.0, 40.0))
            .default_size(egui::vec2(680.0, 500.0))
            .min_size(egui::vec2(340.0, 260.0))
            .max_size((screen - egui::vec2(32.0, 64.0)).max(egui::vec2(340.0, 260.0)))
            .resizable(true)
            .show(context, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.user_library, false, "Files");
                    ui.selectable_value(&mut self.user_library, true, "User Library");
                });
                ui.separator();
                if self.user_library {
                    egui::ScrollArea::vertical().id_salt("user_material_library").auto_shrink([false, false]).show(ui, |ui| library_ui(ui));
                    return;
                }
                ui.horizontal(|ui| {
                    if ui.add_enabled(self.history_index > 0, egui::Button::new("‹")).on_hover_text("Back").clicked() { history = Some(false); }
                    if ui.add_enabled(self.history_index + 1 < self.history.len(), egui::Button::new("›")).on_hover_text("Forward").clicked() { history = Some(true); }
                    if ui.add_enabled(self.path.parent().is_some(), egui::Button::new("↑")).on_hover_text("Parent folder").clicked() { navigate = self.path.parent().map(Path::to_path_buf); }
                    if ui.button("Refresh").on_hover_text("Reload the folder from disk").clicked() { refresh = true; }
                    ui.menu_button("Places", |ui| {
                        for (name, path) in &favorites {
                            if ui.button(name).clicked() { navigate = Some(path.clone()); ui.close(); }
                        }
                        ui.separator();
                        if ui.button("Choose folder…").clicked() {
                            navigate = rfd::FileDialog::new().set_directory(&self.path).pick_folder();
                            ui.close();
                        }
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.selectable_value(&mut self.grid, true, "Grid");
                        ui.selectable_value(&mut self.grid, false, "List");
                    });
                });
                // One bounded path line instead of an ever-growing breadcrumb toolbar.
                ui.horizontal(|ui| {
                    ui.menu_button("Path", |ui| {
                        for path in self.path.ancestors() {
                            let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string());
                            if ui.button(name).on_hover_text(path.display().to_string()).clicked() { navigate = Some(path.to_path_buf()); ui.close(); }
                        }
                        ui.separator();
                        if ui.button("Copy full path").clicked() { ui.ctx().copy_text(self.path.display().to_string()); ui.close(); }
                    });
                    ui.add(egui::Label::new(self.path.display().to_string()).truncate()).on_hover_text(self.path.display().to_string());
                });
                ui.separator();
                let body_height = (ui.available_height() - 34.0).max(100.0);
                let wide = ui.available_width() >= 520.0;
                // Allocate the body once. Child content cannot enlarge the window.
                let (body_rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), body_height), Sense::hover());
                let mut main_rect = body_rect;
                if wide {
                    let sidebar_rect = egui::Rect::from_min_size(body_rect.min, egui::vec2(128.0, body_height));
                    let mut sidebar = ui.new_child(egui::UiBuilder::new().max_rect(sidebar_rect).layout(egui::Layout::top_down(egui::Align::Min)));
                    sidebar.set_clip_rect(sidebar_rect.intersect(ui.clip_rect()));
                    sidebar.weak("PLACES");
                    for (name, path) in &favorites {
                        if sidebar.add_sized([128.0, 28.0], egui::Button::new(name).selected(self.path == *path).frame(false)).clicked() { navigate = Some(path.clone()); }
                    }
                    sidebar.add_space(12.0);
                    sidebar.weak("WORKFLOW");
                    sidebar.add(egui::Label::new("Double-click a folder to browse. Drag files onto compatible editor fields.").wrap());
                    let x = body_rect.left() + 140.0;
                    ui.painter().vline(x, body_rect.y_range(), ui.visuals().widgets.noninteractive.bg_stroke);
                    main_rect.min.x = x + 12.0;
                }
                {
                    let mut main = ui.new_child(egui::UiBuilder::new().max_rect(main_rect).layout(egui::Layout::top_down(egui::Align::Min)));
                    main.set_clip_rect(main_rect.intersect(ui.clip_rect()));
                    let ui = &mut main;
                    let width = main_rect.width();
                        ui.set_min_width(width);
                        ui.horizontal(|ui| {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.checkbox(&mut self.show_hidden, "Hidden").on_hover_text("Show dotfiles and hidden folders");
                                if ui.add_enabled(!self.search.is_empty(), egui::Button::new("×")).on_hover_text("Clear search").clicked() { self.search.clear(); }
                                ui.add_sized([ui.available_width(), 26.0], egui::TextEdit::singleline(&mut self.search).hint_text("Search this folder…"));
                            });
                        });
                        let filter = self.search.trim().to_lowercase();
                        let visible: Vec<_> = self.entries.iter().filter(|e| (self.show_hidden || !e.name.starts_with('.')) && (filter.is_empty() || e.name.to_lowercase().contains(&filter))).collect();
                        ui.weak(format!("{} items", visible.len()));
                        if let Some(error) = &self.error {
                            ui.add(egui::Label::new(egui::RichText::new(error).color(ui.visuals().warn_fg_color)).wrap());
                        }
                        let height = ui.available_height().max(1.0);
                        if visible.is_empty() {
                            ui.allocate_ui(egui::vec2(width, height), |ui| {
                                ui.add_space(24.0);
                                ui.vertical_centered(|ui| {
                                    ui.strong(if self.error.is_some() { "Folder unavailable" } else if filter.is_empty() { "This folder is empty" } else { "No matching files" });
                                    ui.weak(if filter.is_empty() { "Choose another folder or refresh to try again." } else { "Try another name or clear the search." });
                                });
                            });
                        } else {
                            let columns = if self.grid { ((width - 16.0) / 112.0).floor().max(1.0) as usize } else { 1 };
                            let row_height = if self.grid { 104.0 } else { 34.0 };
                            let rows = visible.len().div_ceil(columns);
                            egui::ScrollArea::vertical().id_salt((&self.path, &filter, self.grid, self.show_hidden)).auto_shrink([false, false]).max_height(height)
                                .show_rows(ui, row_height, rows, |ui, range| {
                                    let tile_width = (ui.available_width() - (columns - 1) as f32 * 8.0) / columns as f32;
                                    for row in range {
                                        ui.horizontal(|ui| {
                                            for entry in visible.iter().skip(row * columns).take(columns) {
                                                let response = entry_widget(ui, entry, self.selected.as_ref() == Some(&entry.path), self.grid, egui::vec2(tile_width, row_height));
                                                if response.clicked() { self.selected = Some(entry.path.clone()); }
                                                if entry.directory && (response.double_clicked() || (response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))) { navigate = Some(entry.path.clone()); }
                                                response.context_menu(|ui| {
                                                    if entry.directory && ui.button("Open folder").clicked() { navigate = Some(entry.path.clone()); ui.close(); }
                                                    if ui.button("Copy path").clicked() { ui.ctx().copy_text(entry.path.display().to_string()); ui.close(); }
                                                });
                                            }
                                        });
                                    }
                                });
                        }
                }
                ui.separator();
                let status = self.selected.as_ref().map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned()).unwrap_or_else(|| "Select a file · Drag to an editor field to use it".to_owned());
                ui.add(egui::Label::new(egui::RichText::new(&status).small().weak()).truncate()).on_hover_text(self.selected.as_ref().map(|p| p.display().to_string()).unwrap_or(status));
            });
        if let Some(path) = egui::DragAndDrop::payload::<PathBuf>(context)
            && let Some(pointer) = context.pointer_interact_pos()
        {
            context.set_cursor_icon(egui::CursorIcon::Grabbing);
            egui::Area::new(egui::Id::new("asset_explorer_drag_preview"))
                .order(egui::Order::Tooltip)
                .interactable(false)
                .fixed_pos(pointer + egui::vec2(16.0, 16.0))
                .show(context, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.set_max_width(240.0);
                        ui.add(
                            egui::Label::new(
                                path.file_name().unwrap_or_default().to_string_lossy(),
                            )
                            .truncate(),
                        );
                    });
                });
        }
        if let Some(path) = navigate {
            self.navigate(path);
        } else if let Some(forward) = history {
            self.history_step(forward);
        } else if refresh {
            self.reload();
        }
    }
}

fn favorites() -> Vec<(String, PathBuf)> {
    let mut result = Vec::new();
    if let Ok(path) = std::env::current_dir() {
        result.push(("Workspace".into(), path));
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        result.push(("Home".into(), home.clone()));
        for name in ["Desktop", "Documents", "Downloads"] {
            let path = home.join(name);
            if path.is_dir() {
                result.push((name.into(), path));
            }
        }
    }
    result
}

fn file_kind(entry: &Entry) -> (&'static str, Color32) {
    if entry.directory {
        return ("DIR", Color32::from_rgb(231, 184, 91));
    }
    match entry
        .path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "obj" | "fbx" | "usd" | "usda" | "usdc" | "usdz" => ("3D", Color32::from_rgb(124, 180, 244)),
        "hdr" | "exr" | "rat" => ("HDR", Color32::from_rgb(192, 159, 241)),
        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "tga" | "bmp" => {
            ("IMG", Color32::from_rgb(123, 202, 168))
        }
        "fx" => ("FX", Color32::from_rgb(112, 195, 231)),
        "mtl" | "json" => ("MAT", Color32::from_rgb(231, 159, 123)),
        _ => ("FILE", Color32::from_gray(160)),
    }
}

fn entry_widget(
    ui: &mut egui::Ui,
    entry: &Entry,
    selected: bool,
    grid: bool,
    size: Vec2,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
    let response = response.on_hover_text(format!(
        "{}\n{}",
        entry.path.display(),
        if entry.directory {
            "Folder".into()
        } else {
            entry
                .size
                .map(format_size)
                .unwrap_or_else(|| "Size unavailable".into())
        }
    ));
    response.widget_info(|| {
        egui::WidgetInfo::selected(
            egui::WidgetType::SelectableLabel,
            ui.is_enabled(),
            selected,
            &entry.name,
        )
    });
    if !entry.directory {
        response.dnd_set_drag_payload(entry.path.clone());
    }
    if ui.is_rect_visible(rect) {
        let painter = ui.painter_at(rect);
        let visuals = ui.style().interact_selectable(&response, selected);
        let fill = if selected || response.hovered() {
            visuals.bg_fill
        } else {
            ui.visuals().faint_bg_color
        };
        painter.rect(
            rect.shrink(1.0),
            6.0,
            fill,
            if selected || response.has_focus() {
                Stroke::new(1.0, ui.visuals().selection.stroke.color)
            } else {
                Stroke::NONE
            },
            StrokeKind::Inside,
        );
        let (kind, color) = file_kind(entry);
        if grid {
            let badge = egui::Rect::from_center_size(
                rect.center_top() + egui::vec2(0.0, 31.0),
                egui::vec2(48.0, 34.0),
            );
            painter.rect_filled(badge, 6.0, color.gamma_multiply(0.15));
            painter.text(
                badge.center(),
                Align2::CENTER_CENTER,
                kind,
                FontId::proportional(13.0),
                color,
            );
            elided_text(
                &painter,
                &entry.name,
                rect.left_top() + egui::vec2(8.0, 60.0),
                rect.width() - 16.0,
                visuals.text_color(),
                12.0,
            );
            painter.text(
                rect.center_bottom() - egui::vec2(0.0, 14.0),
                Align2::CENTER_CENTER,
                if entry.directory {
                    "Folder".into()
                } else {
                    entry.size.map(format_size).unwrap_or_default()
                },
                FontId::proportional(10.0),
                ui.visuals().weak_text_color(),
            );
        } else {
            painter.text(
                rect.left_center() + egui::vec2(10.0, 0.0),
                Align2::LEFT_CENTER,
                kind,
                FontId::proportional(10.0),
                color,
            );
            elided_text(
                &painter,
                &entry.name,
                rect.left_top() + egui::vec2(52.0, 9.0),
                (rect.width() - 138.0).max(20.0),
                visuals.text_color(),
                12.0,
            );
            painter.text(
                rect.right_center() - egui::vec2(10.0, 0.0),
                Align2::RIGHT_CENTER,
                if entry.directory {
                    "Folder".into()
                } else {
                    entry.size.map(format_size).unwrap_or_default()
                },
                FontId::proportional(11.0),
                ui.visuals().weak_text_color(),
            );
        }
    }
    response
}

fn elided_text(
    painter: &egui::Painter,
    text: &str,
    position: egui::Pos2,
    width: f32,
    color: Color32,
    size: f32,
) {
    let mut job = egui::text::LayoutJob::simple_singleline(
        text.to_owned(),
        FontId::proportional(size),
        color,
    );
    job.wrap.max_width = width;
    job.wrap.max_rows = 1;
    job.wrap.break_anywhere = true;
    let galley = painter.layout_job(job);
    painter.galley(position, galley, color);
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_names_and_large_folders_do_not_expand_window() {
        for grid in [false, true] {
            let context = egui::Context::default();
            let mut explorer = AssetExplorer::default();
            explorer.loaded = true;
            explorer.grid = grid;
            explorer.path = PathBuf::from(format!("/{}", "long folder name/".repeat(30)));
            explorer.entries = (0..10_000)
                .map(|i| Entry {
                    path: explorer.path.join(format!("{i}.obj")),
                    name: format!("{i} {}.obj", "long filename 世界 ".repeat(30)),
                    directory: false,
                    size: Some(1234),
                })
                .collect();
            for _ in 0..30 {
                let output = context.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1440.0, 900.0),
                        )),
                        ..Default::default()
                    },
                    |ui| explorer.show(ui.ctx(), &mut true),
                );
                let rect = context
                    .memory(|m| m.area_rect(egui::Id::new("asset_explorer_polished")))
                    .unwrap();
                assert!(rect.width() <= 710.0, "grid={grid} window grew: {rect:?}");
                assert!(rect.height() <= 560.0, "grid={grid} window grew: {rect:?}");
                assert!(
                    output.shapes.len() < 600,
                    "offscreen entries should not be painted"
                );
            }
        }
    }

    #[test]
    fn history_navigation_clears_selection_and_discards_forward_branch() {
        let mut explorer = AssetExplorer::default();
        let start = explorer.path.clone();
        explorer.navigate(std::env::temp_dir());
        explorer.search = "old search".into();
        explorer.selected = Some(PathBuf::from("old.obj"));
        explorer.history_step(false);
        assert_eq!(explorer.path, start);
        assert!(explorer.search.is_empty());
        assert!(explorer.selected.is_none());
        explorer.navigate(start.join("nonexistent-explorer-test-folder"));
        assert_eq!(explorer.history.len(), 2);
        assert_eq!(explorer.history_index, 1);
        assert!(explorer.error.is_some());
        assert!(explorer.entries.is_empty());
    }
}

#[cfg(test)]
mod interaction_tests {
    use super::*;

    fn frame(
        context: &egui::Context,
        explorer: &mut AssetExplorer,
        events: Vec<egui::Event>,
    ) -> egui::Rect {
        let _ = context.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1200.0, 800.0),
                )),
                events,
                ..Default::default()
            },
            |ui| explorer.show(ui.ctx(), &mut true),
        );
        context
            .memory(|m| m.area_rect(egui::Id::new("asset_explorer_polished")))
            .unwrap()
    }

    #[test]
    fn native_resize_shrinks_and_expands_without_content_feedback() {
        let context = egui::Context::default();
        let mut explorer = AssetExplorer::default();
        let mut rect = frame(&context, &mut explorer, vec![]);
        for _ in 0..3 {
            rect = frame(&context, &mut explorer, vec![]);
        }
        for delta in [egui::vec2(-280.0, -130.0), egui::vec2(350.0, 200.0)] {
            let start = rect.right_bottom() - egui::vec2(3.0, 3.0);
            frame(
                &context,
                &mut explorer,
                vec![egui::Event::PointerMoved(start)],
            );
            frame(
                &context,
                &mut explorer,
                vec![egui::Event::PointerButton {
                    pos: start,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                }],
            );
            frame(
                &context,
                &mut explorer,
                vec![egui::Event::PointerMoved(start + delta)],
            );
            frame(
                &context,
                &mut explorer,
                vec![egui::Event::PointerButton {
                    pos: start + delta,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                }],
            );
            let resized = frame(&context, &mut explorer, vec![]);
            assert!(
                (resized.width() - rect.width() - delta.x).abs() < 20.0,
                "resize failed: {rect:?} -> {resized:?}"
            );
            for _ in 0..10 {
                let stable = frame(&context, &mut explorer, vec![]);
                assert!((stable.width() - resized.width()).abs() < 1.0);
                assert!((stable.height() - resized.height()).abs() < 1.0);
            }
            rect = resized;
        }
    }
}

#[cfg(test)]
mod drag_tests {
    use super::*;

    #[test]
    fn whole_tile_selects_and_delivers_file_payload_to_drop_target() {
        let context = egui::Context::default();
        let entry = Entry {
            path: PathBuf::from("example.obj"),
            name: "example.obj".into(),
            directory: false,
            size: Some(42),
        };
        let frame = |events: Vec<egui::Event>| {
            let mut result = (egui::Rect::NOTHING, egui::Rect::NOTHING, false, None);
            let _ = context.run_ui(
                egui::RawInput {
                    events,
                    ..Default::default()
                },
                |ui| {
                    let response = entry_widget(ui, &entry, false, true, egui::vec2(112.0, 104.0));
                    result.0 = response.rect;
                    result.2 = response.clicked();
                    ui.add_space(50.0);
                    let (rect, drop) =
                        ui.allocate_exact_size(egui::vec2(200.0, 50.0), Sense::hover());
                    result.1 = rect;
                    result.3 = drop.dnd_release_payload::<PathBuf>();
                },
            );
            result
        };
        let (tile, target, _, _) = frame(vec![]);
        let icon = tile.center_top() + egui::vec2(0.0, 25.0);
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        frame(vec![egui::Event::PointerMoved(icon)]);
        frame(vec![button(icon, true)]);
        assert!(
            frame(vec![button(icon, false)]).2,
            "clicking the icon must select the whole tile"
        );
        frame(vec![button(icon, true)]);
        frame(vec![egui::Event::PointerMoved(target.center())]);
        assert_eq!(
            *egui::DragAndDrop::payload::<PathBuf>(&context).unwrap(),
            entry.path
        );
        let released = frame(vec![button(target.center(), false)]).3;
        assert_eq!(*released.unwrap(), entry.path);
    }
}
