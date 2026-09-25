use super::*;

pub(crate) struct Timeline {
    clip: Option<usd_import::AnimationClip>,
    frame: f64,
    displayed: Option<f64>,
    playing: bool,
    reverse: bool,
    range: Option<(f64, f64)>,
    looping: bool,
    speed: f64,
    last_step: Option<Instant>,
    pending: Option<f64>,
    message: String,
    transforms: HashMap<String, [[f32; 4]; 4]>,
    bindings: HashMap<
        String,
        (
            material_library::MaterialId,
            Vec<material_library::FaceMaterialAssignment>,
            bool,
            usize,
        ),
    >,
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<Worker>,
}
impl Default for Timeline {
    fn default() -> Self {
        Self {
            clip: None,
            frame: 0.0,
            displayed: None,
            playing: false,
            reverse: false,
            range: None,
            looping: true,
            speed: 1.0,
            last_step: None,
            pending: None,
            message: String::new(),
            transforms: HashMap::new(),
            bindings: HashMap::new(),
            #[cfg(not(target_arch = "wasm32"))]
            worker: None,
        }
    }
}
#[cfg(not(target_arch = "wasm32"))]
struct Worker {
    requests: std::sync::mpsc::Sender<f64>,
    results: std::sync::mpsc::Receiver<(f64, Result<AnimationFrame, String>)>,
}
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
enum AnimationFrame {
    Geometry(
        resources::PreparedModel,
        Option<HashMap<String, [[f32; 4]; 4]>>,
    ),
    Transforms(HashMap<String, [[f32; 4]; 4]>),
}
#[cfg(not(target_arch = "wasm32"))]
impl AnimationFrame {
    fn bytes(&self) -> usize {
        match self {
            Self::Geometry(model, _) => model.animation_bytes(),
            Self::Transforms(transforms) => {
                transforms.iter().map(|(path, _)| path.len() + 64).sum()
            }
        }
    }
}
fn valid_clip(clip: &usd_import::AnimationClip) -> bool {
    clip.start.is_finite()
        && clip.end.is_finite()
        && clip.end >= clip.start
        && clip.rate.is_finite()
        && clip.rate > 0.0
}
fn advance_playback(
    frame: f64,
    clip: &usd_import::AnimationClip,
    looping: bool,
    reverse: bool,
    steps: f64,
) -> Option<f64> {
    let next = frame + if reverse { -steps } else { steps };
    if next >= clip.start && next <= clip.end {
        Some(next)
    } else if looping {
        Some(clip.start + (next - clip.start).rem_euclid(clip.end - clip.start + 1.0))
    } else if reverse && frame > clip.start {
        Some(clip.start)
    } else if !reverse && frame < clip.end {
        Some(clip.end)
    } else {
        None
    }
}

impl Timeline {
    pub(crate) fn has_transforms(&self) -> bool {
        !self.transforms.is_empty()
    }
    pub(crate) fn transform_for(&self, name: &str) -> Option<cgmath::Matrix4<f32>> {
        self.transforms
            .get(name.split(" · ").next().unwrap_or(name))
            .copied()
            .map(Into::into)
    }

    pub fn is_running(&self) -> bool {
        self.playing || self.pending.is_some()
    }
    pub fn pause_for_edit(&mut self) {
        self.playing = false;
        if self.pending.is_some() {
            if let Some(frame) = self.displayed {
                self.frame = frame;
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn attach(
        &mut self,
        path: std::path::PathBuf,
        clip: Option<usd_import::AnimationClip>,
        frame: Option<f64>,
    ) {
        *self = Self::default();
        self.looping = true;
        self.speed = 1.0;
        self.clip = clip.filter(valid_clip);
        let Some(clip) = &self.clip else {
            return;
        };
        self.frame = frame.unwrap_or(clip.start).clamp(clip.start, clip.end);
        self.displayed = Some(frame.unwrap_or(clip.start));
        if !clip.animated || clip.end <= clip.start {
            return;
        }
        let (requests, input) = std::sync::mpsc::channel::<f64>();
        let (output, results) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut usd = None;
            let mut cache = std::collections::VecDeque::<(f64, AnimationFrame)>::new();
            let mut bytes = 0usize;
            const LIMIT: usize = 128 * 1024 * 1024;
            while let Ok(mut frame) = input.recv() {
                while let Ok(latest) = input.try_recv() {
                    frame = latest;
                }
                let result = if let Some(index) = cache.iter().position(|(time, _)| *time == frame)
                {
                    let entry = cache.remove(index).unwrap();
                    let model = entry.1.clone();
                    cache.push_back(entry);
                    Ok(model)
                } else {
                    let result = (|| -> anyhow::Result<AnimationFrame> {
                        if usd_import::is_usd(&path) {
                            if usd.is_none() {
                                usd = Some(usd_import::AnimationSession::new(&path)?);
                            }
                            match usd.as_mut().unwrap().sample(frame)? {
                                usd_import::AnimationSample::Transforms(transforms) => {
                                    Ok(AnimationFrame::Transforms(transforms))
                                }
                                usd_import::AnimationSample::Geometry(scene, transforms) => {
                                    resources::prepare_usd_scene(scene, &path)
                                        .map(|model| AnimationFrame::Geometry(model, transforms))
                                }
                            }
                        } else {
                            resources::prepare_model_from_path(&path, Some(frame))
                                .map(|model| AnimationFrame::Geometry(model, None))
                        }
                    })()
                    .map_err(|e| format!("{e:#}"));
                    if let Ok(model) = &result {
                        let size = model.bytes();
                        if size <= LIMIT {
                            while bytes + size > LIMIT || cache.len() >= 240 {
                                if let Some((_, old)) = cache.pop_front() {
                                    bytes -= old.bytes();
                                } else {
                                    break;
                                }
                            }
                            bytes += size;
                            cache.push_back((frame, model.clone()));
                        }
                    }
                    result
                };
                if output.send((frame, result)).is_err() {
                    break;
                }
            }
        });
        self.worker = Some(Worker { requests, results });
    }
}
impl State {
    pub(crate) fn timeline_ui(&mut self, context: &egui::Context) {
        let screen = context.content_rect();
        let id = egui::Id::new("fixed_animation_timeline");
        let clip = self.timeline.clip.clone();
        let enabled = clip.as_ref().is_some_and(|c| c.animated && c.end > c.start);
        let (start, end) = clip
            .as_ref()
            .map(|c| (c.start, c.end))
            .unwrap_or((1.0, 240.0));
        let (mut low, mut high) = self.timeline.range.unwrap_or((start, end));
        egui::Area::new(id)
            .order(egui::Order::Foreground)
            .fixed_pos(egui::pos2(screen.left(), screen.bottom() - HEIGHT))
            .movable(false)
            .show(context, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_gray(55))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(85)))
                    .inner_margin(egui::Margin::symmetric(7, 4))
                    .show(ui, |ui| {
                        ui.set_width((screen.width() - 16.0).max(100.0));
                        ui.set_min_height(HEIGHT - 10.0);
                        ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                        ui.visuals_mut().widgets.inactive.bg_fill = egui::Color32::from_gray(65);
                        ui.visuals_mut().widgets.inactive.corner_radius = egui::CornerRadius::ZERO;
                        ui.visuals_mut().widgets.hovered.corner_radius = egui::CornerRadius::ZERO;
                        ui.visuals_mut().widgets.active.corner_radius = egui::CornerRadius::ZERO;
                        ui.horizontal(|ui| {
                            ui.add_enabled_ui(enabled, |ui| {
                                if transport(ui, 0, false, "First frame").clicked() {
                                    self.timeline.playing = false;
                                    self.timeline.frame = low;
                                }
                                if transport(
                                    ui,
                                    1,
                                    self.timeline.playing && self.timeline.reverse,
                                    "Play backward",
                                )
                                .clicked()
                                {
                                    self.timeline.playing = true;
                                    self.timeline.reverse = true;
                                    self.timeline.last_step = Some(Instant::now());
                                }
                                if transport(ui, 2, !self.timeline.playing, "Stop playback")
                                    .clicked()
                                {
                                    self.timeline.playing = false;
                                }
                                if transport(
                                    ui,
                                    3,
                                    self.timeline.playing && !self.timeline.reverse,
                                    "Play forward / pause",
                                )
                                .clicked()
                                {
                                    self.timeline.playing =
                                        !(self.timeline.playing && !self.timeline.reverse);
                                    self.timeline.reverse = false;
                                    self.timeline.last_step = Some(Instant::now());
                                }
                                if transport(ui, 4, false, "Last frame").clicked() {
                                    self.timeline.playing = false;
                                    self.timeline.frame = high;
                                }
                                ui.add_space(5.0);
                                let response = ui.add(
                                    egui::DragValue::new(&mut self.timeline.frame)
                                        .range(start..=end)
                                        .speed(1.0)
                                        .fixed_decimals(0),
                                );
                                if response.changed() {
                                    self.timeline.playing = false;
                                }
                                response.on_hover_text("Current frame — type a frame number");
                                ui.checkbox(&mut self.timeline.looping, "Loop");
                                ui.add(
                                    egui::DragValue::new(&mut self.timeline.speed)
                                        .range(0.1..=4.0)
                                        .speed(0.1)
                                        .suffix("×"),
                                );
                            });
                            ui.separator();
                            ui.label(
                                egui::RichText::new(
                                    clip.as_ref()
                                        .map(|c| format!("{:.0} FPS", c.rate))
                                        .unwrap_or("24 FPS".into()),
                                )
                                .monospace(),
                            );
                            if self.timeline.pending.is_some() {
                                ui.spinner();
                                ui.label("Evaluating");
                            } else {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(if enabled {
                                            clip.as_ref().unwrap().name.as_str()
                                        } else {
                                            "No animated asset"
                                        })
                                        .small(),
                                    )
                                    .truncate(),
                                );
                            }
                            if !self.timeline.message.is_empty() {
                                ui.label("⚠").on_hover_text(&self.timeline.message);
                            }
                        });
                        let (ruler, response) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), 32.0),
                            egui::Sense::click_and_drag(),
                        );
                        let paint = ui.painter_at(ruler);
                        paint.rect_filled(ruler, 0.0, egui::Color32::from_gray(37));
                        let x = |frame: f64| {
                            ruler.left()
                                + ((frame - start) / (end - start).max(1.0)) as f32 * ruler.width()
                        };
                        let step = (((end - start) / ((ruler.width() / 65.0).max(1.0) as f64))
                            .ceil())
                        .max(1.0);
                        let first = (start / step).ceil() * step;
                        for tick in 0..=((end - first) / step).max(0.0) as usize {
                            let frame = first + tick as f64 * step;
                            let px = x(frame);
                            paint.line_segment(
                                [
                                    egui::pos2(px, ruler.bottom() - 9.0),
                                    egui::pos2(px, ruler.bottom()),
                                ],
                                egui::Stroke::new(1.0, egui::Color32::from_gray(150)),
                            );
                            paint.text(
                                egui::pos2(px + 3.0, ruler.top() + 3.0),
                                egui::Align2::LEFT_TOP,
                                format!("{frame:.0}"),
                                egui::FontId::monospace(10.0),
                                egui::Color32::from_gray(195),
                            );
                            for minor in 1..5 {
                                let px = x(frame + step * minor as f64 / 5.0);
                                if px < ruler.right() {
                                    paint.line_segment(
                                        [
                                            egui::pos2(px, ruler.bottom() - 4.0),
                                            egui::pos2(px, ruler.bottom()),
                                        ],
                                        egui::Stroke::new(1.0, egui::Color32::from_gray(100)),
                                    );
                                }
                            }
                        }
                        let current = x(self.timeline.frame.clamp(start, end));
                        paint.line_segment(
                            [
                                egui::pos2(current, ruler.top()),
                                egui::pos2(current, ruler.bottom()),
                            ],
                            egui::Stroke::new(1.0, ui.visuals().selection.bg_fill),
                        );
                        let badge = egui::Rect::from_center_size(
                            egui::pos2(
                                current.clamp(ruler.left() + 18.0, ruler.right() - 18.0),
                                ruler.top() + 9.0,
                            ),
                            egui::vec2(36.0, 18.0),
                        );
                        paint.rect_filled(badge, 0.0, egui::Color32::BLACK);
                        paint.text(
                            badge.center(),
                            egui::Align2::CENTER_CENTER,
                            format!("{:.0}", self.timeline.frame),
                            egui::FontId::monospace(11.0),
                            egui::Color32::WHITE,
                        );
                        paint.add(egui::Shape::convex_polygon(
                            vec![
                                egui::pos2(current - 4.0, badge.bottom()),
                                egui::pos2(current + 4.0, badge.bottom()),
                                egui::pos2(current, badge.bottom() + 5.0),
                            ],
                            egui::Color32::BLACK,
                            egui::Stroke::NONE,
                        ));
                        if let Some(pos) = response.hover_pos() {
                            paint.line_segment(
                                [
                                    egui::pos2(pos.x, ruler.top()),
                                    egui::pos2(pos.x, ruler.bottom()),
                                ],
                                egui::Stroke::new(1.0, egui::Color32::from_gray(190)),
                            );
                        }
                        if enabled && (response.clicked() || response.dragged()) {
                            if let Some(pos) = response.interact_pointer_pos() {
                                self.timeline.frame = (start
                                    + ((pos.x - ruler.left()) / ruler.width()).clamp(0.0, 1.0)
                                        as f64
                                        * (end - start))
                                    .round()
                                    .clamp(start, end);
                                self.timeline.playing = false;
                            }
                        }
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("{start:.0}"))
                                    .monospace()
                                    .color(egui::Color32::LIGHT_GRAY),
                            );
                            ui.add_enabled(
                                enabled,
                                egui::DragValue::new(&mut low)
                                    .range(start..=high)
                                    .speed(1.0)
                                    .fixed_decimals(0),
                            )
                            .on_hover_text("Playback start");
                            let width = (ui.available_width() - 108.0).max(20.0);
                            let (track, response) = ui.allocate_exact_size(
                                egui::vec2(width, 15.0),
                                egui::Sense::click_and_drag(),
                            );
                            ui.painter()
                                .rect_filled(track, 0.0, egui::Color32::from_gray(28));
                            let left = track.left()
                                + ((low - start) / (end - start).max(1.0)) as f32 * track.width();
                            let right = track.left()
                                + ((high - start) / (end - start).max(1.0)) as f32 * track.width();
                            let band = egui::Rect::from_min_max(
                                egui::pos2(left, track.top() + 2.0),
                                egui::pos2(right, track.bottom() - 2.0),
                            );
                            ui.painter()
                                .rect_filled(band, 1.0, ui.visuals().selection.bg_fill);
                            for px in [left, right] {
                                ui.painter().line_segment(
                                    [
                                        egui::pos2(px, track.top() + 1.0),
                                        egui::pos2(px, track.bottom() - 1.0),
                                    ],
                                    egui::Stroke::new(3.0, egui::Color32::from_gray(175)),
                                );
                            }
                            if enabled && (response.clicked() || response.dragged()) {
                                if let Some(pos) = response.interact_pointer_pos() {
                                    let value = (start
                                        + ((pos.x - track.left()) / track.width()).clamp(0.0, 1.0)
                                            as f64
                                            * (end - start))
                                        .round();
                                    if (value - low).abs() < (value - high).abs() {
                                        low = value.min(high);
                                    } else {
                                        high = value.max(low);
                                    }
                                }
                            }
                            ui.add_enabled(
                                enabled,
                                egui::DragValue::new(&mut high)
                                    .range(low..=end)
                                    .speed(1.0)
                                    .fixed_decimals(0),
                            )
                            .on_hover_text("Playback end");
                            ui.label(
                                egui::RichText::new(format!("{end:.0}"))
                                    .monospace()
                                    .color(egui::Color32::LIGHT_GRAY),
                            );
                        });
                    });
            });
        self.timeline.range = Some((low, high));
        context.move_to_top(egui::LayerId::new(egui::Order::Foreground, id));
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn update_timeline(&mut self) {
        if self.pending_model_load.is_some() || self.editor.pending_asset.is_some() {
            return;
        }
        let response = match self
            .timeline
            .worker
            .as_ref()
            .map(|worker| worker.results.try_recv())
        {
            Some(Ok(response)) => Some(response),
            Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                self.timeline.worker = None;
                self.timeline.pending = None;
                self.timeline.playing = false;
                self.timeline.message = "Animation worker stopped".into();
                None
            }
            _ => None,
        };
        if let Some((frame, result)) = response {
            self.timeline.pending = None;
            if frame == self.timeline.frame {
                match result.and_then(|sample| match sample {
                    AnimationFrame::Geometry(prepared, transforms) => {
                        self.timeline.transforms = transforms.unwrap_or_default();
                        resources::upload_prepared_model(prepared, "Animation frame", &self.device)
                            .map(Some)
                            .map_err(|e| e.to_string())
                    }
                    AnimationFrame::Transforms(transforms) => {
                        if self.obj_model.meshes.iter().any(|mesh| {
                            !transforms
                                .contains_key(mesh.name.split(" · ").next().unwrap_or(&mesh.name))
                        }) {
                            return Err("Animated transform does not match imported mesh".into());
                        }
                        self.timeline.transforms = transforms;
                        self.demo_cached = None;
                        Ok(None)
                    }
                }) {
                    Ok(None) => {
                        self.timeline.displayed = Some(frame);
                        self.timeline.message.clear();
                        self.editor.usd_use_stage_start = false;
                        self.editor.usd_time_code = frame;
                    }
                    Ok(Some(model)) => {
                        // Match by stable mesh name, preserving editor materials and visibility.
                        for (i, mesh) in self.obj_model.meshes.iter().enumerate() {
                            self.timeline.bindings.insert(
                                mesh.name.clone(),
                                (
                                    self.mesh_material_assignments
                                        .get(i)
                                        .copied()
                                        .unwrap_or(material_library::MaterialId(0)),
                                    self.face_material_assignments
                                        .get(i)
                                        .cloned()
                                        .unwrap_or_default(),
                                    self.geo_group_enabled.get(i).copied().unwrap_or(true),
                                    mesh.source_indices.len(),
                                ),
                            );
                        }
                        let assignments = model
                            .meshes
                            .iter()
                            .map(|mesh| {
                                self.timeline
                                    .bindings
                                    .get(&mesh.name)
                                    .map(|binding| binding.0)
                                    .unwrap_or(material_library::MaterialId(0))
                            })
                            .collect();
                        let faces = model
                            .meshes
                            .iter()
                            .map(|mesh| {
                                self.timeline
                                    .bindings
                                    .get(&mesh.name)
                                    .filter(|binding| binding.3 == mesh.source_indices.len())
                                    .map(|binding| binding.1.clone())
                                    .unwrap_or_default()
                            })
                            .collect();
                        let enabled = model
                            .meshes
                            .iter()
                            .map(|mesh| {
                                self.timeline
                                    .bindings
                                    .get(&mesh.name)
                                    .map(|binding| binding.2)
                                    .unwrap_or(true)
                            })
                            .collect();
                        self.obj_model = model;
                        self.mesh_material_assignments = assignments;
                        self.face_material_assignments = faces;
                        self.geo_group_enabled = enabled;
                        self.geo_texture_enabled
                            .resize(self.obj_model.meshes.len(), true);
                        self.part_selection = Default::default();
                        self.demo_cached = None;
                        self.demo_buffers.clear();
                        self.uv_space_cache.clear();
                        self.timeline.displayed = Some(frame);
                        self.timeline.message.clear();
                        self.editor.usd_use_stage_start = false;
                        self.editor.usd_time_code = frame;
                    }
                    Err(error) => {
                        self.timeline.message = error;
                        self.timeline.playing = false;
                        self.timeline.frame = self.timeline.displayed.unwrap_or(frame);
                    }
                }
            }
        }
        if self.timeline.pending.is_none()
            && self.timeline.displayed == Some(self.timeline.frame)
            && self.timeline.playing
        {
            if let Some(clip) = &self.timeline.clip {
                let duration = 1.0 / (clip.rate * self.timeline.speed.max(0.1));
                if self
                    .timeline
                    .last_step
                    .is_none_or(|last| last.elapsed().as_secs_f64() >= duration)
                {
                    let mut playback = clip.clone();
                    if let Some((low, high)) = self.timeline.range {
                        playback.start = low;
                        playback.end = high;
                    }
                    let elapsed = self
                        .timeline
                        .last_step
                        .map_or(duration, |last| last.elapsed().as_secs_f64());
                    let steps = (elapsed / duration).floor().max(1.0);
                    let next = advance_playback(
                        self.timeline.frame,
                        &playback,
                        self.timeline.looping,
                        self.timeline.reverse,
                        steps,
                    );
                    self.timeline.last_step = Some(
                        self.timeline.last_step.unwrap_or_else(Instant::now)
                            + std::time::Duration::from_secs_f64(steps * duration),
                    );
                    if let Some(frame) = next {
                        self.timeline.frame = frame;
                    } else {
                        self.timeline.playing = false;
                    }
                }
            }
        }
        if self.timeline.pending.is_none() && self.timeline.displayed != Some(self.timeline.frame) {
            if let Some(worker) = &self.timeline.worker {
                if worker.requests.send(self.timeline.frame).is_ok() {
                    self.timeline.pending = Some(self.timeline.frame);
                    if !self.timeline.playing {
                        self.timeline.last_step = Some(Instant::now());
                    }
                } else {
                    self.timeline.playing = false;
                    self.timeline.message = "Animation worker stopped".into();
                }
            }
        }
        if self.timeline.playing || self.timeline.pending.is_some() {
            self.window.request_redraw();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn playback_catches_up_at_source_rate_without_slowing_animation() {
        let clip = usd_import::AnimationClip {
            start: 0.,
            end: 6000.,
            rate: 120.,
            animated: true,
            name: "clip".into(),
        };
        assert_eq!(advance_playback(0., &clip, false, false, 120.), Some(120.));
        assert_eq!(advance_playback(120., &clip, false, true, 120.), Some(0.));
        assert_eq!(
            advance_playback(5990., &clip, false, false, 120.),
            Some(6000.)
        );
        assert_eq!(advance_playback(6000., &clip, true, false, 1.), Some(0.));
    }

    #[test]
    fn playback_preserves_nonzero_source_range_and_stops_or_loops() {
        let clip = usd_import::AnimationClip {
            start: 101.,
            end: 103.,
            rate: 24.,
            animated: true,
            name: "clip".into(),
        };
        assert!(valid_clip(&clip));
        assert_eq!(advance_playback(101., &clip, false, false, 1.), Some(102.));
        assert_eq!(advance_playback(103., &clip, false, false, 1.), None);
        assert_eq!(advance_playback(103., &clip, true, false, 1.), Some(101.));
        assert!(!valid_clip(&usd_import::AnimationClip { rate: 0., ..clip }));
    }
}

pub(crate) const HEIGHT: f32 = 96.0;
pub(crate) fn workspace_rect(context: &egui::Context) -> egui::Rect {
    let mut rect = context.content_rect();
    rect.max.y = (rect.max.y - HEIGHT).max(rect.min.y);
    rect
}
fn transport(ui: &mut egui::Ui, kind: u8, active: bool, hint: &str) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(25.0, 22.0), egui::Sense::click());
    let bg = if active {
        egui::Color32::from_gray(38)
    } else if response.hovered() {
        egui::Color32::from_gray(89)
    } else {
        egui::Color32::from_gray(65)
    };
    ui.painter().rect_filled(rect, 0.0, bg);
    ui.painter().rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(26)),
        egui::StrokeKind::Inside,
    );
    let color = if !ui.is_enabled() {
        egui::Color32::from_gray(95)
    } else if active && kind != 2 {
        ui.visuals().selection.bg_fill
    } else {
        egui::Color32::from_gray(220)
    };
    let center = rect.center();
    if kind == 2 {
        ui.painter().rect_filled(
            egui::Rect::from_center_size(center, egui::vec2(9.0, 9.0)),
            0.0,
            color,
        );
    } else {
        let dir = if kind <= 1 { -1.0 } else { 1.0 };
        ui.painter().add(egui::Shape::convex_polygon(
            vec![
                center + egui::vec2(dir * 5.0, 0.0),
                center + egui::vec2(-dir * 4.0, -5.0),
                center + egui::vec2(-dir * 4.0, 5.0),
            ],
            color,
            egui::Stroke::NONE,
        ));
        if kind == 0 || kind == 4 {
            let px = center.x + dir * 8.0;
            ui.painter().line_segment(
                [
                    egui::pos2(px, center.y - 5.0),
                    egui::pos2(px, center.y + 5.0),
                ],
                egui::Stroke::new(2.0, color),
            );
        }
    }
    response.on_hover_text(hint)
}
