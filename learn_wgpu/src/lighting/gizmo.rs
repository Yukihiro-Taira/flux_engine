use cgmath::{InnerSpace, Vector4};

use super::{LightKind, LightingManager};

pub fn show(context: &egui::Context, manager: &mut LightingManager, camera: &crate::Camera) {
    if manager.mode != super::ViewportLightingMode::SceneLights {
        return;
    }
    let screen = context.content_rect();
    let view_projection = camera.build_view_projection_matrix();
    let forward = (camera.target - camera.eye).normalize();
    let right = forward.cross(camera.up).normalize();
    let up = right.cross(forward).normalize();
    let distance = (camera.eye - camera.target).magnitude().max(0.01);
    let units_per_pixel = match camera.projection_mode {
        crate::ProjectionMode::Perspective => {
            2.0 * distance * (camera.fovy.to_radians() * 0.5).tan() / screen.height().max(1.0)
        }
        crate::ProjectionMode::Orthographic => camera.ortho_scale / screen.height().max(1.0),
    };

    let mut changed = false;
    let mut selected_light = manager.selected_light;
    for light in &mut manager.lights {
        if !light.viewport_enabled {
            continue;
        }
        let clip = view_projection
            * Vector4::new(light.position[0], light.position[1], light.position[2], 1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 || !(0.0..=1.0).contains(&ndc.z) {
            continue;
        }
        let position = egui::pos2(
            screen.left() + (ndc.x + 1.0) * 0.5 * screen.width(),
            screen.top() + (1.0 - ndc.y) * 0.5 * screen.height(),
        );

        // Each light owns only its small marker-sized interaction area. A single
        // viewport-sized Area would sit above the rest of egui and capture input
        // everywhere as soon as the first light was created.
        egui::Area::new(egui::Id::new(("scene_light_gizmo", light.id.0)))
            .fixed_pos(position - egui::vec2(12.0, 12.0))
            .order(egui::Order::Middle)
            .movable(false)
            .show(context, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::click_and_drag());
                if response.clicked() {
                    selected_light = Some(light.id);
                }
                if response.dragged() {
                    let delta = ui.ctx().input(|input| input.pointer.delta());
                    let movement =
                        right * (delta.x * units_per_pixel) + up * (-delta.y * units_per_pixel);
                    light.position[0] += movement.x;
                    light.position[1] += movement.y;
                    light.position[2] += movement.z;
                    changed = true;
                }
                let color = if selected_light == Some(light.id) {
                    egui::Color32::YELLOW
                } else if light.enabled {
                    egui::Color32::from_rgb(255, 210, 90)
                } else {
                    egui::Color32::from_gray(110)
                };
                ui.painter()
                    .circle_stroke(rect.center(), 9.0, egui::Stroke::new(2.0, color));
                ui.painter().text(
                    rect.center() + egui::vec2(13.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    light.kind.name(),
                    egui::FontId::proportional(11.0),
                    color,
                );
                if matches!(light.kind, LightKind::Directional | LightKind::Spot) {
                    ui.painter().line_segment(
                        [rect.center(), rect.center() + egui::vec2(0.0, 18.0)],
                        egui::Stroke::new(1.5, color),
                    );
                }
            });
    }
    manager.selected_light = selected_light;
    if changed {
        manager.touch();
    }
}
