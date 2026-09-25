use cgmath::{EuclideanSpace, InnerSpace, Vector3, Vector4};

use crate::Camera;

pub struct GizmoChange {
    pub translation: Vector3<f32>,
    pub rotation_degrees: Vector3<f32>,
}

impl Default for GizmoChange {
    fn default() -> Self {
        Self {
            translation: Vector3::new(0.0, 0.0, 0.0),
            rotation_degrees: Vector3::new(0.0, 0.0, 0.0),
        }
    }
}

/// Draws a Blender-like transform gizmo and returns changes requested by this
/// frame's pointer drag.
pub fn show(
    context: &egui::Context,
    camera: &Camera,
    pivot: Vector3<f32>,
    navigation_active: bool,
    id_namespace: &'static str,
) -> Option<GizmoChange> {
    let screen = context.content_rect();
    let view_projection = camera.build_view_projection_matrix();
    let origin = project(view_projection, pivot, screen)?;
    let distance = (camera.eye.to_vec() - pivot).magnitude().max(0.01);
    let world_length = match camera.projection_mode {
        crate::ProjectionMode::Perspective => {
            2.0 * distance * (camera.fovy.to_radians() * 0.5).tan() * 82.0
                / screen.height().max(1.0)
        }
        crate::ProjectionMode::Orthographic => camera.ortho_scale * 82.0 / screen.height().max(1.0),
    };

    let axes = [
        (Vector3::unit_x(), translucent(238, 59, 59), "X"),
        (-Vector3::unit_y(), translucent(90, 190, 75), "Y"),
        (Vector3::unit_z(), translucent(70, 125, 245), "Z"),
    ];
    let mut change = GizmoChange::default();
    let mut changed = false;

    // Houdini-style world-space rotation rings. Each ring shares an axis with
    // its translation arrow and therefore tumbles with the scene, not the view.
    const RING_SEGMENTS: usize = 64;
    let ring_layer = egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new((id_namespace, "transform_gizmo_rotation_painter")),
    );
    let rotation_rings = [
        (
            Vector3::unit_y(),
            Vector3::unit_z(),
            translucent(238, 59, 59),
        ),
        (
            Vector3::unit_z(),
            Vector3::unit_x(),
            translucent(90, 190, 75),
        ),
        (
            Vector3::unit_x(),
            Vector3::unit_y(),
            translucent(70, 125, 245),
        ),
    ];
    let ring_radius = world_length * 0.68;
    for (axis_index, (basis_u, basis_v, color)) in rotation_rings.into_iter().enumerate() {
        let points = (0..RING_SEGMENTS)
            .map(|segment| {
                let angle = segment as f32 * std::f32::consts::TAU / RING_SEGMENTS as f32;
                let world = pivot + (basis_u * angle.cos() + basis_v * angle.sin()) * ring_radius;
                project(view_projection, world, screen)
            })
            .collect::<Option<Vec<_>>>();
        let Some(points) = points else { continue };
        paint_ring(
            &context.layer_painter(ring_layer),
            &points,
            egui::Stroke::new(1.8, color),
        );

        // Dense, generous hit regions follow each projected ellipse. Tangential
        // pointer motion is converted back into an angle around that world axis.
        for segment in (0..RING_SEGMENTS).step_by(4) {
            let previous = points[(segment + RING_SEGMENTS - 1) % RING_SEGMENTS];
            let next = points[(segment + 1) % RING_SEGMENTS];
            let tangent_pixels = next - previous;
            let tangent_length = tangent_pixels.length();
            if tangent_length < 0.5 {
                continue;
            }
            let tangent = tangent_pixels / tangent_length;
            let radians_per_pixel =
                (2.0 * std::f32::consts::TAU / RING_SEGMENTS as f32) / tangent_length;
            let handle_rect = egui::Rect::from_center_size(points[segment], egui::vec2(24.0, 24.0));
            egui::Area::new(egui::Id::new((
                id_namespace,
                "transform_gizmo_rotation",
                axis_index,
                segment,
            )))
            .order(egui::Order::Foreground)
            .fixed_pos(handle_rect.min)
            .movable(false)
            .show(context, |ui| {
                let (_, response) = ui.allocate_exact_size(
                    handle_rect.size(),
                    if navigation_active {
                        egui::Sense::hover()
                    } else {
                        egui::Sense::click_and_drag()
                    },
                );
                if response.hovered() || response.dragged() {
                    paint_ring(
                        &ui.ctx().layer_painter(ring_layer),
                        &points,
                        egui::Stroke::new(3.15, translucent(255, 220, 70)),
                    );
                }
                if response.dragged() {
                    let delta = ui.ctx().input(|input| input.pointer.delta());
                    change.rotation_degrees[axis_index] +=
                        delta.dot(tangent) * radians_per_pixel * 180.0 / std::f32::consts::PI;
                    changed = true;
                }
            });
        }
    }

    for (index, (axis, color, label)) in axes.into_iter().enumerate() {
        let endpoint = match project(view_projection, pivot + axis * world_length, screen) {
            Some(point) => point,
            None => continue,
        };
        let screen_axis = endpoint - origin;
        let pixels = screen_axis.length();
        if pixels < 5.0 {
            continue;
        }
        let direction = screen_axis / pixels;
        // Restrict translation interaction to the arrowhead. A rectangular hit
        // area around a diagonal shaft can overlap and steal input from rings.
        let interaction_rect = egui::Rect::from_center_size(endpoint, egui::vec2(28.0, 28.0));
        let axis_layer = egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new((id_namespace, "transform_gizmo_axis_painter", index)),
        );
        paint_axis(
            &context.layer_painter(axis_layer),
            origin,
            endpoint,
            direction,
            color,
            label,
            false,
        );

        egui::Area::new(egui::Id::new((id_namespace, "transform_gizmo_axis", index)))
            .order(egui::Order::Foreground)
            .fixed_pos(interaction_rect.min)
            .movable(false)
            .show(context, |ui| {
                let (_, response) = ui.allocate_exact_size(
                    interaction_rect.size(),
                    if navigation_active {
                        egui::Sense::hover()
                    } else {
                        egui::Sense::click_and_drag()
                    },
                );
                let active_color = if response.hovered() || response.dragged() {
                    translucent(255, 220, 70)
                } else {
                    color
                };
                if response.hovered() || response.dragged() {
                    paint_axis(
                        &ui.ctx().layer_painter(axis_layer),
                        origin,
                        endpoint,
                        direction,
                        active_color,
                        label,
                        true,
                    );
                }
                if response.dragged() {
                    let delta = ui.ctx().input(|input| input.pointer.delta());
                    let pixels_along_axis = delta.dot(direction);
                    change.translation += axis * (pixels_along_axis * world_length / pixels);
                    changed = true;
                }
            });
    }

    let view_forward = (camera.target - camera.eye).normalize();
    let view_right = view_forward.cross(camera.up).normalize();
    let view_up = view_right.cross(view_forward).normalize();
    let units_per_pixel = world_length / 82.0;
    egui::Area::new(egui::Id::new((id_namespace, "transform_gizmo_center")))
        .order(egui::Order::Foreground)
        .fixed_pos(origin - egui::vec2(11.0, 11.0))
        .movable(false)
        .show(context, |ui| {
            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(22.0, 22.0),
                if navigation_active {
                    egui::Sense::hover()
                } else {
                    egui::Sense::click_and_drag()
                },
            );
            let color = if response.hovered() || response.dragged() {
                translucent(255, 220, 70)
            } else {
                translucent(255, 255, 255)
            };
            ui.painter().circle_filled(rect.center(), 7.0, color);
            ui.painter().circle_stroke(
                rect.center(),
                7.0,
                egui::Stroke::new(1.35, translucent(35, 35, 35)),
            );
            if response.dragged() {
                let delta = ui.ctx().input(|input| input.pointer.delta());
                change.translation += view_right * (delta.x * units_per_pixel)
                    + view_up * (-delta.y * units_per_pixel);
                changed = true;
            }
        });

    changed.then_some(change)
}

fn paint_ring(painter: &egui::Painter, points: &[egui::Pos2], stroke: egui::Stroke) {
    for index in 0..points.len() {
        painter.line_segment([points[index], points[(index + 1) % points.len()]], stroke);
    }
}

fn paint_axis(
    painter: &egui::Painter,
    origin: egui::Pos2,
    endpoint: egui::Pos2,
    direction: egui::Vec2,
    color: egui::Color32,
    label: &str,
    highlighted: bool,
) {
    painter.line_segment(
        [origin, endpoint - direction * 8.0],
        egui::Stroke::new(if highlighted { 3.6 } else { 2.7 }, color),
    );
    let perpendicular = egui::vec2(-direction.y, direction.x);
    painter.add(egui::Shape::convex_polygon(
        vec![
            endpoint,
            endpoint - direction * 13.0 + perpendicular * 6.0,
            endpoint - direction * 13.0 - perpendicular * 6.0,
        ],
        color,
        egui::Stroke::NONE,
    ));
    painter.text(
        endpoint + direction * 10.0,
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(12.0),
        color,
    );
}

fn translucent(red: u8, green: u8, blue: u8) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(red, green, blue, 217)
}

fn project(
    view_projection: cgmath::Matrix4<f32>,
    world: Vector3<f32>,
    screen: egui::Rect,
) -> Option<egui::Pos2> {
    let clip = view_projection * Vector4::new(world.x, world.y, world.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !(0.0..=1.0).contains(&ndc.z) {
        return None;
    }
    Some(egui::pos2(
        screen.left() + (ndc.x + 1.0) * 0.5 * screen.width(),
        screen.top() + (1.0 - ndc.y) * 0.5 * screen.height(),
    ))
}
