use cgmath::{Deg, Matrix3, Vector3, Vector4};

use super::{AreaShape, LightKind, LightingManager, SceneLight};

pub fn show(
    context: &egui::Context,
    manager: &mut LightingManager,
    camera: &crate::Camera,
    navigation_active: bool,
) {
    if manager.mode != super::ViewportLightingMode::SceneLights {
        return;
    }
    let screen = context.content_rect();
    let view_projection = camera.build_view_projection_matrix();
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
        let color = if selected_light == Some(light.id) {
            egui::Color32::YELLOW
        } else if light.enabled {
            egui::Color32::from_rgb(255, 210, 90)
        } else {
            egui::Color32::from_gray(110)
        };
        let outline_layer = egui::LayerId::new(
            egui::Order::Middle,
            egui::Id::new(("scene_light_world_outline", light.id.0)),
        );
        draw_world_light_outline(
            &context.layer_painter(outline_layer),
            light,
            view_projection,
            screen,
            color,
        );

        // Each light owns only its small marker-sized interaction area. A single
        // viewport-sized Area would sit above the rest of egui and capture input
        // everywhere as soon as the first light was created.
        egui::Area::new(egui::Id::new(("scene_light_gizmo", light.id.0)))
            .fixed_pos(position - egui::vec2(26.0, 26.0))
            .order(egui::Order::Middle)
            .movable(false)
            .show(context, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(egui::vec2(52.0, 52.0), egui::Sense::click());
                if response.clicked() {
                    selected_light = Some(light.id);
                }
                ui.painter()
                    .circle_stroke(rect.center(), 4.0, egui::Stroke::new(1.1, color));
                ui.painter().text(
                    rect.center() + egui::vec2(17.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    light.kind.name(),
                    egui::FontId::proportional(11.0),
                    color,
                );
            });
    }
    manager.selected_light = selected_light;
    let mut changed = false;
    if let Some(light) = manager.selected_mut()
        && light.viewport_enabled
    {
        let pivot = Vector3::new(light.position[0], light.position[1], light.position[2]);
        if let Some(change) =
            crate::object_gizmo::show(context, camera, pivot, navigation_active, "selected_light")
        {
            light.position[0] += change.translation.x;
            light.position[1] += change.translation.y;
            light.position[2] += change.translation.z;
            for axis in 0..3 {
                light.rotation_degrees[axis] =
                    (light.rotation_degrees[axis] + change.rotation_degrees[axis] + 180.0)
                        .rem_euclid(360.0)
                        - 180.0;
            }
            changed = true;
        }
    }
    if changed {
        manager.touch();
    }
}

fn draw_world_light_outline(
    painter: &egui::Painter,
    light: &SceneLight,
    view_projection: cgmath::Matrix4<f32>,
    screen: egui::Rect,
    color: egui::Color32,
) {
    let stroke = egui::Stroke::new(1.1, color);
    let position = Vector3::from(light.position);
    let rotation = Matrix3::from_angle_z(Deg(light.rotation_degrees[2]))
        * Matrix3::from_angle_y(Deg(light.rotation_degrees[1]))
        * Matrix3::from_angle_x(Deg(light.rotation_degrees[0]));
    let world = |local: Vector3<f32>| position + rotation * local;
    let line = |a: Vector3<f32>, b: Vector3<f32>| {
        if let (Some(a), Some(b)) = (
            project_world(view_projection, world(a), screen),
            project_world(view_projection, world(b), screen),
        ) {
            painter.line_segment([a, b], stroke);
        }
    };
    let closed = |points: &[Vector3<f32>]| {
        for index in 0..points.len() {
            line(points[index], points[(index + 1) % points.len()]);
        }
    };
    let circle = |center: Vector3<f32>, u: Vector3<f32>, v: Vector3<f32>, radius: f32| {
        const SEGMENTS: usize = 40;
        for index in 0..SEGMENTS {
            let first = index as f32 * std::f32::consts::TAU / SEGMENTS as f32;
            let second = (index + 1) as f32 * std::f32::consts::TAU / SEGMENTS as f32;
            line(
                center + (u * first.cos() + v * first.sin()) * radius,
                center + (u * second.cos() + v * second.sin()) * radius,
            );
        }
    };
    let half_width = light.size[0].abs().max(0.001) * 0.5;
    let half_height = light.size[1].abs().max(0.001) * 0.5;
    let radius = light.radius.abs().max(0.001);

    match &light.kind {
        LightKind::Point => {
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_x(),
                Vector3::unit_y(),
                radius,
            );
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_x(),
                Vector3::unit_z(),
                radius,
            );
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_y(),
                Vector3::unit_z(),
                radius,
            );
        }
        LightKind::Spot => {
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_x(),
                Vector3::unit_y(),
                radius,
            );
            let length = light.size[1].abs().max(radius * 4.0).max(1.0);
            let cone_radius = length * (light.outer_angle_degrees.to_radians() * 0.5).tan();
            let end = Vector3::new(0.0, 0.0, -length);
            circle(end, Vector3::unit_x(), Vector3::unit_y(), cone_radius);
            for corner in [
                Vector3::new(cone_radius, 0.0, -length),
                Vector3::new(-cone_radius, 0.0, -length),
                Vector3::new(0.0, cone_radius, -length),
                Vector3::new(0.0, -cone_radius, -length),
            ] {
                line(Vector3::new(0.0, 0.0, 0.0), corner);
            }
        }
        LightKind::Directional => {
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_x(),
                Vector3::unit_y(),
                radius,
            );
            line(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, -half_height * 2.0),
            );
        }
        LightKind::Area(AreaShape::Rectangle) => closed(&[
            Vector3::new(-half_width, -half_height, 0.0),
            Vector3::new(half_width, -half_height, 0.0),
            Vector3::new(half_width, half_height, 0.0),
            Vector3::new(-half_width, half_height, 0.0),
        ]),
        LightKind::Area(AreaShape::Disk) => circle(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::unit_x(),
            Vector3::unit_y(),
            half_width,
        ),
        LightKind::Area(AreaShape::Line) => line(
            Vector3::new(-half_width, 0.0, 0.0),
            Vector3::new(half_width, 0.0, 0.0),
        ),
        LightKind::Area(AreaShape::Tube) => {
            for y in [-radius, radius] {
                line(
                    Vector3::new(-half_width, y, 0.0),
                    Vector3::new(half_width, y, 0.0),
                );
            }
            for x in [-half_width, half_width] {
                circle(
                    Vector3::new(x, 0.0, 0.0),
                    Vector3::unit_y(),
                    Vector3::unit_z(),
                    radius,
                );
            }
        }
        LightKind::Area(AreaShape::Sphere) => {
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_x(),
                Vector3::unit_y(),
                radius,
            );
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_x(),
                Vector3::unit_z(),
                radius,
            );
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_y(),
                Vector3::unit_z(),
                radius,
            );
        }
        LightKind::Environment => {
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_x(),
                Vector3::unit_y(),
                radius,
            );
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_x(),
                Vector3::unit_z(),
                radius,
            );
        }
        LightKind::PhysicalSky => {
            circle(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::unit_x(),
                Vector3::unit_y(),
                radius,
            );
        }
        LightKind::Geometry => closed(&[
            Vector3::new(-half_width, -half_height, 0.0),
            Vector3::new(half_width, -half_height, 0.0),
            Vector3::new(half_width, half_height, 0.0),
            Vector3::new(-half_width, half_height, 0.0),
        ]),
        LightKind::Portal => closed(&[
            Vector3::new(-half_width, -half_height, 0.0),
            Vector3::new(half_width, -half_height, 0.0),
            Vector3::new(half_width, half_height, 0.0),
            Vector3::new(-half_width, half_height, 0.0),
        ]),
    }
}

fn project_world(
    view_projection: cgmath::Matrix4<f32>,
    world: Vector3<f32>,
    screen: egui::Rect,
) -> Option<egui::Pos2> {
    let clip = view_projection * Vector4::new(world.x, world.y, world.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(egui::pos2(
        screen.left() + (ndc.x + 1.0) * 0.5 * screen.width(),
        screen.top() + (1.0 - ndc.y) * 0.5 * screen.height(),
    ))
}
