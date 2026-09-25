use cgmath::InnerSpace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewDirection {
    Right,
    Left,
    Back,
    Front,
    Top,
    Bottom,
}

#[derive(Clone, Copy)]
struct AxisEndpoint {
    position: egui::Pos2,
    color: egui::Color32,
    radius: f32,
    label: Option<&'static str>,
    id: &'static str,
    view: ViewDirection,
}

fn project_axis(
    axis: cgmath::Vector3<f32>,
    camera_right: cgmath::Vector3<f32>,
    camera_up: cgmath::Vector3<f32>,
) -> egui::Vec2 {
    egui::vec2(axis.dot(camera_right), -axis.dot(camera_up))
}

fn draw_endpoint(ui: &mut egui::Ui, endpoint: AxisEndpoint) -> bool {
    let hit_size = endpoint.radius * 2.6;

    let hit_rect = egui::Rect::from_center_size(endpoint.position, egui::vec2(hit_size, hit_size));

    let response = ui.interact(hit_rect, egui::Id::new(endpoint.id), egui::Sense::click());

    let color = if response.hovered() {
        endpoint.color.linear_multiply(1.2)
    } else {
        endpoint.color
    };

    ui.painter()
        .circle_filled(endpoint.position, endpoint.radius, color);

    if let Some(label) = endpoint.label {
        ui.painter().text(
            endpoint.position,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
    }

    response.clicked()
}

pub fn show(
    context: &egui::Context,
    camera_eye: cgmath::Point3<f32>,
    camera_target: cgmath::Point3<f32>,
    camera_up: cgmath::Vector3<f32>,
) -> Option<ViewDirection> {
    let mut clicked_view = None;

    egui::Area::new(egui::Id::new("navigation_gizmo"))
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(crate::workspace_layout::viewport_left(context), -12.0 - crate::workspace_layout::bottom_inset(context)))
        .show(context, |ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(110.0, 110.0), egui::Sense::hover());

            let center = rect.center();
            let radius = 38.0;
            let forward = (camera_target - camera_eye).normalize();
            let camera_right = forward.cross(camera_up).normalize();
            let camera_screen_up = camera_right.cross(forward).normalize();

            let x_direction =
                project_axis(cgmath::Vector3::unit_x(), camera_right, camera_screen_up);

            let y_direction =
                project_axis(cgmath::Vector3::unit_y(), camera_right, camera_screen_up);

            let z_direction =
                project_axis(cgmath::Vector3::unit_z(), camera_right, camera_screen_up);

            let x_color = egui::Color32::from_rgb(220, 60, 70);
            let y_color = egui::Color32::from_rgb(80, 190, 90);
            let z_color = egui::Color32::from_rgb(70, 120, 230);

            let x_positive = center + x_direction * radius;
            let x_negative = center - x_direction * radius;

            let y_positive = center + y_direction * radius;
            let y_negative = center - y_direction * radius;

            let z_positive = center + z_direction * radius;
            let z_negative = center - z_direction * radius;

            // Draw axis lines.
            // X
            ui.painter()
                .line_segment([x_negative, x_positive], egui::Stroke::new(2.0, x_color));
            // Y
            ui.painter()
                .line_segment([y_negative, y_positive], egui::Stroke::new(2.0, y_color));
            // Z
            ui.painter()
                .line_segment([z_negative, z_positive], egui::Stroke::new(2.0, z_color));

            let endpoints = [
                // Positive X = Right
                AxisEndpoint {
                    position: x_positive,
                    color: x_color,
                    radius: 11.0,
                    label: Some("X"),
                    id: "gizmo_positive_x",
                    view: ViewDirection::Right,
                },
                // Negative X = Left
                AxisEndpoint {
                    position: x_negative,
                    color: x_color.linear_multiply(0.55),
                    radius: 7.0,
                    label: None,
                    id: "gizmo_negative_x",
                    view: ViewDirection::Left,
                },
                // Positive Y = Back
                AxisEndpoint {
                    position: y_positive,
                    color: y_color,
                    radius: 11.0,
                    label: Some("Y"),
                    id: "gizmo_positive_y",
                    view: ViewDirection::Back,
                },
                // Negative Y = Front
                AxisEndpoint {
                    position: y_negative,
                    color: y_color.linear_multiply(0.55),
                    radius: 7.0,
                    label: None,
                    id: "gizmo_negative_y",
                    view: ViewDirection::Front,
                },
                // Positive Z = Top
                AxisEndpoint {
                    position: z_positive,
                    color: z_color,
                    radius: 11.0,
                    label: Some("Z"),
                    id: "gizmo_positive_z",
                    view: ViewDirection::Top,
                },
                // Negative Z = Bottom
                AxisEndpoint {
                    position: z_negative,
                    color: z_color.linear_multiply(0.55),
                    radius: 7.0,
                    label: None,
                    id: "gizmo_negative_z",
                    view: ViewDirection::Bottom,
                },
            ];

            for endpoint in endpoints {
                if draw_endpoint(ui, endpoint) {
                    clicked_view = Some(endpoint.view);
                }
            }
        });

    clicked_view
}
