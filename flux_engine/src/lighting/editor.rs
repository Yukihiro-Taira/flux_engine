use super::{AreaShape, LightKind, LightingManager, ViewportLightingMode};

pub fn show(
    ui: &mut egui::Ui,
    manager: &mut LightingManager,
    geometry_instances: &[(usize, String)],
) {
    ui.heading("Lighting Mode");
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(
            &mut manager.mode,
            ViewportLightingMode::DefaultLight,
            "Work Light",
        );
        ui.selectable_value(
            &mut manager.mode,
            ViewportLightingMode::SceneLights,
            "Scene Lights",
        );
    });

    ui.separator();
    ui.menu_button("＋ Create Light", |ui| {
        for kind in [
            LightKind::Point,
            LightKind::Spot,
            LightKind::Directional,
            LightKind::Area(AreaShape::Rectangle),
            LightKind::Area(AreaShape::Disk),
            LightKind::Area(AreaShape::Line),
            LightKind::Area(AreaShape::Tube),
            LightKind::Area(AreaShape::Sphere),
            LightKind::Environment,
            LightKind::PhysicalSky,
            LightKind::Geometry,
            LightKind::Portal,
        ] {
            if ui.button(kind.name()).clicked() {
                manager.add(kind);
                manager.mode = ViewportLightingMode::SceneLights;
                ui.close();
            }
        }
    });

    ui.separator();
    ui.heading("Scene Lights");
    let mut changed = false;
    for light in &mut manager.lights {
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut light.enabled, "").changed();
            changed |= ui
                .checkbox(&mut light.viewport_enabled, "Viewport")
                .changed();
            if ui
                .selectable_label(manager.selected_light == Some(light.id), &light.name)
                .clicked()
            {
                manager.selected_light = Some(light.id);
            }
        });
    }

    if manager.selected_light.is_some() {
        ui.horizontal(|ui| {
            if ui.button("Delete selected").clicked() {
                manager.remove_selected();
            }
        });
    }

    ui.separator();
    if let Some(light) = manager.selected_mut() {
        ui.heading("Selected Light");
        changed |= ui.text_edit_singleline(&mut light.name).changed();
        ui.label(format!("Type: {}", light.kind.name()));
        changed |= ui.checkbox(&mut light.enabled, "Enable").changed();
        changed |= ui
            .checkbox(&mut light.viewport_enabled, "Enable in viewport")
            .changed();
        let selected_target_name = light
            .target_instance
            .and_then(|target| {
                geometry_instances
                    .iter()
                    .find(|(index, _)| *index == target)
                    .map(|(_, name)| name.as_str())
            })
            .unwrap_or("None (Manual Rotation)");
        egui::ComboBox::from_label("Look At")
            .selected_text(selected_target_name)
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut light.target_instance, None, "None (Manual Rotation)")
                    .changed();
                for (index, name) in geometry_instances {
                    changed |= ui
                        .selectable_value(&mut light.target_instance, Some(*index), name)
                        .changed();
                }
            });
        if light.target_instance.is_some() {
            ui.small("Rotation is controlled by the selected geometry target.");
        }

        ui.collapsing("Transform", |ui| {
            ui.label("Position");
            changed |= vector3(ui, &mut light.position, 0.05);
            ui.label("Rotation");
            ui.add_enabled_ui(light.target_instance.is_none(), |ui| {
                changed |= vector3(ui, &mut light.rotation_degrees, 0.5);
            });
        });

        ui.collapsing("Color and Power", |ui| {
            changed |= ui.color_edit_button_rgb(&mut light.color).changed();
            changed |= ui
                .checkbox(&mut light.use_temperature, "Use temperature")
                .changed();
            if light.use_temperature {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut light.temperature_kelvin, 1000.0..=40_000.0)
                            .text("Kelvin"),
                    )
                    .changed();
            }
            changed |= ui
                .add(
                    egui::DragValue::new(&mut light.intensity)
                        .speed(0.05)
                        .prefix("Intensity "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut light.exposure)
                        .speed(0.1)
                        .prefix("Exposure "),
                )
                .changed();
        });

        ui.collapsing("Shape", |ui| {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut light.radius)
                        .speed(0.01)
                        .prefix("Radius "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut light.size[0])
                        .speed(0.02)
                        .prefix("Width "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut light.size[1])
                        .speed(0.02)
                        .prefix("Height "),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut light.spread, 0.0..=1.0).text("Spread"))
                .changed();
            changed |= ui
                .checkbox(&mut light.single_sided, "Single sided")
                .changed();
        });

        if matches!(light.kind, LightKind::Spot) {
            ui.collapsing("Spot", |ui| {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut light.inner_angle_degrees, 0.1..=179.0)
                            .text("Inner angle"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut light.outer_angle_degrees, 0.1..=179.0)
                            .text("Outer angle"),
                    )
                    .changed();
                light.outer_angle_degrees =
                    light.outer_angle_degrees.max(light.inner_angle_degrees);
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut light.rolloff)
                            .speed(0.05)
                            .prefix("Rolloff "),
                    )
                    .changed();
            });
        }

        ui.collapsing("Contributions", |ui| {
            changed |= ui
                .add(egui::Slider::new(&mut light.diffuse_contribution, 0.0..=2.0).text("Diffuse"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut light.specular_contribution, 0.0..=2.0).text("Specular"),
                )
                .changed();
        });

        ui.collapsing("Shadows", |ui| {
            changed |= ui
                .checkbox(&mut light.shadows_enabled, "Enable shadows")
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut light.shadow_softness, 0.0..=10.0).text("Softness"))
                .changed();
            ui.label("Casts real-time mapped shadows onto scene geometry.");
        });
    } else {
        ui.label("No light selected");
    }

    ui.separator();
    ui.heading("Viewport Quality");
    changed |= ui
        .add(egui::Slider::new(&mut manager.max_lights, 1..=64).text("Max lights"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut manager.area_samples, 1..=64).text("Area samples"))
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut manager.environment_samples, 1..=128)
                .text("Environment samples"),
        )
        .changed();

    if changed {
        manager.touch();
    }
}

fn vector3(ui: &mut egui::Ui, value: &mut [f32; 3], speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        for (axis, component) in ["X", "Y", "Z"].into_iter().zip(value.iter_mut()) {
            changed |= ui
                .add(
                    egui::DragValue::new(component)
                        .speed(speed)
                        .prefix(format!("{axis} ")),
                )
                .changed();
        }
    });
    changed
}
