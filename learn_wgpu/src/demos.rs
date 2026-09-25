//! Reversible presentation presets: model geometry and object transforms stay untouched.
use cgmath::{InnerSpace, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Deconstruction {
    pub enabled: bool,
    pub distance: f32,
    pub random: bool,
    pub seed: u32,
}

impl Default for Deconstruction {
    fn default() -> Self {
        Self {
            enabled: false,
            distance: 0.0,
            random: false,
            seed: 1,
        }
    }
}

impl Deconstruction {
    pub fn sanitize(&mut self) {
        self.distance = if self.distance.is_finite() {
            self.distance.clamp(0.0, 100_000.0)
        } else {
            0.0
        };
    }

    pub fn offset(&self, index: usize, count: usize) -> Vector3<f32> {
        if !self.enabled || count < 2 || self.distance <= 0.0 {
            return Vector3::new(0.0, 0.0, 0.0);
        }
        let (z, angle) = if self.random {
            let key = self
                .seed
                .wrapping_add((index as u32).wrapping_mul(0x9e3779b9));
            (
                unit_random(key) * 2.0 - 1.0,
                unit_random(key ^ 0xa511e9b3) * std::f32::consts::TAU,
            )
        } else {
            // Fibonacci sphere: evenly distributed directions without shared axes.
            (
                1.0 - 2.0 * (index as f32 + 0.5) / count as f32,
                index as f32 * 2.3999631,
            )
        };
        let radius = (1.0 - z * z).max(0.0).sqrt();
        Vector3::new(radius * angle.cos(), radius * angle.sin(), z).normalize() * self.distance
    }
}

fn unit_random(mut value: u32) -> f32 {
    value = (value ^ (value >> 16)).wrapping_mul(0x7feb352d);
    value = (value ^ (value >> 15)).wrapping_mul(0x846ca68b);
    value ^= value >> 16;
    (value >> 8) as f32 / 16_777_216.0
}

impl crate::State {
    pub(super) fn demo_instance_raw(
        &self,
        instance: &crate::Instance,
        group: usize,
    ) -> crate::InstanceRaw {
        let mut raw = instance.to_raw();
        if let Some(transform)=self.obj_model.meshes.get(group).and_then(|mesh|self.timeline.transform_for(&mesh.name)) {
            let model: cgmath::Matrix4<f32> = raw.model.into();
            raw.model=(model*transform).into();
        }
        let offset = self
            .deconstruction
            .offset(group, self.obj_model.meshes.len());
        // World-space translation guarantees the same travel distance even for
        // nonuniformly scaled instances. Rotation, scale, and mesh data are preserved.
        raw.model[3][0] += offset.x;
        raw.model[3][1] += offset.y;
        raw.model[3][2] += offset.z;
        raw
    }

    pub(super) fn update_demo_buffers(&mut self) {
        if ((!self.deconstruction.enabled || self.deconstruction.distance == 0.0 || self.obj_model.meshes.len() < 2)
            && !self.timeline.has_transforms()) || self.instances.is_empty()
        {
            self.demo_buffers.clear();
            return;
        }
        if !self.demo_buffers.is_empty()
            && self
                .demo_cached
                .as_ref()
                .is_some_and(|(preset, instances)| {
                    preset == &self.deconstruction && instances == &self.instances
                })
            && self.demo_buffers.len() == self.obj_model.meshes.len()
        {
            return;
        }
        self.demo_cached = Some((self.deconstruction.clone(), self.instances.clone()));
        let size = (self.instances.len() * std::mem::size_of::<crate::InstanceRaw>()) as u64;
        if self.demo_buffers.len() != self.obj_model.meshes.len()
            || self.demo_buffers.first().is_some_and(|b| b.size() != size)
        {
            self.demo_buffers = (0..self.obj_model.meshes.len())
                .map(|_| {
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Deconstruction group transforms"),
                        size,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect();
        }
        for (group, buffer) in self.demo_buffers.iter().enumerate() {
            let transforms: Vec<_> = self
                .instances
                .iter()
                .map(|instance| self.demo_instance_raw(instance, group))
                .collect();
            self.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&transforms));
        }
    }

    pub(super) fn demo_buffer(&self, group: usize) -> &wgpu::Buffer {
        self.demo_buffers
            .get(group)
            .unwrap_or(&self.instance_buffer)
    }

    pub(super) fn demos_window(&mut self, context: &egui::Context) {
        if self.active_side_panel != Some(6) {
            return;
        }
        let groups = self.obj_model.meshes.len();
        let span = self
            .model_local_bounds()
            .map(|(min, max)| {
                let extent = max - min;
                let scale = self
                    .instances
                    .iter()
                    .map(|i| {
                        i.global_scale.abs()
                            * i.scale.x.abs().max(i.scale.y.abs()).max(i.scale.z.abs())
                    })
                    .fold(0.0_f32, f32::max)
                    .max(0.000001);
                extent.x.max(extent.y).max(extent.z) * scale
            })
            .unwrap_or(1.0)
            .max(0.01);
        let slider_max = (span * 2.0)
            .max(self.deconstruction.distance)
            .min(100_000.0);
        let mut frame = false;
        egui::Window::new("Demos").constrain_to(crate::timeline::workspace_rect(context))
            .id(egui::Id::new("visual_demos_window"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-170.0, 12.0))
            .default_width(360.0).resizable(true).vscroll(true)
            .max_height((crate::timeline::workspace_rect(context).height() - 24.0).max(160.0))
            .show(context, |ui| {
                ui.heading("Visual demos");
                ui.weak("Presentation presets for your model");
                ui.separator();
                ui.strong("01  Product Deconstruction");
                ui.add(egui::Label::new("Separate the model into its mesh groups for an exploded product view. Every group travels the same distance.").wrap());
                ui.add_space(8.0);
                ui.label(format!("{groups} mesh groups"));
                if groups < 2 {
                    ui.add(egui::Label::new("Import an OBJ or FBX with at least two mesh groups to use this demo. A single merged mesh cannot be separated into parts.").wrap());
                }
                ui.add_enabled_ui(groups >= 2, |ui| {
                    ui.checkbox(&mut self.deconstruction.enabled, "Enable deconstruction");
                    ui.add(egui::Slider::new(&mut self.deconstruction.distance, 0.0..=slider_max).text("Distance").suffix(" m"));
                    ui.small("0 = assembled · distance is per group in world units");
                    if ui.checkbox(&mut self.deconstruction.random, "Random directions").changed() && self.deconstruction.random {
                        self.deconstruction.seed = self.deconstruction.seed.wrapping_add(1);
                    }
                    ui.add_enabled_ui(self.deconstruction.random, |ui| {
                        if ui.button("New random arrangement").clicked() { self.deconstruction.seed = self.deconstruction.seed.wrapping_add(1); }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Frame result").clicked() { frame = true; }
                        if ui.button("Reassemble").clicked() { self.deconstruction.distance = 0.0; self.deconstruction.enabled = false; }
                    });
                });
                ui.separator();
                ui.add(egui::Label::new(egui::RichText::new("Your original geometry stays unchanged. Save an .fx project to keep this arrangement and its random seed.").small()).wrap());
            });
        if frame {
            self.frame_all_instances();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn groups_travel_equal_distances_and_reassemble_exactly() {
        for random in [false, true] {
            let mut demo = Deconstruction {
                enabled: true,
                distance: 3.5,
                random,
                seed: 42,
            };
            let offsets: Vec<_> = (0..20).map(|i| demo.offset(i, 20)).collect();
            for (i, offset) in offsets.iter().enumerate() {
                assert!((offset.magnitude() - 3.5).abs() < 0.00001);
                for other in &offsets[..i] {
                    assert!((offset - other).magnitude() > 0.001);
                }
            }
            demo.distance = 0.0;
            assert_eq!(demo.offset(0, 20), Vector3::new(0.0, 0.0, 0.0));
            demo.distance = 3.5;
            demo.enabled = false;
            assert_eq!(demo.offset(0, 20), Vector3::new(0.0, 0.0, 0.0));
        }
    }
    #[test]
    fn random_arrangements_are_stable_and_project_serializable() {
        let mut demo = Deconstruction {
            enabled: true,
            distance: 2.0,
            random: true,
            seed: 123,
        };
        let saved = serde_json::to_string(&demo).unwrap();
        let restored: Deconstruction = serde_json::from_str(&saved).unwrap();
        assert_eq!(demo.offset(3, 10), restored.offset(3, 10));
        demo.seed += 1;
        assert_ne!(demo.offset(3, 10), restored.offset(3, 10));
        assert_eq!(demo.offset(0, 1), Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(
            serde_json::from_str::<Deconstruction>("{}").unwrap(),
            Deconstruction::default()
        );
    }
}
