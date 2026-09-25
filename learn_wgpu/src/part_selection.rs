//! Transient viewport part selection. Material overrides use the existing saved face ranges.
use super::*;

#[derive(Default)]
pub(crate) struct PartSelection {
    pub active: bool,
    selected: Option<SelectedPart>,
    highlight: Option<material::PbrMaterial>,
}
struct SelectedPart {
    mesh: usize,
    instance: usize,
    faces: Vec<u32>,
    indices: wgpu::Buffer,
    index_count: u32,
    island: bool,
}

fn triangle_hit(
    origin: cgmath::Vector3<f32>,
    direction: cgmath::Vector3<f32>,
    points: [[f32; 3]; 3],
) -> Option<f32> {
    let a = cgmath::Vector3::from(points[0]);
    let e1 = cgmath::Vector3::from(points[1]) - a;
    let e2 = cgmath::Vector3::from(points[2]) - a;
    let p = direction.cross(e2);
    let determinant = e1.dot(p);
    if determinant.abs() < 1e-10 {
        return None;
    }
    let inv = determinant.recip();
    let t = origin - a;
    let u = t.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = t.cross(e1);
    let v = direction.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = e2.dot(q) * inv;
    (distance >= 0.0).then_some(distance)
}

// Include both position and UV so overlapping UV shells on disconnected geometry
// remain separate. Ignore normals: a hard normal edge is not a UV seam.
fn corner_key(vertex: &model::ModelVertex) -> [u32; 5] {
    let canonical = |v: f32| if v == 0.0 { 0 } else { v.to_bits() };
    [
        vertex.position[0],
        vertex.position[1],
        vertex.position[2],
        vertex.tex_coords[0],
        vertex.tex_coords[1],
    ]
    .map(canonical)
}
fn uv_island(vertices: &[model::ModelVertex], indices: &[u32], seed: usize) -> Vec<u32> {
    let count = indices.len() / 3;
    if seed >= count {
        return Vec::new();
    }
    let mut parent: Vec<usize> = (0..count).collect();
    fn root(parent: &mut [usize], mut n: usize) -> usize {
        while parent[n] != n {
            parent[n] = parent[parent[n]];
            n = parent[n];
        }
        n
    }
    let mut edges = HashMap::new();
    for (face, triangle) in indices.chunks_exact(3).enumerate() {
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let mut a = corner_key(&vertices[triangle[a] as usize]);
            let mut b = corner_key(&vertices[triangle[b] as usize]);
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            if a == b {
                continue;
            }
            if let Some(&other) = edges.get(&(a, b)) {
                let x = root(&mut parent, face);
                let y = root(&mut parent, other);
                parent[x] = y;
            } else {
                edges.insert((a, b), face);
            }
        }
    }
    let selected = root(&mut parent, seed);
    (0..count)
        .filter(|&face| root(&mut parent, face) == selected)
        .map(|f| f as u32)
        .collect()
}
fn face_ranges(
    faces: &[u32],
    material: material_library::MaterialId,
) -> Vec<material_library::FaceMaterialAssignment> {
    let mut ranges: Vec<material_library::FaceMaterialAssignment> = Vec::new();
    for &face in faces {
        if let Some(last) = ranges.last_mut().filter(|last| last.end_face == face) {
            last.end_face += 1;
        } else {
            ranges.push(material_library::FaceMaterialAssignment {
                first_face: face,
                end_face: face + 1,
                material,
            });
        }
    }
    ranges
}

impl State {
    pub(crate) fn toggle_part_selection(&mut self) {
        if self.part_selection.active {
            self.part_selection.active = false;
            self.part_selection.selected = None;
            self.editor.status = "Object selection · T to select model parts".into();
            return;
        }
        if self.obj_model.meshes.is_empty()
            || self.ground_plane.selected
            || self.lighting.selected_light.is_some()
        {
            self.editor.status = "Select an imported model first, then press T".into();
            return;
        }
        if self.part_selection.highlight.is_none() {
            let mut highlight =
                material::PbrMaterial::new_instance_from(&self.device, &self.pbr_material);
            highlight.uniform.base_color = [0.05, 0.75, 1.0, 1.0];
            highlight.uniform.options = [0.0, 2.0, 0.0, 0.0];
            highlight.uniform.emissive_color = [0.05, 0.65, 1.0, 1.0];
            highlight.uniform.properties = [0.0, 1.0, 0.0, 0.0];
            highlight.uniform.color_adjustments = [0.0, 0.0, 0.0, 1.0];
            highlight.uniform.transparency = [3.0, 0.5, 1.0, 0.35];
            highlight.uniform.inspection = [0.0; 4];
            highlight.upload(&self.queue);
            self.part_selection.highlight = Some(highlight);
        }
        self.part_selection.active = true;
        self.editor.status =
            "Part selection · click a group (UV island for a single-group model) · T to exit"
                .into();
    }

    pub(crate) fn select_part_at_cursor(&mut self) {
        let Some((x, y)) = self.houdini_navigation.cursor_position else {
            return;
        };
        let Some(inverse) = self.camera.build_view_projection_matrix().invert() else {
            return;
        };
        let nx = x as f32 / self.config.width.max(1) as f32 * 2.0 - 1.0;
        let ny = 1.0 - y as f32 / self.config.height.max(1) as f32 * 2.0;
        let near = inverse * cgmath::Vector4::new(nx, ny, 0.0, 1.0);
        let far = inverse * cgmath::Vector4::new(nx, ny, 1.0, 1.0);
        let origin = near.truncate() / near.w;
        let direction = (far.truncate() / far.w - origin).normalize();
        let instance_index = self.editor.selected_instance;
        let Some(instance) = self.instances.get(instance_index) else {
            return;
        };
        let mut hit: Option<(usize, usize, f32)> = None;
        for (group, mesh) in self.obj_model.meshes.iter().enumerate() {
            if !self.geo_group_enabled.get(group).copied().unwrap_or(true) {
                continue;
            }
            let matrix: cgmath::Matrix4<f32> = self.demo_instance_raw(instance, group).model.into();
            let Some(inverse) = matrix.invert() else {
                continue;
            };
            let o = inverse
                .transform_point(cgmath::Point3::from_vec(origin))
                .to_vec();
            let d = inverse.transform_vector(direction);
            if ray_box_distance(o, d, mesh.bounds_min.into(), mesh.bounds_max.into()).is_none() {
                continue;
            }
            for (face, triangle) in mesh.source_indices.chunks_exact(3).enumerate() {
                let points = [0, 1, 2].map(|i| mesh.source_vertices[triangle[i] as usize].position);
                if let Some(distance) = triangle_hit(o, d, points) {
                    if hit.is_none_or(|(_, _, closest)| distance < closest) {
                        hit = Some((group, face, distance));
                    }
                }
            }
        }
        let Some((group, face, _)) = hit else {
            self.part_selection.selected = None;
            return;
        };
        let mesh = &self.obj_model.meshes[group];
        let island = self.obj_model.meshes.len() == 1;
        let faces = if island {
            uv_island(&mesh.source_vertices, &mesh.source_indices, face)
        } else {
            (0..mesh.num_elements / 3).collect()
        };
        let indices: Vec<u32> = faces
            .iter()
            .flat_map(|&face| {
                mesh.source_indices[face as usize * 3..face as usize * 3 + 3]
                    .iter()
                    .copied()
            })
            .collect();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Selected part highlight indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let current_material = self
            .face_material_assignments
            .get(group)
            .and_then(|ranges| {
                ranges
                    .iter()
                    .rev()
                    .find(|r| face as u32 >= r.first_face && (face as u32) < r.end_face)
            })
            .map(|r| r.material)
            .or_else(|| self.mesh_material_assignments.get(group).copied());
        if let Some(id) =
            current_material.filter(|id| self.material_library.iter().any(|m| m.id == *id))
        {
            self.selected_library_material = id;
        }
        self.selected_material_mesh = group;
        self.editor.status = format!(
            "Selected {} · {} triangles",
            if island { "UV island" } else { "model group" },
            faces.len()
        );
        self.part_selection.selected = Some(SelectedPart {
            mesh: group,
            instance: instance_index,
            faces,
            indices: buffer,
            index_count: indices.len() as u32,
            island,
        });
    }

    pub(crate) fn part_selection_window(&mut self, context: &egui::Context) {
        if !self.part_selection.active {
            return;
        }
        let mut exit = false;
        let mut assign = false;
        let mut unique = false;
        let mut clear = false;
        egui::Window::new("Selected Geometry").id(egui::Id::new("selected_geometry_material_layer"))
            .default_pos(egui::pos2(20.0,280.0)).default_width(290.0).show(context,|ui| {
            ui.strong("Part selection active · T to exit");
            if let Some(selected) = &self.part_selection.selected {
                if let Some(mesh) = self.obj_model.meshes.get(selected.mesh) {
                    ui.add(egui::Label::new(&mesh.name).wrap());
                    ui.label(format!("{} · {} triangles",if selected.island {"UV island"} else {"Model group"},selected.faces.len()));
                    ui.separator();
                    ui.label("Material override");
                    let name = self.material_library.iter().find(|m|m.id==self.selected_library_material).map(|m|m.name.as_str()).unwrap_or("Choose material");
                    egui::ComboBox::from_id_salt("part_material").selected_text(name).show_ui(ui,|ui| {
                        for entry in &self.material_library { ui.selectable_value(&mut self.selected_library_material,entry.id,&entry.name); }
                    });
                    assign = ui.button("Apply to selected geometry").clicked();
                    unique = ui.button("Create independent material & edit").clicked();
                    if ui.button("Open material editor / import textures").clicked() { self.active_side_panel = Some(5); }
                    clear = ui.button("Remove selected geometry override").clicked();
                    ui.small("Overrides are saved with the project and shared by its model instances.");
                }
            } else { ui.label("Click a part of the selected model. Single-group models use UV islands."); }
            exit = ui.button("Exit part selection (T)").clicked();
        });
        if unique {
            if let Some(source) = self
                .material_library
                .iter()
                .position(|m| m.id == self.selected_library_material)
            {
                let id = material_library::MaterialId(self.next_material_id);
                self.next_material_id += 1;
                let entry = self.material_library[source].duplicate(
                    &self.device,
                    id,
                    format!("{} · Selected geometry", self.material_library[source].name),
                );
                self.material_library.push(entry);
                self.selected_library_material = id;
                assign = true;
                self.active_side_panel = Some(5);
            }
        }
        if let Some(selected) = &self.part_selection.selected {
            if (assign || clear) && selected.mesh < self.face_material_assignments.len() {
                // Split existing ranges so replacing/removing one island preserves all other overrides.
                let old = std::mem::take(&mut self.face_material_assignments[selected.mesh]);
                let mut kept = Vec::new();
                for range in old {
                    let mut start = range.first_face;
                    let first = selected
                        .faces
                        .partition_point(|&face| face < range.first_face);
                    let end = selected
                        .faces
                        .partition_point(|&face| face < range.end_face);
                    for &face in &selected.faces[first..end] {
                        if start < face {
                            kept.push(material_library::FaceMaterialAssignment {
                                first_face: start,
                                end_face: face,
                                material: range.material,
                            });
                        }
                        start = face + 1;
                    }
                    if start < range.end_face {
                        kept.push(material_library::FaceMaterialAssignment {
                            first_face: start,
                            end_face: range.end_face,
                            material: range.material,
                        });
                    }
                }
                if assign {
                    kept.extend(face_ranges(&selected.faces, self.selected_library_material));
                }
                self.face_material_assignments[selected.mesh] = kept;
            }
        }
        if exit {
            self.toggle_part_selection();
        }
    }

    pub(crate) fn draw_part_highlight<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if !self.part_selection.active {
            return;
        }
        let (Some(selected), Some(material)) = (
            &self.part_selection.selected,
            &self.part_selection.highlight,
        ) else {
            return;
        };
        let Some(mesh) = self.obj_model.meshes.get(selected.mesh) else {
            return;
        };
        if selected.instance >= self.editor.visible_instance_count
            || !self
                .geo_group_enabled
                .get(selected.mesh)
                .copied()
                .unwrap_or(true)
        {
            return;
        }
        model::SurfaceDraw {
            vertices: &mesh.vertex_buffer,
            indices: &selected.indices,
            instances: self.demo_buffer(selected.mesh),
            material: &material.bind_group,
            index_range: 0..selected.index_count,
            instance_range: selected.instance as u32..selected.instance as u32 + 1,
            depth: 0.0,
            blend: true,
        }
        .draw(pass, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn vertex(x: f32, y: f32, u: f32, v: f32) -> model::ModelVertex {
        model::ModelVertex {
            position: [x, y, 0.],
            tex_coords: [u, v],
            normal: [0., 0., 1.],
            tangent: [1., 0., 0., 1.],
        }
    }
    #[test]
    fn uv_islands_join_duplicated_vertices_but_respect_seams_and_disconnected_shells() {
        let vertices = vec![
            vertex(0., 0., 0., 0.),
            vertex(1., 0., 1., 0.),
            vertex(0., 1., 0., 1.),
            vertex(1., 0., 1., 0.),
            vertex(1., 1., 1., 1.),
            vertex(0., 1., 0., 1.),
            // Shares a geometric edge, but the UV edge differs.
            vertex(1., 0., 0., 0.),
            vertex(2., 0., 1., 0.),
            vertex(1., 1., 0., 1.),
            // Same UVs as the first triangle on disconnected geometry.
            vertex(4., 0., 0., 0.),
            vertex(5., 0., 1., 0.),
            vertex(4., 1., 0., 1.),
        ];
        let indices: Vec<u32> = (0..12).collect();
        assert_eq!(uv_island(&vertices, &indices, 0), vec![0, 1]);
        assert_eq!(uv_island(&vertices, &indices, 2), vec![2]);
        assert_eq!(uv_island(&vertices, &indices, 3), vec![3]);
    }
    #[test]
    fn ray_hits_triangles_not_their_empty_bounding_box_corners() {
        let points = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]];
        let direction = cgmath::Vector3::new(0., 0., -2.);
        assert_eq!(
            triangle_hit(cgmath::Vector3::new(0.2, 0.2, 1.), direction, points),
            Some(0.5)
        );
        assert_eq!(
            triangle_hit(cgmath::Vector3::new(0.9, 0.9, 1.), direction, points),
            None
        );
        assert_eq!(
            triangle_hit(cgmath::Vector3::new(0.2, 0.2, -1.), direction, points),
            None
        );
    }
    #[test]
    fn island_ranges_never_include_unselected_faces() {
        let ranges = face_ranges(&[0, 1, 4, 6, 7], material_library::MaterialId(3));
        assert_eq!(
            ranges
                .iter()
                .map(|r| (r.first_face, r.end_face))
                .collect::<Vec<_>>(),
            vec![(0, 2), (4, 5), (6, 8)]
        );
    }
}
