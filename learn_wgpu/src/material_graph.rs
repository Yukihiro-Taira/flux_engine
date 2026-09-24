use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::material::MaterialUniform;

type OutputPositions = HashMap<u64, egui::Pos2>;
type InputPositions = HashMap<(u64, String), egui::Pos2>;
type SocketPositions = (OutputPositions, InputPositions);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SocketType {
    Float,
    Color,
    Normal,
    Any,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    BaseColor { color: [f32; 4] },
    Float { name: String, value: f32 },
    Noise { scale: f32, detail: f32 },
    NormalMap { strength: f32 },
    Multiply,
    PbrOutput,
}

impl NodeKind {
    fn title(&self) -> &str {
        match self {
            Self::BaseColor { .. } => "Color",
            Self::Float { name, .. } => name,
            Self::Noise { .. } => "Noise",
            Self::NormalMap { .. } => "Normal Map",
            Self::Multiply => "Multiply",
            Self::PbrOutput => "Geo Output",
        }
    }

    fn output_type(&self) -> Option<SocketType> {
        match self {
            Self::BaseColor { .. } => Some(SocketType::Color),
            Self::Float { .. } | Self::Noise { .. } => Some(SocketType::Float),
            Self::NormalMap { .. } => Some(SocketType::Normal),
            Self::Multiply => Some(SocketType::Any),
            Self::PbrOutput => None,
        }
    }

    fn inputs(&self) -> &'static [(&'static str, SocketType)] {
        match self {
            Self::Multiply => &[("A", SocketType::Any), ("B", SocketType::Any)],
            Self::PbrOutput => &[
                ("Geometry", SocketType::Any),
                ("Visibility", SocketType::Float),
                ("Texture", SocketType::Color),
                ("Base Color", SocketType::Color),
                ("Metallic", SocketType::Float),
                ("Roughness", SocketType::Float),
                ("Normal", SocketType::Normal),
                ("Emission", SocketType::Any),
            ],
            _ => &[],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialNode {
    pub id: u64,
    pub position: [f32; 2],
    pub kind: NodeKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from: u64,
    pub to: u64,
    pub input: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialGraph {
    pub version: u32,
    pub nodes: Vec<MaterialNode>,
    pub connections: Vec<Connection>,
    pub next_id: u64,
}

impl Default for MaterialGraph {
    fn default() -> Self {
        Self {
            version: 2,
            nodes: vec![
                MaterialNode {
                    id: 1,
                    position: [20.0, 20.0],
                    kind: NodeKind::BaseColor {
                        color: [1.0, 1.0, 1.0, 1.0],
                    },
                },
                MaterialNode {
                    id: 2,
                    position: [215.0, 20.0],
                    kind: NodeKind::Float {
                        name: "Metallic".into(),
                        value: 0.0,
                    },
                },
                MaterialNode {
                    id: 3,
                    position: [215.0, 150.0],
                    kind: NodeKind::Float {
                        name: "Roughness".into(),
                        value: 0.5,
                    },
                },
                MaterialNode {
                    id: 4,
                    position: [430.0, 55.0],
                    kind: NodeKind::PbrOutput,
                },
                MaterialNode {
                    id: 5,
                    position: [20.0, 160.0],
                    kind: NodeKind::NormalMap { strength: 1.0 },
                },
            ],
            connections: vec![
                Connection {
                    from: 1,
                    to: 4,
                    input: "Base Color".into(),
                },
                Connection {
                    from: 2,
                    to: 4,
                    input: "Metallic".into(),
                },
                Connection {
                    from: 3,
                    to: 4,
                    input: "Roughness".into(),
                },
                Connection {
                    from: 5,
                    to: 4,
                    input: "Normal".into(),
                },
            ],
            next_id: 6,
        }
    }
}

pub struct MaterialGraphEditor {
    pub open: bool,
    pub graph: MaterialGraph,
    status: String,
    dragging_from: Option<u64>,
    cut_mode: bool,
    canvas_size: [f32; 2],
    node_menu_position: Option<egui::Pos2>,
    layout_initialized: bool,
}

impl Default for MaterialGraphEditor {
    fn default() -> Self {
        Self {
            open: false,
            graph: MaterialGraph::default(),
            status: "Drag from an output circle to an input circle".into(),
            dragging_from: None,
            cut_mode: false,
            canvas_size: [900.0, 360.0],
            node_menu_position: None,
            layout_initialized: false,
        }
    }
}

impl MaterialGraphEditor {
    pub fn toggle_open(&mut self) {
        self.open = !self.open;
        if self.open {
            // Reflow against the panel's current dimensions every time it is
            // opened. Its previous coordinates may belong to a differently
            // sized viewport and can otherwise be clamped on top of each other.
            self.layout_initialized = false;
        }
    }

    pub fn show(
        &mut self,
        root_ui: &mut egui::Ui,
        uniform: &mut MaterialUniform,
        preview: egui::TextureId,
    ) {
        if !self.open {
            return;
        }

        let context = root_ui.ctx().clone();
        let previous_cut_mode = self.cut_mode;
        self.cut_mode = context.input(|input| input.key_down(egui::Key::C))
            && !context.egui_wants_keyboard_input();
        if self.cut_mode != previous_cut_mode {
            self.dragging_from = None;
            self.status = if self.cut_mode {
                "Scissors active while C is held: click a wire to cut it".into()
            } else {
                "Connect mode active".into()
            };
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let save_pressed =
                context.input(|input| input.modifiers.command && input.key_pressed(egui::Key::S));
            let load_pressed =
                context.input(|input| input.modifiers.command && input.key_pressed(egui::Key::O));
            if save_pressed {
                self.save();
            }
            if load_pressed {
                self.load();
            }
        }

        egui::Panel::bottom("material_graph_panel")
            .resizable(true)
            .default_size(460.0)
            .size_range(300.0..=760.0)
            .show(root_ui, |ui| {
                let canvas = ui.available_rect_before_wrap();
                self.canvas_size = [canvas.width(), canvas.height()];
                if !self.layout_initialized {
                    self.arrange_default_nodes(canvas);
                    self.layout_initialized = true;
                }
                let painter = ui.painter_at(canvas);
                let socket_painter = context
                    .layer_painter(egui::LayerId::new(
                        egui::Order::Tooltip,
                        egui::Id::new("material_graph_sockets"),
                    ))
                    .with_clip_rect(canvas);
                painter.rect_filled(canvas, 0.0, egui::Color32::from_rgb(24, 26, 30));

                let tab_pressed = context.input(|input| input.key_pressed(egui::Key::Tab));
                if tab_pressed
                    && !context.egui_wants_keyboard_input()
                    && let Some(pointer) = context.pointer_latest_pos()
                    && canvas.contains(pointer)
                {
                    self.node_menu_position = Some(pointer);
                }
                if context.input(|input| input.key_pressed(egui::Key::Escape)) {
                    self.node_menu_position = None;
                }

                let mut positions: SocketPositions = (HashMap::new(), HashMap::new());

                for node in &mut self.graph.nodes {
                    node.position[0] = if matches!(node.kind, NodeKind::PbrOutput) {
                        canvas.width() * 0.8
                    } else {
                        node.position[0].clamp(0.0, (canvas.width() - 190.0).max(0.0))
                    };
                    node.position[1] =
                        node.position[1].clamp(0.0, (canvas.height() - 150.0).max(0.0));
                    let position = canvas.min + egui::vec2(node.position[0], node.position[1]);
                    let area = egui::Area::new(egui::Id::new(("material_node", node.id)))
                        .fixed_pos(position)
                        .order(egui::Order::Foreground)
                        .movable(!self.cut_mode)
                        .constrain_to(canvas)
                        .show(&context, |ui| {
                            egui::Frame::window(ui.style()).show(ui, |ui| {
                                ui.set_width(138.0);
                                ui.horizontal(|ui| {
                                    ui.strong(node.kind.title());
                                    if node.kind.output_type().is_some() {
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                let socket = ui.allocate_response(
                                                    egui::vec2(18.0, 18.0),
                                                    egui::Sense::drag(),
                                                );
                                                ui.painter().circle_filled(
                                                    socket.rect.center(),
                                                    6.0,
                                                    egui::Color32::from_rgb(80, 200, 255),
                                                );
                                                positions.0.insert(node.id, socket.rect.center());
                                                if socket.drag_started() && !self.cut_mode {
                                                    self.dragging_from = Some(node.id);
                                                    self.status =
                                                        "Drag onto an input socket".into();
                                                }
                                            },
                                        );
                                    }
                                });
                                ui.separator();
                                match &mut node.kind {
                                    NodeKind::BaseColor { color } => {
                                        ui.color_edit_button_rgba_unmultiplied(color);
                                    }
                                    NodeKind::Float { name, value } => {
                                        ui.add(
                                            egui::TextEdit::singleline(name).desired_width(126.0),
                                        );
                                        ui.add(egui::Slider::new(value, 0.0..=1.0));
                                    }
                                    NodeKind::Noise { scale, detail } => {
                                        ui.add(egui::Slider::new(scale, 0.1..=50.0).text("Scale"));
                                        ui.add(
                                            egui::Slider::new(detail, 1.0..=10.0).text("Detail"),
                                        );
                                    }
                                    NodeKind::NormalMap { strength } => {
                                        ui.add_sized(
                                            [126.0, 18.0],
                                            egui::Slider::new(strength, 0.0..=2.0).text("Strength"),
                                        );
                                    }
                                    NodeKind::Multiply => {
                                        for input in ["A", "B"] {
                                            ui.horizontal(|ui| {
                                                let socket = ui.allocate_response(
                                                    egui::vec2(14.0, 14.0),
                                                    egui::Sense::hover(),
                                                );
                                                ui.painter().circle_filled(
                                                    socket.rect.center(),
                                                    6.0,
                                                    if self.graph.connections.iter().any(|wire| {
                                                        wire.to == node.id && wire.input == input
                                                    }) {
                                                        egui::Color32::from_rgb(230, 160, 70)
                                                    } else {
                                                        egui::Color32::from_rgb(115, 125, 145)
                                                    },
                                                );
                                                positions.1.insert(
                                                    (node.id, input.to_owned()),
                                                    socket.rect.center(),
                                                );
                                                ui.label(input);
                                            });
                                        }
                                        ui.label("A x B");
                                    }
                                    NodeKind::PbrOutput => {
                                        for (name, _) in node.kind.inputs() {
                                            let connected =
                                                self.graph.connections.iter().any(|wire| {
                                                    wire.to == node.id && wire.input == *name
                                                });
                                            ui.horizontal(|ui| {
                                                let socket = ui.allocate_response(
                                                    egui::vec2(14.0, 14.0),
                                                    egui::Sense::hover(),
                                                );
                                                ui.painter().circle_filled(
                                                    socket.rect.center(),
                                                    6.0,
                                                    if connected {
                                                        egui::Color32::from_rgb(230, 160, 70)
                                                    } else {
                                                        egui::Color32::from_rgb(115, 125, 145)
                                                    },
                                                );
                                                positions.1.insert(
                                                    (node.id, (*name).to_owned()),
                                                    socket.rect.center(),
                                                );
                                                ui.label(if connected {
                                                    format!("{name}  [active]")
                                                } else {
                                                    format!("{name}  [disconnected]")
                                                });
                                            });
                                        }
                                    }
                                }
                            });
                        });
                    let new_position = area.response.rect.min - canvas.min;
                    node.position = [
                        new_position
                            .x
                            .clamp(0.0, (canvas.width() - area.response.rect.width()).max(0.0)),
                        new_position.y.clamp(
                            0.0,
                            (canvas.height() - area.response.rect.height()).max(0.0),
                        ),
                    ];
                }

                self.draw_connections(ui, &painter, &positions);
                self.handle_socket_interactions(ui, &positions);
                self.handle_cutting(ui, &socket_painter, &positions, canvas);

                if let Some(menu_position) = self.node_menu_position {
                    let mut selection = None;
                    egui::Area::new(egui::Id::new("material_node_menu"))
                        .fixed_pos(menu_position)
                        .order(egui::Order::Tooltip)
                        .constrain_to(canvas)
                        .show(&context, |ui| {
                            egui::Frame::popup(ui.style()).show(ui, |ui| {
                                ui.set_min_width(150.0);
                                ui.strong("Add Node");
                                ui.separator();
                                if ui.button("Color").clicked() {
                                    selection = Some(NodeKind::BaseColor {
                                        color: [0.8, 0.8, 0.8, 1.0],
                                    });
                                }
                                if ui.button("Value").clicked() {
                                    selection = Some(NodeKind::Float {
                                        name: "Value".into(),
                                        value: 0.5,
                                    });
                                }
                                if ui.button("Noise").clicked() {
                                    selection = Some(NodeKind::Noise {
                                        scale: 5.0,
                                        detail: 4.0,
                                    });
                                }
                                if ui.button("Multiply").clicked() {
                                    selection = Some(NodeKind::Multiply);
                                }
                                if ui.button("Normal Map").clicked() {
                                    selection = Some(NodeKind::NormalMap { strength: 1.0 });
                                }
                            });
                        });
                    if let Some(kind) = selection {
                        self.add_node_at(kind, menu_position - canvas.min);
                        self.node_menu_position = None;
                    }
                }

                let preview_size = egui::vec2(104.0, 124.0);
                let preview_position = canvas.max - preview_size - egui::vec2(10.0, 10.0);
                egui::Area::new(egui::Id::new("material_live_preview"))
                    .fixed_pos(preview_position)
                    .order(egui::Order::Foreground)
                    .constrain_to(canvas)
                    .show(&context, |ui| {
                        egui::Frame::window(ui.style()).show(ui, |ui| {
                            ui.label("Live Preview");
                            ui.image((preview, egui::vec2(88.0, 88.0)));
                        });
                    });
                ui.allocate_rect(canvas, egui::Sense::hover());
            });

        let previous_uniform = *uniform;
        self.apply_to_uniform(uniform);
        if previous_uniform.base_color != uniform.base_color
            || previous_uniform.properties != uniform.properties
            || previous_uniform.options != uniform.options
        {
            // State::update runs before the UI is evaluated. Request one more
            // frame so that the new graph material is guaranteed to be the
            // uniform used by the viewport, without requiring a gizmo click.
            context.request_repaint();
        }
    }

    pub fn apply_to_uniform(&self, uniform: &mut MaterialUniform) {
        self.evaluate(uniform);
    }

    fn arrange_default_nodes(&mut self, canvas: egui::Rect) {
        const MARGIN: f32 = 20.0;
        const NODE_WIDTH: f32 = 165.0;
        const NODE_HEIGHT: f32 = 190.0;

        let output_x = canvas.width() * 0.8;
        let usable_width = (output_x - MARGIN * 2.0).max(NODE_WIDTH);
        let columns = (usable_width / NODE_WIDTH).floor().max(1.0) as usize;

        let ordered = self
            .graph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                (!matches!(node.kind, NodeKind::PbrOutput)).then_some(index)
            })
            .collect::<Vec<_>>();

        for (cell, node_index) in ordered.into_iter().enumerate() {
            let column = cell % columns;
            let row = cell / columns;
            self.graph.nodes[node_index].position = [
                MARGIN + column as f32 * NODE_WIDTH,
                MARGIN + row as f32 * NODE_HEIGHT,
            ];
        }

        if let Some(output) = self
            .graph
            .nodes
            .iter_mut()
            .find(|node| matches!(node.kind, NodeKind::PbrOutput))
        {
            output.position = [output_x, 55.0];
        }
    }

    fn draw_connections(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        positions: &SocketPositions,
    ) {
        for connection in &self.graph.connections {
            if let (Some(start), Some(end)) = (
                positions.0.get(&connection.from),
                positions.1.get(&(connection.to, connection.input.clone())),
            ) {
                painter.add(connection_curve(
                    *start,
                    *end,
                    egui::Color32::from_rgb(230, 160, 70),
                    2.5,
                ));
            }
        }
        if let Some(from) = self.dragging_from
            && let (Some(start), Some(pointer)) =
                (positions.0.get(&from), ui.ctx().pointer_latest_pos())
        {
            painter.add(connection_curve(
                *start,
                pointer,
                egui::Color32::from_rgb(80, 200, 255),
                2.0,
            ));
        }
    }

    fn handle_socket_interactions(&mut self, ui: &mut egui::Ui, positions: &SocketPositions) {
        let released = ui.ctx().input(|input| input.pointer.any_released());
        let pointer = ui.ctx().pointer_latest_pos();
        let mut connection = None;
        for ((node_id, input_name), &position) in &positions.1 {
            if released
                && let (Some(from), Some(pointer)) = (self.dragging_from, pointer)
                && pointer.distance(position) <= 14.0
                && self.connection_types_match(from, *node_id, input_name)
            {
                connection = Some((from, *node_id, input_name.clone()));
            }
        }

        if released {
            if let Some((from, to, input)) = connection {
                self.graph
                    .connections
                    .retain(|wire| !(wire.to == to && wire.input == input));
                if !self.would_create_cycle(from, to) {
                    self.graph.connections.push(Connection {
                        from,
                        to,
                        input: input.clone(),
                    });
                    self.status = format!("Connected to {input}");
                    ui.ctx().request_repaint();
                } else {
                    self.status = "Connection rejected: it would create a cycle".into();
                }
            }
            self.dragging_from = None;
        }
    }

    fn handle_cutting(
        &mut self,
        ui: &mut egui::Ui,
        painter: &egui::Painter,
        positions: &SocketPositions,
        canvas: egui::Rect,
    ) {
        if !self.cut_mode {
            return;
        }
        let Some(pointer) = ui.ctx().pointer_latest_pos() else {
            return;
        };
        if !canvas.contains(pointer) {
            return;
        }
        ui.ctx().set_cursor_icon(egui::CursorIcon::None);
        painter.text(
            pointer + egui::vec2(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            "✂",
            egui::FontId::proportional(22.0),
            egui::Color32::from_rgb(255, 100, 110),
        );

        if ui.ctx().input(|input| input.pointer.primary_clicked()) {
            let cut = self
                .graph
                .connections
                .iter()
                .enumerate()
                .filter_map(|(index, wire)| {
                    let start = positions.0.get(&wire.from)?;
                    let end = positions.1.get(&(wire.to, wire.input.clone()))?;
                    Some((index, distance_to_curve(pointer, *start, *end)))
                })
                .filter(|(_, distance)| *distance < 12.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(index, _)| index);
            if let Some(index) = cut {
                let removed = self.graph.connections.remove(index);
                self.status = format!("Cut {} input wire", removed.input);
            }
        }
    }

    fn connection_types_match(&self, from: u64, to: u64, input: &str) -> bool {
        let Some(output_type) = self
            .graph
            .nodes
            .iter()
            .find(|node| node.id == from)
            .and_then(|node| node.kind.output_type())
        else {
            return false;
        };
        let Some(input_type) =
            self.graph
                .nodes
                .iter()
                .find(|node| node.id == to)
                .and_then(|node| {
                    node.kind
                        .inputs()
                        .iter()
                        .find(|(name, _)| *name == input)
                        .map(|(_, kind)| *kind)
                })
        else {
            return false;
        };
        output_type == SocketType::Any || input_type == SocketType::Any || output_type == input_type
    }

    fn would_create_cycle(&self, from: u64, to: u64) -> bool {
        if from == to {
            return true;
        }
        let mut stack = vec![to];
        let mut visited = HashSet::new();
        while let Some(node) = stack.pop() {
            if node == from {
                return true;
            }
            if visited.insert(node) {
                stack.extend(
                    self.graph
                        .connections
                        .iter()
                        .filter(|wire| wire.from == node)
                        .map(|wire| wire.to),
                );
            }
        }
        false
    }

    fn add_node_at(&mut self, kind: NodeKind, position: egui::Vec2) {
        let id = self.graph.next_id;
        self.graph.next_id += 1;
        self.graph.nodes.push(MaterialNode {
            id,
            position: [
                position
                    .x
                    .clamp(0.0, (self.canvas_size[0] - 150.0).max(0.0)),
                position
                    .y
                    .clamp(0.0, (self.canvas_size[1] - 100.0).max(0.0)),
            ],
            kind,
        });
    }

    fn evaluate(&self, uniform: &mut MaterialUniform) {
        uniform.base_color = [1.0; 4];
        uniform.properties[0] = 0.0;
        uniform.properties[1] = 0.5;
        uniform.properties[2] = 0.0;
        // Keep the material editor intensity unless the graph explicitly drives emission.
        uniform.options[3] = 0.0;

        let Some(output) = self
            .graph
            .nodes
            .iter()
            .find(|node| matches!(node.kind, NodeKind::PbrOutput))
        else {
            return;
        };

        if let Some(value) = self.evaluate_input(output.id, "Base Color", &mut HashSet::new()) {
            match value {
                NodeValue::Color(color) => uniform.base_color = color,
                NodeValue::Float(value) => uniform.base_color = [value, value, value, 1.0],
                NodeValue::Noise(scale) => uniform.options[3] = scale,
                NodeValue::Normal(_) => {}
            }
        }
        if let Some(NodeValue::Float(value)) =
            self.evaluate_input(output.id, "Metallic", &mut HashSet::new())
        {
            uniform.properties[0] = value.clamp(0.0, 1.0);
        }
        if let Some(NodeValue::Float(value)) =
            self.evaluate_input(output.id, "Roughness", &mut HashSet::new())
        {
            uniform.properties[1] = value.clamp(0.0, 1.0);
        }
        if let Some(NodeValue::Normal(strength)) =
            self.evaluate_input(output.id, "Normal", &mut HashSet::new())
        {
            uniform.properties[2] = strength;
        }
        if let Some(value) = self.evaluate_input(output.id, "Emission", &mut HashSet::new()) {
            uniform.options[1] = match value {
                NodeValue::Float(value) => value,
                NodeValue::Color(color) => (color[0] + color[1] + color[2]) / 3.0,
                NodeValue::Noise(_) | NodeValue::Normal(_) => 0.0,
            };
        }
    }

    fn evaluate_input(
        &self,
        node: u64,
        input: &str,
        visiting: &mut HashSet<u64>,
    ) -> Option<NodeValue> {
        let source = self
            .graph
            .connections
            .iter()
            .find(|wire| wire.to == node && wire.input == input)?
            .from;
        self.evaluate_node(source, visiting)
    }

    fn evaluate_node(&self, id: u64, visiting: &mut HashSet<u64>) -> Option<NodeValue> {
        if !visiting.insert(id) {
            return None;
        }
        let node = self.graph.nodes.iter().find(|node| node.id == id)?;
        let value = match &node.kind {
            NodeKind::BaseColor { color } => NodeValue::Color(*color),
            NodeKind::Float { value, .. } => NodeValue::Float(*value),
            NodeKind::Noise { scale, .. } => NodeValue::Noise(*scale),
            NodeKind::NormalMap { strength } => NodeValue::Normal(*strength),
            NodeKind::Multiply => {
                let left = self.evaluate_input(id, "A", visiting)?;
                let right = self.evaluate_input(id, "B", visiting)?;
                multiply_values(left, right)?
            }
            NodeKind::PbrOutput => return None,
        };
        visiting.remove(&id);
        Some(value)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Material Graph", &["json"])
            .set_file_name("material.json")
            .save_file()
        {
            match serde_json::to_string_pretty(&self.graph)
                .map_err(anyhow::Error::from)
                .and_then(|json| std::fs::write(&path, json).map_err(anyhow::Error::from))
            {
                Ok(()) => self.status = format!("Saved {}", path.display()),
                Err(error) => self.status = format!("Save failed: {error}"),
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Material Graph", &["json"])
            .pick_file()
        {
            match std::fs::read_to_string(&path)
                .map_err(anyhow::Error::from)
                .and_then(|json| serde_json::from_str(&json).map_err(anyhow::Error::from))
            {
                Ok(graph) => {
                    self.graph = graph;
                    self.dragging_from = None;
                    self.status = format!("Loaded {}", path.display());
                }
                Err(error) => self.status = format!("Load failed: {error}"),
            }
        }
    }
}

#[derive(Clone, Copy)]
enum NodeValue {
    Float(f32),
    Color([f32; 4]),
    Noise(f32),
    Normal(f32),
}

fn multiply_values(left: NodeValue, right: NodeValue) -> Option<NodeValue> {
    match (left, right) {
        (NodeValue::Float(a), NodeValue::Float(b)) => Some(NodeValue::Float(a * b)),
        (NodeValue::Color(mut color), NodeValue::Float(value))
        | (NodeValue::Float(value), NodeValue::Color(mut color)) => {
            for channel in &mut color[..3] {
                *channel *= value;
            }
            Some(NodeValue::Color(color))
        }
        (NodeValue::Noise(scale), NodeValue::Float(value))
        | (NodeValue::Float(value), NodeValue::Noise(scale)) => {
            Some(NodeValue::Noise(scale * value))
        }
        _ => None,
    }
}

fn connection_curve(
    start: egui::Pos2,
    end: egui::Pos2,
    color: egui::Color32,
    width: f32,
) -> egui::epaint::CubicBezierShape {
    let control = ((end.x - start.x).abs() * 0.5).max(40.0);
    egui::epaint::CubicBezierShape::from_points_stroke(
        [
            start,
            start + egui::vec2(control, 0.0),
            end - egui::vec2(control, 0.0),
            end,
        ],
        false,
        egui::Color32::TRANSPARENT,
        egui::Stroke::new(width, color),
    )
}

fn distance_to_curve(point: egui::Pos2, start: egui::Pos2, end: egui::Pos2) -> f32 {
    let control = ((end.x - start.x).abs() * 0.5).max(40.0);
    let p1 = start + egui::vec2(control, 0.0);
    let p2 = end - egui::vec2(control, 0.0);
    let mut closest = f32::INFINITY;
    let mut previous = start;
    for step in 1..=24 {
        let t = step as f32 / 24.0;
        let inverse = 1.0 - t;
        let current = start * (inverse * inverse * inverse)
            + p1.to_vec2() * (3.0 * inverse * inverse * t)
            + p2.to_vec2() * (3.0 * inverse * t * t)
            + end.to_vec2() * (t * t * t);
        closest = closest.min(distance_to_segment(point, previous, current));
        previous = current;
    }
    closest
}

fn distance_to_segment(point: egui::Pos2, start: egui::Pos2, end: egui::Pos2) -> f32 {
    let segment = end - start;
    let length_squared = segment.length_sq();
    if length_squared <= f32::EPSILON {
        return point.distance(start);
    }
    let t = ((point - start).dot(segment) / length_squared).clamp(0.0, 1.0);
    point.distance(start + segment * t)
}

#[cfg(test)]
mod emission_tests {
    use super::*;

    #[test]
    fn graph_without_emission_input_keeps_material_color_controls() {
        let mut uniform: MaterialUniform = bytemuck::Zeroable::zeroed();
        uniform.options[1] = 3.5;
        uniform.color_adjustments = [45.0, -90.0, 1.0, 0.0];
        uniform.emissive_color = [0.2, 0.5, 1.0, 1.0];
        MaterialGraphEditor::default().apply_to_uniform(&mut uniform);
        assert_eq!(uniform.options[1], 3.5);
        assert_eq!(uniform.color_adjustments, [45.0, -90.0, 1.0, 0.0]);
        assert_eq!(uniform.emissive_color, [0.2, 0.5, 1.0, 1.0]);
    }
}
