use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct EditorUi {
    pub selected_instance: usize,
    pub texture_enabled: bool,
    pub background: [f32; 4],
    pub pending_asset: Option<PathBuf>,
    pub pending_texture: Option<PathBuf>,
    pub asset_name: String,
    pub texture_name: String,
    pub show_grid: bool,
    pub grid_size: f32,
    pub grid_spacing: f32,
    pub status: String,

}

impl Default for EditorUi {
    fn default() -> Self {
        Self {
            selected_instance: 0,
            texture_enabled: true,
            background: [0.1, 0.2, 0.3, 1.0],
            pending_asset: None,
            pending_texture: None,
            asset_name: "t-pose.obj".into(),
            texture_name: "Happy-tree.png".into(),
            show_grid: true,
            grid_size: 20.0,
            grid_spacing: 1.0,
            status: "Ready".into(),
        }
    }
}
