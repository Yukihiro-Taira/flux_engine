use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct EditorUi {
    pub selected_instance: usize,
    pub visible_instance_count: usize,
    pub texture_enabled: bool,
    pub background: [f32; 4],
    pub pending_asset: Option<PathBuf>,
    pub model_path_input: String,
    pub pending_texture: Option<PathBuf>,
    pub pending_normal: Option<PathBuf>,
    pub pending_metallic_roughness: Option<PathBuf>,
    pub mtl_path_input: String,
    pub asset_name: String,
    pub texture_name: String,
    pub pending_hdri: Option<PathBuf>,
    pub hdri_name: Option<String>,
    pub hdri_image_disabled: bool,
    pub hdri_intensity: f32,
    pub hdri_exposure: f32,
    pub hdri_rotation: f32,

    pub show_grid: bool,
    pub grid_size: f32,
    pub grid_spacing: f32,
    pub grid_color: [f32; 4],

    pub status: String,
}

impl Default for EditorUi {
    fn default() -> Self {
        Self {
            selected_instance: 0,
            visible_instance_count: 1,
            texture_enabled: false,
            background: [0.0, 0.0, 0.0, 1.0],
            pending_asset: None,
            model_path_input: String::new(),
            pending_texture: None,
            pending_normal: None,
            pending_metallic_roughness: None,
            mtl_path_input: String::new(),
            asset_name: "t-pose.obj".into(),
            texture_name: "None".into(),
            pending_hdri: None,
            hdri_name: None,
            hdri_image_disabled: false,
            hdri_intensity: 1.0,
            hdri_exposure: 0.0,
            hdri_rotation: 0.0,
            show_grid: true,
            grid_size: 20.0,
            // One world unit is one meter.
            grid_spacing: 1.0,
            grid_color: [0.55, 0.58, 0.62, 0.15],
            status: "Ready".into(),
        }
    }
}
