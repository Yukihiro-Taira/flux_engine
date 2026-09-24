use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LightId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewportLightingMode {
    DefaultLight,
    SceneLights,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AreaShape {
    Rectangle,
    Disk,
    Line,
    Tube,
    Sphere,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LightKind {
    Point,
    Spot,
    Directional,
    Area(AreaShape),
    Environment,
    PhysicalSky,
    Geometry,
    Portal,
}

impl LightKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Point => "Point",
            Self::Spot => "Spot",
            Self::Directional => "Distant",
            Self::Area(AreaShape::Rectangle) => "Rectangle",
            Self::Area(AreaShape::Disk) => "Disk",
            Self::Area(AreaShape::Line) => "Line",
            Self::Area(AreaShape::Tube) => "Tube",
            Self::Area(AreaShape::Sphere) => "Sphere",
            Self::Environment => "Environment",
            Self::PhysicalSky => "Physical Sky",
            Self::Geometry => "Geometry",
            Self::Portal => "Portal",
        }
    }

    pub fn gpu_type(&self) -> u32 {
        match self {
            Self::Point => 0,
            Self::Spot => 1,
            Self::Directional => 2,
            Self::Area(AreaShape::Rectangle) => 3,
            Self::Area(AreaShape::Disk) => 4,
            Self::Area(AreaShape::Line) => 5,
            Self::Area(AreaShape::Tube) => 6,
            Self::Area(AreaShape::Sphere) => 7,
            Self::Geometry => 8,
            Self::PhysicalSky => 9,
            Self::Environment => 10,
            Self::Portal => 11,
        }
    }

    pub fn contributes_directly(&self) -> bool {
        !matches!(self, Self::Environment | Self::Portal)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneLight {
    pub id: LightId,
    pub name: String,
    pub enabled: bool,
    pub viewport_enabled: bool,
    pub kind: LightKind,
    pub position: [f32; 3],
    pub rotation_degrees: [f32; 3],
    #[serde(default)]
    pub target_instance: Option<usize>,
    pub color: [f32; 3],
    pub use_temperature: bool,
    pub temperature_kelvin: f32,
    pub intensity: f32,
    pub exposure: f32,
    pub radius: f32,
    pub size: [f32; 2],
    pub spread: f32,
    pub inner_angle_degrees: f32,
    pub outer_angle_degrees: f32,
    pub rolloff: f32,
    pub single_sided: bool,
    pub diffuse_contribution: f32,
    pub specular_contribution: f32,
    pub shadows_enabled: bool,
    pub shadow_softness: f32,
}

impl SceneLight {
    pub fn new(id: LightId, kind: LightKind) -> Self {
        let name = format!("{} {}", kind.name(), id.0);
        let intensity = match &kind {
            // Point/spot power is interpreted as luminous-power-like energy;
            // area types use emitted radiance; distant/sky types use radiance.
            LightKind::Point | LightKind::Spot => 1_000.0,
            LightKind::Directional | LightKind::PhysicalSky => 3.0,
            LightKind::Area(_) | LightKind::Geometry => 100.0,
            LightKind::Environment | LightKind::Portal => 1.0,
        };
        Self {
            id,
            name,
            enabled: true,
            viewport_enabled: true,
            kind,
            position: [0.0, 0.0, 2.0],
            rotation_degrees: [45.0, 0.0, 35.0],
            target_instance: None,
            color: [1.0, 1.0, 1.0],
            use_temperature: false,
            temperature_kelvin: 6500.0,
            intensity,
            exposure: 0.0,
            radius: 0.25,
            size: [1.0, 1.0],
            spread: 1.0,
            inner_angle_degrees: 35.0,
            outer_angle_degrees: 45.0,
            rolloff: 1.0,
            single_sided: false,
            diffuse_contribution: 1.0,
            specular_contribution: 1.0,
            shadows_enabled: false,
            shadow_softness: 1.0,
        }
    }

    pub fn active_in_viewport(&self) -> bool {
        self.enabled && self.viewport_enabled
    }

    pub fn power(&self) -> f32 {
        self.intensity * self.exposure.exp2()
    }
}
