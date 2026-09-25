use winit::event::MouseButton;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationAction {
    Tumble,
    Track,
    Dolly,
    Tilt,
    LensZoom,
}

#[derive(Debug)]
pub struct HoudiniNavigation {
    pub space_held: bool,
    pub alt_held: bool,
    pub shift_held: bool,
    pub control_held: bool,

    pub active_action: Option<NavigationAction>,

    pub cursor_position: Option<(f64, f64)>,
    pub previous_cursor: Option<(f64, f64)>,

    pub tumble_speed: f32,
    pub track_speed: f32,
    pub dolly_speed: f32,
    pub lens_speed: f32,
    pub precision_multiplier: f32,
}

impl Default for HoudiniNavigation {
    fn default() -> Self {
        Self {
            space_held: false,
            alt_held: false,
            shift_held: false,
            control_held: false,

            active_action: None,
            cursor_position: None,
            previous_cursor: None,

            tumble_speed: 0.006,
            track_speed: 1.0,
            dolly_speed: 0.01,
            lens_speed: 0.15,
            precision_multiplier: 0.2,
        }
    }
}

impl HoudiniNavigation {
    pub fn view_mode_active(&self) -> bool {
        self.space_held || self.alt_held
    }

    pub fn begin_mouse_action(&mut self, button: MouseButton) {
        if !self.view_mode_active() {
            return;
        }

        self.active_action = match (self.control_held, button) {
            (true, MouseButton::Left) => Some(NavigationAction::Tilt),
            (true, MouseButton::Right) => Some(NavigationAction::LensZoom),
            (false, MouseButton::Left) => Some(NavigationAction::Tumble),
            (false, MouseButton::Middle) => Some(NavigationAction::Track),
            (false, MouseButton::Right) => Some(NavigationAction::Dolly),
            _ => None,
        };
    }

    pub fn end_mouse_action(&mut self) {
        self.active_action = None;
        self.previous_cursor = None;
    }

    pub fn movement_scale(&self) -> f32 {
        if self.shift_held {
            self.precision_multiplier
        } else {
            1.0
        }
    }
}
