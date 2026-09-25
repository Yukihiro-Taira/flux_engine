use super::{LightId, LightKind, SceneLight, ViewportLightingMode};

#[derive(Clone, PartialEq)]
pub struct LightingManager {
    pub mode: ViewportLightingMode,
    pub lights: Vec<SceneLight>,
    pub selected_light: Option<LightId>,
    pub max_lights: usize,
    pub area_samples: u32,
    pub environment_samples: u32,
    pub revision: u64,
    next_id: u64,
}

impl Default for LightingManager {
    fn default() -> Self {
        Self {
            mode: ViewportLightingMode::DefaultLight,
            lights: Vec::new(),
            selected_light: None,
            max_lights: 64,
            area_samples: 16,
            environment_samples: 64,
            revision: 1,
            next_id: 1,
        }
    }
}

impl LightingManager {
    pub fn add(&mut self, kind: LightKind) -> LightId {
        let id = LightId(self.next_id);
        self.next_id += 1;
        self.lights.push(SceneLight::new(id, kind));
        self.selected_light = Some(id);
        self.touch();
        id
    }

    pub fn remove_selected(&mut self) {
        let Some(id) = self.selected_light.take() else {
            return;
        };
        self.lights.retain(|light| light.id != id);
        self.touch();
    }

    pub fn selected_mut(&mut self) -> Option<&mut SceneLight> {
        let id = self.selected_light?;
        self.lights.iter_mut().find(|light| light.id == id)
    }

    pub fn active_environment(&self) -> Option<&SceneLight> {
        self.lights.iter().find(|light| {
            light.active_in_viewport()
                && matches!(light.kind, LightKind::Environment | LightKind::PhysicalSky)
        })
    }

    pub fn active_direct_lights(&self) -> impl Iterator<Item = &SceneLight> {
        self.lights
            .iter()
            .filter(|light| light.active_in_viewport() && light.kind.contributes_directly())
    }

    pub fn effective_gpu_mode(&self) -> u32 {
        match self.mode {
            ViewportLightingMode::DefaultLight => 1,
            ViewportLightingMode::SceneLights => {
                if self.active_environment().is_none()
                    && self.active_direct_lights().next().is_none()
                {
                    1
                } else {
                    2
                }
            }
        }
    }

    pub fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    pub fn restore_project(
        &mut self,
        mode: ViewportLightingMode,
        lights: Vec<SceneLight>,
        selected_light: Option<LightId>,
        area_samples: u32,
        environment_samples: u32,
    ) {
        self.next_id = lights
            .iter()
            .map(|light| light.id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.mode = mode;
        self.lights = lights;
        self.selected_light =
            selected_light.filter(|selected| self.lights.iter().any(|light| light.id == *selected));
        self.area_samples = area_samples;
        self.environment_samples = environment_samples;
        self.touch();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_mode_falls_back_when_all_scene_lights_are_disabled() {
        let mut manager = LightingManager {
            mode: ViewportLightingMode::SceneLights,
            ..Default::default()
        };
        let id = manager.add(LightKind::Point);
        assert_eq!(manager.effective_gpu_mode(), 2);
        manager
            .lights
            .iter_mut()
            .find(|light| light.id == id)
            .unwrap()
            .enabled = false;
        assert_eq!(manager.effective_gpu_mode(), 1);
    }

    #[test]
    fn active_environment_prevents_default_fallback() {
        let mut manager = LightingManager {
            mode: ViewportLightingMode::SceneLights,
            ..Default::default()
        };
        manager.add(LightKind::Environment);
        assert_eq!(manager.effective_gpu_mode(), 2);
    }

    #[test]
    fn work_light_is_the_default() {
        let manager = LightingManager::default();
        assert_eq!(manager.mode, ViewportLightingMode::DefaultLight);
        assert_eq!(manager.effective_gpu_mode(), 1);
    }
}
