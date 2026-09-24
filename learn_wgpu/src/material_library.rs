use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaterialId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaceMaterialAssignment {
    /// Inclusive triangle index within one imported mesh group.
    pub first_face: u32,
    /// Exclusive triangle index within one imported mesh group.
    pub end_face: u32,
    pub material: MaterialId,
}

impl FaceMaterialAssignment {
    pub fn normalized(mut self, face_count: u32) -> Option<Self> {
        self.first_face = self.first_face.min(face_count);
        self.end_face = self.end_face.min(face_count);
        (self.first_face < self.end_face).then_some(self)
    }
}

pub struct SceneMaterial {
    pub id: MaterialId,
    pub name: String,
    pub material: crate::material::PbrMaterial,
    pub base_color_path: String,
    pub normal_path: String,
    pub roughness_path: String,
    pub emissive_path: String,
}

impl SceneMaterial {
    pub fn duplicate(&self, device: &wgpu::Device, id: MaterialId, name: String) -> Self {
        Self {
            id,
            name,
            material: crate::material::PbrMaterial::new_instance_from(device, &self.material),
            base_color_path: self.base_color_path.clone(),
            normal_path: self.normal_path.clone(),
            roughness_path: self.roughness_path.clone(),
            emissive_path: self.emissive_path.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_assignments_are_clamped_and_empty_ranges_rejected() {
        let material = MaterialId(4);
        let assignment = FaceMaterialAssignment {
            first_face: 2,
            end_face: 99,
            material,
        }
        .normalized(8)
        .unwrap();
        assert_eq!((assignment.first_face, assignment.end_face), (2, 8));
        assert!(
            FaceMaterialAssignment {
                first_face: 8,
                end_face: 9,
                material,
            }
            .normalized(8)
            .is_none()
        );
    }
}
