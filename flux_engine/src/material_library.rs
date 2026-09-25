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

/// Resolve last-wins overrides at range boundaries, rather than searching every
/// override for every triangle (UV islands may contain many disjoint ranges).
pub fn resolved_face_ranges(
    count: u32,
    base: Option<MaterialId>,
    overrides: &[FaceMaterialAssignment],
) -> Vec<(u32, u32, Option<MaterialId>)> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut events = BTreeMap::<u32, Vec<(usize, bool)>>::new();
    events.entry(0).or_default();
    events.entry(count).or_default();
    for (index, range) in overrides.iter().enumerate() {
        let start = range.first_face.min(count);
        let end = range.end_face.min(count);
        if start < end {
            events.entry(start).or_default().push((index, true));
            events.entry(end).or_default().push((index, false));
        }
    }
    let mut active = BTreeSet::new();
    let mut result: Vec<(u32, u32, Option<MaterialId>)> = Vec::new();
    let mut previous = 0;
    for (position, changes) in events {
        if position > previous {
            let material = active
                .last()
                .map(|&index: &usize| overrides[index].material)
                .or(base);
            if let Some(last) = result.last_mut().filter(|last| last.2 == material) {
                last.1 = position;
            } else {
                result.push((previous, position, material));
            }
        }
        for (index, enabled) in changes {
            if enabled {
                active.insert(index);
            } else {
                active.remove(&index);
            }
        }
        previous = position;
    }
    result
}

#[cfg(test)]
mod range_tests {
    use super::*;
    #[test]
    fn boundary_resolution_matches_last_override_wins() {
        let base = Some(MaterialId(1));
        let overrides = vec![
            FaceMaterialAssignment {
                first_face: 1,
                end_face: 8,
                material: MaterialId(2),
            },
            FaceMaterialAssignment {
                first_face: 3,
                end_face: 5,
                material: MaterialId(3),
            },
            FaceMaterialAssignment {
                first_face: 7,
                end_face: 99,
                material: MaterialId(4),
            },
        ];
        let ranges = resolved_face_ranges(10, base, &overrides);
        for face in 0..10 {
            let expected = overrides
                .iter()
                .rev()
                .find(|r| face >= r.first_face && face < r.end_face)
                .map(|r| r.material)
                .or(base);
            assert_eq!(
                ranges.iter().find(|r| face >= r.0 && face < r.1).unwrap().2,
                expected
            );
        }
        assert!(resolved_face_ranges(0, base, &overrides).is_empty());
    }
}
