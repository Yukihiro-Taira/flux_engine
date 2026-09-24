struct ShadowIndex { index: vec4<u32> };
@group(0) @binding(0) var<storage, read> shadow_matrices: array<mat4x4<f32>>;
@group(0) @binding(1) var<uniform> shadow_index: ShadowIndex;

struct VertexInput { @location(0) position: vec3<f32> };
struct InstanceInput {
    @location(5) model_0: vec4<f32>, @location(6) model_1: vec4<f32>,
    @location(7) model_2: vec4<f32>, @location(8) model_3: vec4<f32>,
};

@vertex
fn vs_shadow(vertex: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(instance.model_0, instance.model_1, instance.model_2, instance.model_3);
    return shadow_matrices[shadow_index.index.x] * model * vec4<f32>(vertex.position, 1.0);
}
