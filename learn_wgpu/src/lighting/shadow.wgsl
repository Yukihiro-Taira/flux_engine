struct ShadowIndex { index: vec4<u32> };
@group(0) @binding(0) var<storage, read> shadow_matrices: array<mat4x4<f32>>;
@group(0) @binding(1) var<uniform> shadow_index: ShadowIndex;
struct MaterialUniform {
    base_color: vec4<f32>,
    color_adjustments: vec4<f32>,
    emissive_color: vec4<f32>,
    transparency: vec4<f32>,
    properties: vec4<f32>,
    options: vec4<f32>,
    inspection: vec4<f32>,
    udim: vec4<f32>,
    udim_mask: vec4<u32>,
};


@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;
@group(1) @binding(6) var<uniform> material: MaterialUniform;
struct VertexInput { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32> };
struct InstanceInput {
    @location(5) model_0: vec4<f32>, @location(6) model_1: vec4<f32>,
    @location(7) model_2: vec4<f32>, @location(8) model_3: vec4<f32>,
};
struct ShadowVertex { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs_shadow(vertex: VertexInput, instance: InstanceInput) -> ShadowVertex {
    let model = mat4x4<f32>(instance.model_0, instance.model_1, instance.model_2, instance.model_3);
    var output: ShadowVertex;
    output.position = shadow_matrices[shadow_index.index.x] * model * vec4<f32>(vertex.position, 1.0);
    output.uv = vertex.uv;
    return output;
}
@fragment
fn fs_shadow(input: ShadowVertex, @builtin(front_facing) front: bool) {
    let udim_enabled = material.inspection.y > 0.5;
    let texture_scale = select(1.0, max(material.inspection.z, 0.01), material.inspection.z > 0.0);
    let scaled_uv = floor(input.uv) + fract(input.uv) * texture_scale;
    let tile = vec2<i32>(floor(scaled_uv));
    let local_tile = tile - vec2<i32>(i32(material.udim.x), i32(material.udim.y));
    let columns = i32(material.udim.z);
    let rows = i32(material.udim.w);
    var tile_assigned = false;
    if (local_tile.x >= 0 && local_tile.y >= 0 && local_tile.x < columns && local_tile.y < rows) {
        let slot = u32(local_tile.y * columns + local_tile.x);
        if (slot < 128u) {
            let word = slot / 32u;
            let bit = slot % 32u;
            tile_assigned = (material.udim_mask[word] & (1u << bit)) != 0u;
        }
    }
    let udim_texture_weight = select(1.0, select(0.0, 1.0, tile_assigned), udim_enabled);
    let texture_weight = step(0.5, material.options.z) * udim_texture_weight;
    let atlas_uv = vec2<f32>(
        (scaled_uv.x - material.udim.x) / max(material.udim.z, 1.0),
        (scaled_uv.y - material.udim.y) / max(material.udim.w, 1.0)
    );
    let texture_uv = select(scaled_uv, atlas_uv, udim_enabled);
    // Let screen-space derivatives select the mip level. Forcing level zero at
    // a distance skips most source texels and makes only scattered parts of a
    // texture survive; the trilinear/anisotropic sampler keeps the full
    // material stable as the projected model becomes smaller.
    let sampled_texture = textureSample(
        base_color_texture,
        base_color_sampler,
        texture_uv
    );
    let sampled_base = mix(vec4<f32>(1.0), sampled_texture, vec4<f32>(texture_weight));

    let alpha = material_opacity(sampled_base.a, material.base_color.a, material.transparency);
    if (!front && material.color_adjustments.w < 0.5) || !alpha_coverage(alpha, input.position.xy) { discard; }
}
