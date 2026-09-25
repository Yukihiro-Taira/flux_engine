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

@group(0) @binding(0) var base_texture: texture_2d<f32>;
@group(0) @binding(1) var base_sampler: sampler;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var normal_sampler: sampler;
@group(0) @binding(4) var mr_texture: texture_2d<f32>;
@group(0) @binding(5) var mr_sampler: sampler;
@group(0) @binding(7) var emissive_texture: texture_2d<f32>;
@group(0) @binding(8) var emissive_sampler: sampler;
@group(0) @binding(6) var<uniform> material: MaterialUniform;
@group(1) @binding(0) var environment_texture: texture_2d<f32>;
@group(1) @binding(1) var environment_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var output: VertexOutput;
    output.uv = vec2<f32>(x, y);
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return output;
}

fn environment_uv(direction: vec3<f32>) -> vec2<f32> {
    let pi = 3.14159265;
    return vec2<f32>(
        fract((atan2(direction.y, direction.x) + material.options.x) / (2.0 * pi) + 0.5),
        0.5 - asin(clamp(direction.z, -1.0, 1.0)) / pi
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let point = input.uv * 2.0 - 1.0;
    let radius_squared = dot(point, point);
    let checker_cell = vec2<u32>(input.position.xy / 12.0);
    let checker = vec3<f32>(select(0.10, 0.19, (checker_cell.x + checker_cell.y) % 2u == 0u));
    if radius_squared > 0.82 { return vec4<f32>(checker, 1.0); }
    let geometric_normal = normalize(vec3<f32>(point.x, -point.y, sqrt(0.82 - radius_squared)));
    let view = vec3<f32>(0.0, 0.0, 1.0);
    let pi = 3.14159265;
    let sphere_uv = vec2<f32>(
        atan2(geometric_normal.y, geometric_normal.x) / (2.0 * pi) + 0.5,
        0.5 - asin(geometric_normal.z) / pi
    );
    let sample = textureSample(base_texture, base_sampler, sphere_uv);
    let sampled_base = mix(vec4<f32>(1.0), sample, vec4<f32>(step(0.5, material.options.z)));
    let base = rotate_hue(sampled_base.rgb * material.base_color.rgb, material.color_adjustments.x);
    let emission_map = textureSample(emissive_texture, emissive_sampler, sphere_uv).rgb;
    let emission_source = select(base, emission_map, material.color_adjustments.z > 0.5);
    let emission = rotate_hue(emission_source * material.emissive_color.rgb, material.color_adjustments.y) * material.options.y;
    let sampled_normal = textureSample(normal_texture, normal_sampler, sphere_uv).xyz * 2.0 - 1.0;
    let tangent = normalize(vec3<f32>(-geometric_normal.y, geometric_normal.x, 0.0001));
    let bitangent = normalize(cross(geometric_normal, tangent));
    let normal = normalize(
        tangent * sampled_normal.x * material.properties.z
        + bitangent * sampled_normal.y * material.properties.z
        + geometric_normal * sampled_normal.z
    );
    let reflection = reflect(-view, normal);
    let mr = textureSample(mr_texture, mr_sampler, sphere_uv).rg;
    let metallic = clamp(mr.r * material.properties.x, 0.0, 1.0);
    let roughness = clamp(mr.g * material.properties.y, 0.045, 1.0);
    let environment = textureSampleLevel(
        environment_texture, environment_sampler, environment_uv(reflection), 0.0
    ).rgb * material.properties.w;
    let diffuse = base * (1.0 - metallic) * max(dot(normal, normalize(vec3<f32>(-0.5, 0.4, 1.0))), 0.0);
    let specular_color = mix(vec3<f32>(0.04), base, vec3<f32>(metallic));
    var color = diffuse * 2.0
        + environment * specular_color * (1.0 - roughness * 0.75)
        + emission;
    color = color / (color + vec3<f32>(1.0));
    var alpha = material_opacity(sampled_base.a, material.base_color.a, material.transparency);
    if material.transparency.x == 4.0 {
        alpha = select(0.0, 1.0, alpha_coverage(alpha, input.position.xy));
    }
    return vec4<f32>(mix(checker, color, alpha), 1.0);
}

fn rotate_hue(color: vec3<f32>, degrees: f32) -> vec3<f32> {
    let angle = degrees * 0.01745329252;
    let axis = vec3<f32>(0.57735026919);
    return max(color * cos(angle) + cross(axis, color) * sin(angle)
        + axis * dot(axis, color) * (1.0 - cos(angle)), vec3<f32>(0.0));
}
