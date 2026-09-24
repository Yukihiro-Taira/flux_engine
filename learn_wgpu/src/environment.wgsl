struct EnvironmentUniform {
    inverse_view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    settings: vec4<f32>,
};

@group(0) @binding(0) var environment_texture: texture_2d<f32>;
@group(0) @binding(1) var environment_sampler: sampler;
@group(0) @binding(2) var<uniform> environment: EnvironmentUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) clip_xy: vec2<f32>,
};

fn sample_environment_bilinear(uv: vec2<f32>) -> vec3<f32> {
    let dimensions = textureDimensions(environment_texture);
    let size = vec2<f32>(dimensions);
    let coordinate = uv * size - vec2<f32>(0.5);
    let base = vec2<i32>(floor(coordinate));
    let fraction = fract(coordinate);
    let width = i32(dimensions.x);
    let height = i32(dimensions.y);
    let x0 = ((base.x % width) + width) % width;
    let x1 = (x0 + 1) % width;
    let y0 = clamp(base.y, 0, height - 1);
    let y1 = clamp(base.y + 1, 0, height - 1);
    let top = mix(
        textureLoad(environment_texture, vec2<i32>(x0, y0), 0).rgb,
        textureLoad(environment_texture, vec2<i32>(x1, y0), 0).rgb,
        fraction.x
    );
    let bottom = mix(
        textureLoad(environment_texture, vec2<i32>(x0, y1), 0).rgb,
        textureLoad(environment_texture, vec2<i32>(x1, y1), 0).rgb,
        fraction.x
    );
    return mix(top, bottom, fraction.y);
}

fn aces_fitted(color: vec3<f32>) -> vec3<f32> {
    return clamp(
        color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14),
        vec3<f32>(0.0), vec3<f32>(1.0)
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    var output: VertexOutput;
    output.clip_xy = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    output.position = vec4<f32>(output.clip_xy, 0.0, 1.0);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var world = environment.inverse_view_projection * vec4<f32>(input.clip_xy, 1.0, 1.0);
    world /= world.w;
    let direction = normalize(world.xyz - environment.camera_position.xyz);

    let pi = 3.141592653589793;
    let longitude = atan2(direction.y, direction.x) + environment.settings.z;
    let latitude = asin(clamp(direction.z, -1.0, 1.0));
    let uv = vec2<f32>(fract(longitude / (2.0 * pi) + 0.5), 0.5 - latitude / pi);

    var color = sample_environment_bilinear(uv);
    color *= environment.settings.x * exp2(environment.settings.y);
    // The swap-chain's sRGB format performs the final linear-to-sRGB conversion.
    color = aces_fitted(color);
    return vec4<f32>(color, 1.0);
}
