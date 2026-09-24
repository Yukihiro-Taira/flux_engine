struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
};

struct MaterialUniform {
    base_color: vec4<f32>,
    color_adjustments: vec4<f32>,
    emissive_color: vec4<f32>,
    properties: vec4<f32>,
    options: vec4<f32>,
    inspection: vec4<f32>,
    udim: vec4<f32>,
    udim_mask: vec4<u32>,
};

@group(0) @binding(0) var base_color_texture: texture_2d<f32>;
@group(0) @binding(1) var base_color_sampler: sampler;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var normal_sampler: sampler;
@group(0) @binding(4) var metallic_roughness_texture: texture_2d<f32>;
@group(0) @binding(5) var metallic_roughness_sampler: sampler;
@group(0) @binding(7) var emissive_texture: texture_2d<f32>;
@group(0) @binding(8) var emissive_sampler: sampler;
@group(0) @binding(6) var<uniform> material: MaterialUniform;
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(0) var environment_texture: texture_2d<f32>;
@group(2) @binding(1) var environment_sampler: sampler;

struct LightingUniform {
    counts: vec4<u32>,
    default_settings: vec4<f32>,
    default_direction: vec4<f32>,
    default_color: vec4<f32>,
};

struct GpuLight {
    position_type: vec4<f32>,
    direction_range: vec4<f32>,
    color_power: vec4<f32>,
    shape: vec4<f32>,
    spot: vec4<f32>,
    contribution: vec4<f32>,
    shadow: vec4<f32>,
};

@group(3) @binding(0) var<uniform> lighting: LightingUniform;
@group(3) @binding(1) var<storage, read> scene_lights: array<GpuLight>;
@group(3) @binding(2) var shadow_maps: texture_depth_2d_array;
@group(3) @binding(3) var shadow_sampler: sampler_comparison;
@group(3) @binding(4) var<storage, read> shadow_matrices: array<mat4x4<f32>>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec4<f32>,
};

struct InstanceInput {
    @location(5) model_0: vec4<f32>,
    @location(6) model_1: vec4<f32>,
    @location(7) model_2: vec4<f32>,
    @location(8) model_3: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(instance.model_0, instance.model_1, instance.model_2, instance.model_3);
    let world = model * vec4<f32>(vertex.position, 1.0);
    let model_x = model[0].xyz;
    let model_y = model[1].xyz;
    let model_z = model[2].xyz;
    // The inverse-transpose of an orthogonal rotation/scale matrix. Dividing
    // each model column by its squared length keeps normals correct when the
    // model's X/Y/Z scale controls are not uniform.
    let normal = normalize(
        model_x * vertex.normal.x / max(dot(model_x, model_x), 0.00000001)
        + model_y * vertex.normal.y / max(dot(model_y, model_y), 0.00000001)
        + model_z * vertex.normal.z / max(dot(model_z, model_z), 0.00000001)
    );
    let transformed_tangent = mat3x3<f32>(model_x, model_y, model_z) * vertex.tangent.xyz;
    let tangent = normalize(transformed_tangent - normal * dot(normal, transformed_tangent));
    let orientation = select(-1.0, 1.0, dot(cross(model_x, model_y), model_z) >= 0.0);
    var output: VertexOutput;
    output.clip_position = camera.view_proj * world;
    output.uv = vertex.tex_coords;
    output.world_position = world.xyz;
    output.normal = normal;
    output.tangent = tangent;
    output.bitangent = normalize(cross(normal, tangent)) * vertex.tangent.w * orientation;
    return output;
}

fn distribution_ggx(normal: vec3<f32>, halfway: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(normal, halfway), 0.0);
    let denominator = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * denominator * denominator, 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0)
        * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn environment_brdf_approximation(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

fn aces_fitted(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn environment_uv(direction: vec3<f32>) -> vec2<f32> {
    let pi = 3.14159265;
    let longitude = atan2(direction.y, direction.x) + material.options.x;
    let latitude = asin(clamp(direction.z, -1.0, 1.0));
    return vec2<f32>(fract(longitude / (2.0 * pi) + 0.5), 0.5 - latitude / pi);
}

fn sample_environment(direction: vec3<f32>, mip_level: u32) -> vec3<f32> {
    // Rgba32Float is intentionally kept for HDR energy. It is not filterable
    // on every WebGPU adapter, so perform portable bilinear filtering manually.
    let level = min(mip_level, textureNumLevels(environment_texture) - 1u);
    let dimensions = textureDimensions(environment_texture, level);
    let size = vec2<f32>(dimensions);
    let coordinate = environment_uv(direction) * size - vec2<f32>(0.5);
    let base = vec2<i32>(floor(coordinate));
    let fraction = fract(coordinate);
    let width = i32(dimensions.x);
    let height = i32(dimensions.y);
    let x0 = ((base.x % width) + width) % width;
    let x1 = (x0 + 1) % width;
    let y0 = clamp(base.y, 0, height - 1);
    let y1 = clamp(base.y + 1, 0, height - 1);
    let top = mix(
        textureLoad(environment_texture, vec2<i32>(x0, y0), i32(level)).rgb,
        textureLoad(environment_texture, vec2<i32>(x1, y0), i32(level)).rgb,
        fraction.x
    );
    let bottom = mix(
        textureLoad(environment_texture, vec2<i32>(x0, y1), i32(level)).rgb,
        textureLoad(environment_texture, vec2<i32>(x1, y1), i32(level)).rgb,
        fraction.x
    );
    return mix(top, bottom, fraction.y);
}

fn tangent_to_world(sample_direction: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.z) > 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * sample_direction.x + bitangent * sample_direction.y + normal * sample_direction.z);
}

fn sample_ggx_half_vector(index: u32, count: u32, roughness: f32, normal: vec3<f32>) -> vec3<f32> {
    let xi_x = (f32(index) + 0.5) / f32(count);
    let xi_y = fract(f32(index) * 0.61803398875);
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let phi = 6.28318530 * xi_y;
    let cos_theta = sqrt((1.0 - xi_x) / max(1.0 + (alpha2 - 1.0) * xi_x, 0.0001));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return tangent_to_world(
        vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta),
        normal
    );
}

fn prefiltered_environment(normal: vec3<f32>, view: vec3<f32>, roughness: f32) -> vec3<f32> {
    let sample_count = clamp(lighting.counts.w, 1u, 128u);
    let roughness_level = u32(round(
        roughness * 0.65 * f32(textureNumLevels(environment_texture) - 1u)
    ));
    var lighting = vec3<f32>(0.0);
    var total_weight = 0.0;
    for (var index = 0u; index < 128u; index++) {
        if (index >= sample_count) { break; }
        let halfway = sample_ggx_half_vector(index, sample_count, roughness, normal);
        let light = normalize(reflect(-view, halfway));
        let weight = max(dot(normal, light), 0.0);
        if (weight > 0.0) {
            lighting += sample_environment(light, roughness_level) * weight;
            total_weight += weight;
        }
    }
    if (total_weight > 0.0001) {
        return lighting / total_weight;
    }

    // Grazing or degenerate sample sets must never turn the material black.
    // Fall back to the exact world-space mirror direction in that case.
    let reflection = normalize(reflect(-view, normal));
    return sample_environment(reflection, roughness_level);
}

fn diffuse_irradiance(normal: vec3<f32>) -> vec3<f32> {
    let sample_count = clamp(lighting.counts.w / 2u, 8u, 64u);
    var lighting = vec3<f32>(0.0);
    for (var index = 0u; index < 64u; index++) {
        if (index >= sample_count) { break; }
        let xi_x = (f32(index) + 0.5) / f32(sample_count);
        let xi_y = fract(f32(index) * 0.61803398875);
        let radius = sqrt(xi_x);
        let phi = 6.28318530 * xi_y;
        let direction = tangent_to_world(
            vec3<f32>(radius * cos(phi), radius * sin(phi), sqrt(1.0 - xi_x)),
            normal
        );
        lighting += sample_environment(
            direction,
            (textureNumLevels(environment_texture) - 1u) / 2u
        );
    }
    return lighting / f32(sample_count);
}

fn hash22(point: vec2<f32>) -> f32 {
    return fract(sin(dot(point, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn value_noise(point: vec2<f32>) -> f32 {
    let cell = floor(point);
    let local = fract(point);
    let blend = local * local * (3.0 - 2.0 * local);
    return mix(
        mix(hash22(cell), hash22(cell + vec2<f32>(1.0, 0.0)), blend.x),
        mix(hash22(cell + vec2<f32>(0.0, 1.0)), hash22(cell + vec2<f32>(1.0)), blend.x),
        blend.y
    );
}

fn uv_inspection_overlay(shaded: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    if (material.inspection.x < 0.5) {
        return shaded;
    }
    // A conventional DCC UV checker: alternating cells expose stretching and
    // orientation, while derivative-filtered teal lines expose island flow.
    let tiled = uv * 10.0;
    let cell = floor(tiled);
    let checker = abs((cell.x + cell.y) - 2.0 * floor((cell.x + cell.y) * 0.5));
    let checker_color = mix(vec3<f32>(0.10, 0.12, 0.14), vec3<f32>(0.58, 0.61, 0.64), checker);
    let within = fract(tiled);
    let edge_distance = min(min(within.x, 1.0 - within.x), min(within.y, 1.0 - within.y));
    let filter_width = max(fwidth(tiled.x), fwidth(tiled.y));
    let grid = 1.0 - smoothstep(0.0, filter_width * 1.35, edge_distance);
    let checker_over_shading = mix(shaded, checker_color, 0.72);
    return mix(checker_over_shading, vec3<f32>(0.0, 0.82, 0.74), grid * 0.9);
}

struct DirectLighting {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

fn point_shadow_face(direction: vec3<f32>) -> u32 {
    let absolute = abs(direction);
    if (absolute.x >= absolute.y && absolute.x >= absolute.z) {
        return select(1u, 0u, direction.x >= 0.0);
    }
    if (absolute.y >= absolute.z) {
        return select(3u, 2u, direction.y >= 0.0);
    }
    return select(5u, 4u, direction.z >= 0.0);
}

fn shadow_visibility(light: GpuLight, world_position: vec3<f32>, normal: vec3<f32>) -> f32 {
    if (light.shadow.x < 0.5 || light.shadow.z < 0.0) {
        return 1.0;
    }
    var layer = u32(light.shadow.z + 0.5);
    if (u32(light.shadow.w + 0.5) == 6u) {
        layer += point_shadow_face(world_position - light.position_type.xyz);
    }
    let clip = shadow_matrices[layer] * vec4<f32>(world_position, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let projected = clip.xyz / clip.w;
    let uv = vec2<f32>(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || projected.z < 0.0 || projected.z > 1.0) {
        return 1.0;
    }

    let to_light = normalize(select(
        light.position_type.xyz - world_position,
        -light.direction_range.xyz,
        u32(light.position_type.w + 0.5) == 2u
    ));
    let slope_bias = mix(0.0007, 0.003, 1.0 - max(dot(normal, to_light), 0.0));
    let radius = 1.0 + clamp(light.shadow.y, 0.0, 10.0) * 1.5;
    let texel = radius / 1024.0;
    var visibility = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            visibility += textureSampleCompare(
                shadow_maps,
                shadow_sampler,
                uv + vec2<f32>(f32(x), f32(y)) * texel,
                i32(layer),
                projected.z - slope_bias
            );
        }
    }
    return visibility / 9.0;
}

fn apply_shadow(result: DirectLighting, visibility: f32) -> DirectLighting {
    var shadowed = result;
    shadowed.diffuse *= visibility;
    shadowed.specular *= visibility;
    return shadowed;
}

fn evaluate_direct_light(
    light_direction: vec3<f32>,
    radiance: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    diffuse_scale: f32,
    specular_scale: f32,
) -> DirectLighting {
    var result: DirectLighting;
    let n_dot_l = max(dot(normal, light_direction), 0.0);
    let n_dot_v = max(dot(normal, view), 0.0);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        result.diffuse = vec3<f32>(0.0);
        result.specular = vec3<f32>(0.0);
        return result;
    }
    let halfway = normalize(view + light_direction);
    let f0 = mix(vec3<f32>(0.04), base_color, vec3<f32>(metallic));
    let fresnel = fresnel_schlick(max(dot(halfway, view), 0.0), f0);
    let distribution = distribution_ggx(normal, halfway, roughness);
    let geometry = geometry_schlick_ggx(n_dot_v, roughness)
        * geometry_schlick_ggx(n_dot_l, roughness);
    result.specular = radiance * distribution * geometry * fresnel
        / max(4.0 * n_dot_v * n_dot_l, 0.0001) * n_dot_l * specular_scale;
    result.diffuse = radiance * (vec3<f32>(1.0) - fresnel)
        * (1.0 - metallic) * base_color / 3.14159265 * n_dot_l * diffuse_scale;
    return result;
}

fn evaluate_gpu_light(
    light: GpuLight,
    world_position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> DirectLighting {
    let light_type = u32(light.position_type.w + 0.5);
    let emitter_normal = normalize(light.direction_range.xyz);
    let visibility = shadow_visibility(light, world_position, normal);
    if (light_type == 2u || light_type == 9u) {
        return apply_shadow(evaluate_direct_light(
            -emitter_normal, light.color_power.rgb * light.color_power.w,
            normal, view, base_color, metallic, roughness,
            light.contribution.x, light.contribution.y
        ), visibility);
    }

    if (light_type >= 3u && light_type <= 8u) {
        var result: DirectLighting;
        result.diffuse = vec3<f32>(0.0);
        result.specular = vec3<f32>(0.0);
        let sample_count = clamp(lighting.counts.z, 1u, 64u);
        let helper = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(emitter_normal.z) > 0.98);
        let emitter_x = normalize(cross(helper, emitter_normal));
        let emitter_y = normalize(cross(emitter_normal, emitter_x));
        var area = max(light.shape.y * light.shape.z, 0.000001);
        if (light_type == 4u) { area *= 0.78539816; }
        if (light_type == 5u) { area = max(light.shape.y * light.shape.x * 2.0, 0.000001); }
        if (light_type == 6u) { area = max(6.2831853 * light.shape.x * light.shape.y, 0.000001); }
        if (light_type == 7u) { area = 12.5663706 * light.shape.x * light.shape.x; }

        for (var index = 0u; index < 64u; index++) {
            if (index >= sample_count) { break; }
            let xi = vec2<f32>(
                (f32(index) + 0.5) / f32(sample_count),
                fract(f32(index) * 0.61803398875)
            );
            var sample_position = light.position_type.xyz;
            var sample_normal = emitter_normal;
            if (light_type == 3u || light_type == 8u) {
                sample_position += emitter_x * ((xi.x - 0.5) * light.shape.y)
                    + emitter_y * ((xi.y - 0.5) * light.shape.z);
            } else if (light_type == 4u) {
                let radius = sqrt(xi.x) * 0.5;
                let angle = 6.2831853 * xi.y;
                sample_position += emitter_x * (cos(angle) * radius * light.shape.y)
                    + emitter_y * (sin(angle) * radius * light.shape.z);
            } else if (light_type == 5u) {
                sample_position += emitter_x * ((xi.x - 0.5) * light.shape.y);
            } else if (light_type == 6u) {
                let angle = 6.2831853 * xi.y;
                sample_normal = normalize(emitter_y * cos(angle) + emitter_normal * sin(angle));
                sample_position += emitter_x * ((xi.x - 0.5) * light.shape.y)
                    + sample_normal * light.shape.x;
            } else if (light_type == 7u) {
                let z = 1.0 - 2.0 * xi.x;
                let radius = sqrt(max(1.0 - z * z, 0.0));
                let angle = 6.2831853 * xi.y;
                sample_normal = vec3<f32>(radius * cos(angle), radius * sin(angle), z);
                sample_position += sample_normal * light.shape.x;
            }

            let to_light = sample_position - world_position;
            let distance_squared = max(dot(to_light, to_light), 0.000001);
            let direction = to_light * inverseSqrt(distance_squared);
            var emitter_cosine = dot(sample_normal, -direction);
            emitter_cosine = select(abs(emitter_cosine), max(emitter_cosine, 0.0), light.spot.w > 0.5);
            let spread_power = mix(16.0, 1.0, light.shape.w);
            emitter_cosine = pow(max(emitter_cosine, 0.0), spread_power);
            let solid_angle = emitter_cosine * area / max(distance_squared + area, 0.000001);
            let sample_radiance = light.color_power.rgb * light.color_power.w * solid_angle;
            let sample_result = evaluate_direct_light(
                direction, sample_radiance, normal, view, base_color, metallic, roughness,
                light.contribution.x, light.contribution.y
            );
            result.diffuse += sample_result.diffuse / f32(sample_count);
            result.specular += sample_result.specular / f32(sample_count);
        }
        return apply_shadow(result, visibility);
    }

    let to_light = light.position_type.xyz - world_position;
    let distance_squared = max(dot(to_light, to_light), light.shape.x * light.shape.x);
    let direction = to_light * inverseSqrt(max(distance_squared, 0.000001));
    var attenuation = 1.0 / (12.5663706 * max(distance_squared, 0.000001));
    if (light_type == 1u) {
        let cone_cosine = dot(-direction, emitter_normal);
        attenuation *= pow(
            smoothstep(light.spot.y, light.spot.x, cone_cosine),
            light.spot.z
        );
    }
    return apply_shadow(evaluate_direct_light(
        direction, light.color_power.rgb * light.color_power.w * attenuation,
        normal, view, base_color, metallic, roughness,
        light.contribution.x, light.contribution.y
    ), visibility);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
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
    let noise_mix = select(1.0, mix(0.35, 1.0, value_noise(input.uv * material.options.w)), material.options.w > 0.0);
    let base_color = rotate_hue(sampled_base.rgb * material.base_color.rgb * noise_mix, material.color_adjustments.x);
    let mr = textureSample(
        metallic_roughness_texture,
        metallic_roughness_sampler,
        texture_uv
    ).rg;
    let metallic = clamp(mr.r * material.properties.x, 0.0, 1.0);
    let roughness = clamp(mr.g * material.properties.y, 0.045, 1.0);

    let tangent_normal = textureSample(normal_texture, normal_sampler, texture_uv).xyz * 2.0 - 1.0;
    let adjusted_normal = normalize(vec3<f32>(
        tangent_normal.xy * material.properties.z,
        tangent_normal.z
    ));
    let surface_normal = normalize(input.normal);
    let surface_tangent = normalize(
        input.tangent - surface_normal * dot(surface_normal, input.tangent)
    );
    let handedness = select(
        -1.0,
        1.0,
        dot(cross(surface_normal, surface_tangent), input.bitangent) >= 0.0
    );
    let surface_bitangent = normalize(cross(surface_normal, surface_tangent)) * handedness;
    let tbn = mat3x3<f32>(surface_tangent, surface_bitangent, surface_normal);
    let normal = normalize(tbn * adjusted_normal);
    // The observer vector must change with the camera for physically correct
    // Fresnel and reflection parallax. Environment directions remain in world
    // space, so orbiting never rotates the HDRI or any scene light.
    let view = normalize(camera.position.xyz - input.world_position);
    let n_dot_v = max(dot(normal, view), 0.0);

    // Principled metal/roughness model: dielectrics use an untinted 4% F0;
    // metals use Base Color as F0 and have no diffuse lobe.
    let f0 = mix(vec3<f32>(0.04), base_color, vec3<f32>(metallic));
    var direct = vec3<f32>(0.0);
    if (lighting.counts.x == 1u) {
        let work = evaluate_direct_light(
            normalize(-lighting.default_direction.xyz),
            lighting.default_color.rgb * lighting.default_color.w,
            normal, view, base_color, metallic, roughness,
            lighting.default_settings.y, lighting.default_settings.z
        );
        direct = work.diffuse + work.specular
            + base_color * lighting.default_settings.x * (1.0 - metallic);
    } else {
        for (var light_index = 0u; light_index < lighting.counts.y; light_index++) {
            let contribution = evaluate_gpu_light(
                scene_lights[light_index], input.world_position, normal, view,
                base_color, metallic, roughness
            );
            direct += contribution.diffuse + contribution.specular;
        }
    }

    let ibl_fresnel = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let diffuse_weight = (vec3<f32>(1.0) - ibl_fresnel) * (1.0 - metallic);
    let environment_diffuse = diffuse_irradiance(normal) * base_color;
    let environment_specular = prefiltered_environment(normal, view, roughness);
    let environment_brdf = environment_brdf_approximation(n_dot_v, roughness);
    var ambient = vec3<f32>(0.0);
    if (lighting.counts.x == 2u && lighting.default_settings.w > 0.5) {
        ambient = (diffuse_weight * environment_diffuse
            + environment_specular * (f0 * environment_brdf.x + environment_brdf.y))
            * material.properties.w;
    }

    let emission_map = textureSample(emissive_texture, emissive_sampler, texture_uv).rgb;
    let emission_source = select(base_color, emission_map, material.color_adjustments.z > 0.5 && texture_weight > 0.0);
    let emission = rotate_hue(emission_source * material.emissive_color.rgb, material.color_adjustments.y) * material.options.y;
    var color = direct + ambient + emission;
    color = aces_fitted(color);
    return vec4<f32>(uv_inspection_overlay(color, input.uv), sampled_base.a * material.base_color.a);
}

fn rotate_hue(color: vec3<f32>, degrees: f32) -> vec3<f32> {
    let angle = degrees * 0.01745329252;
    let axis = vec3<f32>(0.57735026919);
    return max(color * cos(angle) + cross(axis, color) * sin(angle)
        + axis * dot(axis, color) * (1.0 - cos(angle)), vec3<f32>(0.0));
}
