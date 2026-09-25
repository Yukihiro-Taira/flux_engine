struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
};

struct NormalUniform {
    color: vec4<f32>,
    settings: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> marker: NormalUniform;

struct VertexInput {
    @location(0) origin: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) shape: vec2<f32>,
};

struct InstanceInput {
    @location(5) model_0: vec4<f32>,
    @location(6) model_1: vec4<f32>,
    @location(7) model_2: vec4<f32>,
    @location(8) model_3: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(instance.model_0, instance.model_1, instance.model_2, instance.model_3);
    let model_x = model[0].xyz;
    let model_y = model[1].xyz;
    let model_z = model[2].xyz;
    let world_normal = normalize(
        model_x * vertex.normal.x / max(dot(model_x, model_x), 0.00000001)
        + model_y * vertex.normal.y / max(dot(model_y, model_y), 0.00000001)
        + model_z * vertex.normal.z / max(dot(model_z, model_z), 0.00000001)
    );
    let world_origin = (model * vec4<f32>(vertex.origin, 1.0)).xyz;
    let start_clip = camera.view_proj * vec4<f32>(world_origin, 1.0);
    let end_clip = camera.view_proj * vec4<f32>(world_origin + world_normal * marker.settings.x, 1.0);
    var clip = mix(start_clip, end_clip, vertex.shape.x);

    let start_ndc = start_clip.xy / max(abs(start_clip.w), 0.000001);
    let end_ndc = end_clip.xy / max(abs(end_clip.w), 0.000001);
    let direction = end_ndc - start_ndc;
    var perpendicular = vec2<f32>(1.0, 0.0);
    if (dot(direction, direction) > 0.00000001) {
        perpendicular = normalize(vec2<f32>(-direction.y, direction.x));
    }
    let pixel_ndc = vec2<f32>(2.0 / marker.settings.z, 2.0 / marker.settings.w);
    let offset = perpendicular * vertex.shape.y * marker.settings.y * pixel_ndc * clip.w;
    clip = vec4<f32>(clip.xy + offset, clip.zw);

    var output: VertexOutput;
    output.position = clip;
    return output;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return marker.color;
}
