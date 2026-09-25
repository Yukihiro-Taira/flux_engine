struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
};

struct PointUniform {
    color: vec4<f32>,
    settings: vec4<f32>, // size in pixels, viewport width, viewport height, unused
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> points: PointUniform;

struct VertexInput {
    @location(0) center: vec3<f32>,
    @location(1) corner: vec2<f32>,
};

struct InstanceInput {
    @location(5) model_0: vec4<f32>,
    @location(6) model_1: vec4<f32>,
    @location(7) model_2: vec4<f32>,
    @location(8) model_3: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) corner: vec2<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(instance.model_0, instance.model_1, instance.model_2, instance.model_3);
    var clip = camera.view_proj * model * vec4<f32>(vertex.center, 1.0);
    let pixel_ndc = vec2<f32>(2.0 / points.settings.y, 2.0 / points.settings.z);
    let offset = vertex.corner * points.settings.x * 0.5 * pixel_ndc * clip.w;
    clip = vec4<f32>(clip.xy + offset, clip.zw);
    var output: VertexOutput;
    output.position = clip;
    output.corner = vertex.corner;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if (dot(input.corner, input.corner) > 1.0) {
        discard;
    }
    return points.color;
}
