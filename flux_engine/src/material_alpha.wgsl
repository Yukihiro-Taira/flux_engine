// Shared by viewport, preview, and shadow shaders.
// Modes: automatic, opaque, cutout, alpha blend, dithered.
fn material_opacity(texture_alpha: f32, base_alpha: f32, settings: vec4<f32>) -> f32 {
    if settings.x == 1.0 { return 1.0; }
    let alpha = clamp(select(texture_alpha, 1.0, settings.z > 0.5) * base_alpha * settings.w, 0.0, 1.0);
    if settings.x == 2.0 {
        return select(0.0, 1.0, alpha > 0.0 && alpha >= settings.y);
    }
    return alpha;
}

fn alpha_coverage(alpha: f32, pixel: vec2<f32>) -> bool {
    // Stable ordered coverage, also used to approximate translucent shadows.
    let bayer = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0,
                              3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    let p = vec2<u32>(pixel) % vec2<u32>(4u);
    return alpha >= (bayer[p.y * 4u + p.x] + 0.5) / 16.0;
}
