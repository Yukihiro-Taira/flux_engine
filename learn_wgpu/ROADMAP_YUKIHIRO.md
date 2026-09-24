# WGPU Roadmap

Goal: decode video frames, draw them on GPU, apply shaders, build playback controls.

> 🤖 Don't forget to use LLMs (codex or other) to help you on this. It will help on every step and to better understand how it works.

## Steps

### Step 1 - WGPU basics
- Setup: instance, adapter, device, queue, surface
- Draw a colored triangle/quad on screen
- Understand pipeline: vertex buffer, shader (WGSL), render pass

### Step 2 - Static image on GPU
- Load one image (not video yet) as a texture
- Upload texture to GPU, draw it on a full-screen quad
- Write a basic fragment shader (grayscale, brightness, crop) to confirm shader flow works

### Step 3 - Decode video frames with ffmpeg
- Use ffmpeg to decode a video file frame by frame (no GPU yet, just get raw frames in memory)
- Frames usually come out as YUV420, not RGB, note that for later
- Confirm you can extract frames at the right pace (matching video FPS)

### Step 4 - Frame to GPU, one video, no controls
- Take decoded frames from Step 3, upload each as a texture (reuse Step 2 code)
- Convert YUV to RGB, either in a shader (recommended, faster) or in CPU before upload
- Get continuous playback: decode thread feeds frames, render loop displays them
- No play/pause/seek yet, just play start to end

### Step 5 - Playback controls
- Add play/pause (stop feeding frames to renderer, or stop render loop)
- Add basic seek (jump ffmpeg decode position, resync)
- Keep decode and render on separate threads so UI never blocks (this is where W6 async/multithreading applies)

### Step 6 - Shaders on top of playback
- Apply real-time shader effects while video plays (color grading, filters, whatever the app needs)
- This becomes easy once Step 4-5 are solid, since frame-to-texture pipeline already works



## Resources
- [sotrh/learn-wgpu](https://github.com/sotrh/learn-wgpu) - main reference, official recommendation, triangle to textures/lighting/instancing
- [mpizenberg/wgpu-tutorial](https://github.com/mpizenberg/wgpu-tutorial) - simplest, no windowing, writes shader output straight to disk
- [jack1232/wgpu-step-by-step](https://github.com/jack1232/wgpu-step-by-step) - progressive with video episodes, up to textures and 3D transforms
- [gfx-rs/wgpu examples](https://github.com/gfx-rs/wgpu/tree/v30/examples) - official examples, useful for texture upload code
- [matthewjberger/wgpu-example](https://github.com/matthewjberger/wgpu-example) - minimal cross-platform (Win/Linux/Mac/Web/Android)
- [YouTube playlist](https://www.youtube.com/watch?v=i6WMfY-XTZE&list=PL_UrKDEhALdJS0VrLPn7dqC5A4W1vCAUT) - video series

