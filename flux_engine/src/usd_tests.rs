use super::*;
#[test]
fn extensions() {
    for ext in ["usd", "usda", "usdc", "usdz", "USDZ", "UsDc"] {
        let path = PathBuf::from(format!("asset.{ext}"));
        assert!(is_usd(&path));
        assert!(supported_model(&path));
    }
    assert!(!is_usd(Path::new("asset.obj")));
}
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "flux-usd-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
    fn write(&self, name: &str, source: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, source).unwrap();
        path
    }
    fn scene(&self, source: &str, time: Option<f64>) -> Result<UsdScene> {
        let path = self.write("scene.usda", source);
        evaluate(&open(&path)?, &self.0.join("cache"), time, false, true)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
const MESH: &str = r#"def Mesh "Mesh" {
    point3f[] points = [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0,1,2,3]
    normal3f[] normals = [(0,0,1)] (interpolation = "constant")
    uniform token subdivisionScheme = "none"
}"#;
fn source(body: &str) -> String {
    format!("#usda 1.0\n( metersPerUnit = 1 )\n{body}\n")
}
fn min_x(scene: &UsdScene) -> f32 {
    scene.meshes[0]
        .positions
        .iter()
        .step_by(3)
        .copied()
        .fold(f32::INFINITY, f32::min)
}
#[test]
fn native_formats_and_source_preservation() -> Result<()> {
    let f = Fixture::new();
    let text = source(MESH);
    let path = f.write("text.usda", &text);
    let stage = open(&path)?;
    for ext in ["usda", "usdc", "usd", "usdz"] {
        let file = f.0.join(format!("roundtrip.{ext}"));
        stage.root_layer().export(file.to_str().unwrap())?;
        let scene = evaluate(&open(&file)?, &f.0.join("cache"), None, false, true)?;
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.meshes[0].indices.len(), 6);
    }
    assert_eq!(std::fs::read_to_string(path)?, text);
    Ok(())
}
#[test]
fn units_transforms_uvs_normals_and_handedness() -> Result<()> {
    let f = Fixture::new();
    let mesh = MESH.replace(
        "uniform token subdivisionScheme",
        r#"texCoord2f[] primvars:st = [(0,0),(1,0),(1,1),(0,1)] (interpolation = "faceVarying")
    int[] primvars:st:indices = [3,2,1,0]
    uniform token subdivisionScheme"#,
    );
    let text = format!(
        "#usda 1.0\n( metersPerUnit = 0.01\n upAxis = \"Z\" )\ndef Xform \"World\" {{\n double3 xformOp:translate = (100,0,0)\n uniform token[] xformOpOrder = [\"xformOp:translate\"]\n {mesh}\n}}"
    );
    let scene = f.scene(&text, None)?;
    assert!((min_x(&scene) - 1.).abs() < 1e-6);
    assert_eq!(&scene.meshes[0].normals[..3], &[0., 1., 0.]);
    assert_eq!(scene.meshes[0].texcoords.len(), 12);
    let mirrored = MESH.replace(
        "uniform token subdivisionScheme",
        r#"uniform token orientation = "leftHanded"
    float3 xformOp:scale = (-1,1,1)
    uniform token[] xformOpOrder = ["xformOp:scale"]
    uniform token subdivisionScheme"#,
    );
    assert_eq!(min_x(&f.scene(&source(&mirrored), None)?), -1.);
    Ok(())
}
#[test]
fn composition_instances_and_visibility() -> Result<()> {
    let f = Fixture::new();
    f.write(
        "asset.usda",
        &format!("#usda 1.0\n(defaultPrim = \"Asset\")\ndef Xform \"Asset\" {{\n{MESH}\n}}"),
    );
    let scene = f.scene(
        &source(
            r#"def Xform "A" (references = @asset.usda@; instanceable = true) {}
    def Xform "B" (references = @asset.usda@; instanceable = true) {
        double3 xformOp:translate = (4,0,0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Xform "Hidden" {
        token visibility = "invisible"
        def Cube "Cube" {}
    }"#,
        ),
        None,
    )?;
    assert_eq!(scene.meshes.len(), 2);
    assert_eq!(scene.meshes[0].name, "/A/Mesh");
    assert_eq!(
        scene.meshes[1]
            .positions
            .iter()
            .step_by(3)
            .copied()
            .fold(f32::INFINITY, f32::min),
        4.
    );
    Ok(())
}
#[test]
fn primitives_and_nested_instancers() -> Result<()> {
    let f = Fixture::new();
    let scene = f.scene(
        &source(
            r#"def PointInstancer "Outer" {
        rel prototypes = </Outer/Inner>
        int[] protoIndices = [0,0]
        point3f[] positions = [(0,0,0),(10,0,0)]
        def PointInstancer "Inner" {
            rel prototypes = </Outer/Inner/Cube>
            int[] protoIndices = [0,0,0]
            point3f[] positions = [(0,0,0),(3,0,0),(20,0,0)]
            int64[] ids = [10,20,30]
            int64[] invisibleIds = [30]
            def Cube "Cube" {}
        }
    }
    def Sphere "Sphere" {}
    def Cone "Cone" {}
    def Cylinder "Cylinder" {}
    def Capsule "Capsule" {}
    def Plane "Plane" {}"#,
        ),
        None,
    )?;
    assert_eq!(scene.meshes.len(), 9);
    let mut xs: Vec<_> = scene
        .meshes
        .iter()
        .filter(|m| m.name.contains("instance_"))
        .map(|m| {
            m.positions
                .iter()
                .step_by(3)
                .copied()
                .fold(f32::INFINITY, f32::min) as i32
        })
        .collect();
    xs.sort();
    assert_eq!(xs, [-1, 2, 9, 12]);
    Ok(())
}
#[test]
fn cameras_and_lights() -> Result<()> {
    let f = Fixture::new();
    let scene = f.scene(
        r#"#usda 1.0
    (metersPerUnit = 0.01
    upAxis = "Z")
    def Camera "View" {
        double3 xformOp:translate = (100,200,300)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float focalLength = 50
        float verticalAperture = 24
    }
    def RectLight "Key" {
        float inputs:width = 200
        float inputs:height = 100
        float inputs:intensity = 12
        float inputs:exposure = 2
    }"#,
        None,
    )?;
    assert_eq!(scene.scene.cameras[0].eye, [1., 2., 3.]);
    assert_eq!(scene.scene.cameras[0].target, [1., 2., 2.]);
    assert_eq!(scene.scene.lights[0].size, [2., 1.]);
    assert_eq!(scene.scene.lights[0].intensity, 12.);
    Ok(())
}
#[test]
fn timeline_scrubs_both_directions_and_zero_scale() -> Result<()> {
    let f = Fixture::new();
    let text = source(&MESH.replace(
        "uniform token subdivisionScheme",
        r#"float3 xformOp:scale.timeSamples = {0:(0,0,0), 120:(1,1,1)}
        uniform token[] xformOpOrder = ["xformOp:scale"]
        uniform token subdivisionScheme"#,
    ));
    let path = f.write("anim.usda", &text);
    let mut session = AnimationSession::new(&path)?;
    assert!(session.transform_only);
    for time in [0., 120., 60., 0.] {
        let sample = session.sample(time)?;
        let transforms = match sample {
            AnimationSample::Geometry(scene, Some(t)) => {
                assert_eq!(
                    scene.meshes[0].positions.iter().copied().fold(0., f32::max),
                    1.
                );
                t
            }
            AnimationSample::Transforms(t) => t,
            _ => panic!("Expected transform sampling"),
        };
        assert!((transforms["/Mesh"][0][0] - time as f32 / 120.).abs() < 1e-6);
    }
    assert_eq!(std::fs::read_to_string(path)?, text);
    assert!(session.sample(f64::NAN).is_err());
    Ok(())
}
#[test]
fn animated_points_and_visibility_use_geometry() -> Result<()> {
    let f = Fixture::new();
    let text=source(&MESH.replace("point3f[] points = [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]",r#"point3f[] points.timeSamples = {0:[(0,0,0),(1,0,0),(1,1,0),(0,1,0)], 1:[(0,0,0),(2,0,0),(1,1,0),(0,1,0)]}
        token visibility.timeSamples = {0:"inherited", 2:"invisible"}"#));
    let path = f.write("deform.usda", &text);
    let mut session = AnimationSession::new(&path)?;
    assert!(!session.transform_only);
    match session.sample(1.)? {
        AnimationSample::Geometry(s, None) => assert_eq!(
            s.meshes[0]
                .positions
                .iter()
                .step_by(3)
                .copied()
                .fold(0., f32::max),
            2.
        ),
        _ => panic!("Expected geometry"),
    }
    match session.sample(2.)? {
        AnimationSample::Geometry(s, None) => assert!(s.meshes.is_empty()),
        _ => panic!("Expected geometry"),
    };
    Ok(())
}
#[test]
fn invalid_topology_is_an_error() -> Result<()> {
    let f = Fixture::new();
    let result = f.scene(&source(&MESH.replace("[0,1,2,3]", "[0,1,2,99]")), None);
    assert!(result.err().unwrap().to_string().contains("topology"));
    assert!(f.scene(&source(MESH), Some(f64::INFINITY)).is_err());
    Ok(())
}
#[test]
fn concave_polygons_holes_and_subsets() -> Result<()> {
    let f = Fixture::new();
    let scene = f.scene(
        &source(
            r#"def Mesh "Shape" {
        point3f[] points = [(0,0,0),(2,0,0),(2,2,0),(1,1,0),(0,2,0)]
        int[] faceVertexCounts = [5,3,3]
        int[] faceVertexIndices = [0,1,2,3,4,0,3,4,0,1,3]
        int[] holeIndices = [2]
        def GeomSubset "Accent" {
            uniform token elementType = "face"
            int[] indices = [1]
        }
    }"#,
        ),
        None,
    )?;
    let mut lengths: Vec<_> = scene.meshes.iter().map(|m| m.indices.len()).collect();
    lengths.sort();
    assert_eq!(lengths, [3, 9]);
    Ok(())
}
fn material_scene() -> String {
    source(&format!(
        r#"{}
def Material "Mat" {{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>
    def Shader "Surface" {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Mat/Tex.outputs:rgb>
        normal3f inputs:normal.connect = </Mat/Normal.outputs:rgb>
        float inputs:metallic.connect = </Mat/Tex.outputs:b>
        float inputs:roughness.connect = </Mat/Tex.outputs:g>
        float inputs:opacity = 0.6
        float inputs:opacityThreshold = 0.4
    }}
    def Shader "Tex" {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @channels.png@
        token inputs:sourceColorSpace = "raw"
    }}
    def Shader "Normal" {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @channels.png@
        float4 inputs:scale = (2,2,2,1)
        float4 inputs:bias = (-1,-1,-1,0)
    }}
}}"#,
        MESH.replace(
            "uniform token subdivisionScheme",
            "rel material:binding = </Mat>\n    uniform token subdivisionScheme"
        )
    ))
}
#[test]
fn material_channels_and_packaged_textures() -> Result<()> {
    let f = Fixture::new();
    let image = image::RgbaImage::from_pixel(2, 2, image::Rgba([64, 128, 240, 255]));
    let texture = f.0.join("channels.png");
    image.save(&texture)?;
    let text = material_scene();
    let path = f.write("material.usda", &text);
    let package = f.0.join("material.usdz");
    let mut archive = openusd::usdz::ArchiveWriter::create(&package)?;
    archive.add_layer("material.usda", text.as_bytes())?;
    archive.add_layer("channels.png", &std::fs::read(&texture)?)?;
    archive.finish()?;
    for path in [&path, &package] {
        let scene = evaluate(&open(path)?, &f.0.join("cache"), None, false, true)?;
        assert!(scene.warnings.is_empty(), "{:?}", scene.warnings);
        let material = &scene.materials[0];
        assert_eq!(material.alpha_cutoff, Some(0.4));
        assert_eq!(material.pbr.as_ref().unwrap().opacity, 0.6);
        assert_eq!(
            image::open(&material.roughness_texture)?
                .to_rgba8()
                .get_pixel(0, 0)
                .0,
            [240, 128, 0, 255]
        );
        assert_eq!(
            image::open(&material.normal_texture)?
                .to_rgba8()
                .get_pixel(0, 0)
                .0,
            [64, 128, 240, 255]
        );
        assert!(Path::new(&material.diffuse_texture).is_file());
    }
    std::fs::remove_file(&texture)?;
    let scene = evaluate(&open(&package)?, &f.0.join("cache"), None, false, true)?;
    assert!(scene.warnings.is_empty(), "{:?}", scene.warnings);
    let scene = evaluate(&open(&path)?, &f.0.join("cache"), None, false, true)?;
    assert!(scene.warnings.iter().any(|w| w.contains("texture")));
    Ok(())
}
#[test]
fn skeletal_deformation_is_evaluated_in_rust() -> Result<()> {
    let f = Fixture::new();
    let mesh = MESH
        .replace(
            "uniform token subdivisionScheme",
            r#"rel skel:skeleton = </Rig/Skeleton>
        matrix4d primvars:skel:geomBindTransform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        int[] primvars:skel:jointIndices = [0,0,0,0] (elementSize = 1; interpolation = "vertex")
        float[] primvars:skel:jointWeights = [1,1,1,1] (elementSize = 1; interpolation = "vertex")
        uniform token subdivisionScheme"#,
        )
        .replace(
            "def Mesh \"Mesh\" {",
            "def Mesh \"Mesh\" (prepend apiSchemas = [\"SkelBindingAPI\"]) {",
        );
    let text = source(&format!(
        r#"def SkelRoot "Rig" {{
    {mesh}
    def Skeleton "Skeleton" (prepend apiSchemas = ["SkelBindingAPI"]) {{
        uniform token[] joints = ["root"]
        uniform matrix4d[] bindTransforms = [((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))]
        uniform matrix4d[] restTransforms = [((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))]
        rel skel:animationSource = </Rig/Animation>
    }}
    def SkelAnimation "Animation" {{
        uniform token[] joints = ["root"]
        float3[] translations.timeSamples = {{2:[(2,0,0)]}}
        quatf[] rotations.timeSamples = {{2:[(1,0,0,0)]}}
        half3[] scales.timeSamples = {{2:[(1,1,1)]}}
    }}
}}"#
    ));
    let scene = f.scene(&text, Some(2.))?;
    assert_eq!(min_x(&scene), 2.);
    Ok(())
}
#[test]
fn gpu_transform_samples_match_baked_vertices() -> Result<()> {
    for up in ["Y", "Z"] {
        let f = Fixture::new();
        let mesh = MESH.replace(
            "uniform token subdivisionScheme",
            r#"double3 xformOp:translate = (3,4,5)
        float3 xformOp:rotateXYZ.timeSamples = {0:(0,0,0),120:(30,60,90)}
        float3 xformOp:scale = (2,1,3)
        uniform token[] xformOpOrder = ["xformOp:translate","xformOp:rotateXYZ","xformOp:scale"]
        uniform token subdivisionScheme"#,
        );
        let text = format!("#usda 1.0\n(metersPerUnit = 0.01\nupAxis = \"{up}\")\n{mesh}");
        let path = f.write("transforms.usda", &text);
        let mut session = AnimationSession::new(&path)?;
        let local = evaluate(&session.stage, &f.0.join("cache"), Some(0.), true, true)?;
        for time in [0., 60., 120.] {
            let sample = session.sample(time)?;
            let t = match sample {
                AnimationSample::Geometry(_, Some(t)) | AnimationSample::Transforms(t) => t,
                _ => panic!(),
            };
            let matrix = gf::Matrix4d(std::array::from_fn(|i| t["/Mesh"][i / 4][i % 4] as f64));
            let baked = evaluate(&session.stage, &f.0.join("cache"), Some(time), false, true)?;
            for (local, baked) in local.meshes[0]
                .positions
                .chunks_exact(3)
                .zip(baked.meshes[0].positions.chunks_exact(3))
            {
                let actual = matrix.transform_point([local[0], -local[2], local[1]].into());
                let expected = [baked[0], -baked[2], baked[1]];
                for (a, b) in <[f32; 3]>::from(actual).into_iter().zip(expected) {
                    assert!((a - b).abs() < 1e-5, "{up} time {time}: {a} vs {b}");
                }
            }
        }
    }
    Ok(())
}
