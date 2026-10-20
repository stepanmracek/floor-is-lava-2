use crate::{game::STEP_SIZE, player::Player};
use kiss3d::{glamx::vec3, prelude::*};
use rand::{RngExt, seq::IteratorRandom};
use std::{path::Path, time::Instant};

const LAVA_SPEED: f32 = 1.0 / 2.0;

fn camera() -> OrbitCamera3d {
    let mut camera = OrbitCamera3d::new(vec3(0.0, 3.0, -20.0), Vec3::ZERO);
    camera.rebind_rotate_button(None);
    camera.rebind_drag_button(None);
    camera.rebind_reset_key(None);
    camera.set_min_dist(camera.dist());
    camera.set_max_dist(camera.dist());
    camera
}

fn create_block_mesh() -> kiss3d::procedural::RenderMesh {
    let mut mesh = kiss3d::procedural::unit_cuboid();
    mesh.unify_index_buffer();

    let normals = mesh.normals.clone().unwrap();
    if let (coords, Some(uvs)) = (&mesh.coords.clone(), &mut mesh.uvs) {
        for (i, uv) in uvs.iter_mut().enumerate() {
            if normals[i].z < -0.5 {
                // Front face (-Z): map to top half [V: 0.0 .. 0.5]
                let u = if coords[i].x > 0.0 { 0.0 } else { 1.0 };
                let v = if coords[i].y > 0.0 { 0.0 } else { 0.5 };
                *uv = Vec2::new(u, v);
            } else {
                // All other 5 faces: sample the plain solid color in the lower half
                *uv = Vec2::new(0.5, 0.75);
            }
        }
    }

    mesh
}

#[derive(PartialEq)]
pub enum BlockColor {
    Neutral,
    Red,
    Blue,
}

impl std::fmt::Display for BlockColor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BlockColor::Neutral => write!(f, "gray"),
            BlockColor::Red => write!(f, "red"),
            BlockColor::Blue => write!(f, "blue"),
        }
    }
}

pub struct Block {
    pub cube: SceneNode3d,
    pub points: usize,
    pub color: BlockColor,
}

impl Block {
    pub fn set_color(&mut self, color: BlockColor) {
        self.color = color;
        let texture_name = format!("block-{}-{}", self.points, self.color);
        self.cube.set_texture_with_name(&texture_name);
    }
}

pub struct Scene {
    pub world: SceneNode3d,
    light: SceneNode3d,
    pub lava: SceneNode3d,
    pub camera: OrbitCamera3d,
    pub blocks: Vec<Block>,
    pub red_score: usize,
    pub blue_score: usize,
}

impl Scene {
    pub fn new() -> Self {
        MeshManager3d::get_global_manager(|mesh_mngr| {
            if mesh_mngr.get("block_cube").is_none() {
                mesh_mngr.add_render_mesh(create_block_mesh(), false, "block_cube");
            }
        });

        for p in 1..=9 {
            for color in [BlockColor::Neutral, BlockColor::Blue, BlockColor::Red] {
                let name = format!("block-{p}-{color}");
                let path = format!("assets/textures/{name}.png");
                TextureManager::get_global_manager(|texture_mngr| {
                    texture_mngr.add(Path::new(&path), &name);
                });
            }
        }

        let mut world = SceneNode3d::empty();
        let light = world
            .add_light(Light::point(100.0))
            .set_position(Vec3::new(5.0, 5.0, -10.0));

        let mut blocks = vec![];
        let z = 0.0;
        for x in (-6..=6).step_by(2) {
            Scene::spawn_block(&mut world, &mut blocks, x as f32, z);
        }

        let lava = world
            .add_cylinder_with_subdiv(12.0, 0.1, 32)
            .set_position(vec3(0.0, -5.0, -5.0))
            //.set_texture_from_file(Path::new("assets/textures/lava.png"), "lava")
            .set_color(DARK_RED.with_alpha(0.5));

        Self {
            world,
            light,
            lava,
            camera: camera(),
            blocks,
            red_score: 0,
            blue_score: 0,
        }
    }

    pub fn update(&mut self, clock: &Clock, players: &[&Player]) {
        let d = vec3(0.0, 1.0, 1.0) * (LAVA_SPEED * clock.dt);
        self.lava.set_position(self.lava.position() + d);
        self.light.set_position(self.light.position() + d);

        let player_pos: Vec3 = players.iter().map(|p| p.scene_node.position()).sum();
        let camera_at = self.camera.at().lerp(player_pos / 2.0, clock.dt);
        self.camera.look_at(self.camera.eye() + d, camera_at);

        let lava_z = self.lava.position().z;
        let mut max_z: Option<f32> = None;
        self.blocks.retain_mut(|block| {
            let pos = block.cube.position();
            if pos.z < lava_z {
                block.cube.remove();
                match block.color {
                    BlockColor::Blue => self.blue_score += block.points,
                    BlockColor::Red => self.red_score += block.points,
                    _ => {}
                }
                false
            } else {
                if max_z.is_none_or(|max_z| pos.z > max_z) {
                    max_z = Some(pos.z);
                };
                true
            }
        });

        if max_z.is_some_and(|max_z| max_z - lava_z < 5.0) {
            let new_z = max_z.unwrap() + STEP_SIZE;
            let mut rng = rand::rng();
            let guaranteed_x = (-6..=6).step_by(2).choose(&mut rng).unwrap();

            for x in (-6..=6).step_by(2) {
                if x == guaranteed_x || rng.random::<f32>() < 0.7 {
                    Scene::spawn_block(&mut self.world, &mut self.blocks, x as f32, new_z);
                }
            }
        }
    }

    pub fn is_standing_on_block(&mut self, target_xz: Vec2) -> Option<&mut Block> {
        self.blocks.iter_mut().find(|block| {
            let pos = block.cube.position();
            (pos.x - target_xz.x).abs() <= 0.05 && (pos.z - target_xz.y).abs() <= 0.05
        })
    }

    fn spawn_block(world: &mut SceneNode3d, blocks: &mut Vec<Block>, x: f32, z: f32) {
        let points = rand::rng().random_range(1..=9);
        let texture_name = format!("block-{points}-gray");
        let cube = world
            .add_geom_with_name("block_cube", Vec3::splat(1.5))
            .expect("block_cube mesh not registered")
            .set_position(vec3(x, z - 0.75, z))
            .set_texture_with_name(&texture_name);

        let block = Block {
            cube,
            points,
            color: BlockColor::Neutral,
        };
        blocks.push(block);
    }
}

pub struct Clock {
    start: Instant,
    last_time: Instant,
    pub dt: f32,
    pub game_time: f32,
}

impl Default for Clock {
    fn default() -> Self {
        let now = Instant::now();
        Clock {
            start: now,
            last_time: now,
            dt: 0.0,
            game_time: 0.0,
        }
    }
}

impl Clock {
    pub fn tick(&mut self) {
        let now = Instant::now();
        self.game_time = now.duration_since(self.start).as_secs_f32();
        self.dt = now.duration_since(self.last_time).as_secs_f32().min(0.05);
        self.last_time = now;
    }
}
