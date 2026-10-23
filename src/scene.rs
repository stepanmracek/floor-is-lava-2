use crate::{
    game::STEP_SIZE,
    player::{Player, PlayerColor},
};
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

#[derive(PartialEq, Clone, Copy)]
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
    pub position: Vec3,
    pub points: usize,
    pub color: BlockColor,
    cube: Option<SceneNode3d>,
}

impl Block {
    pub fn set_color(&mut self, color: BlockColor) {
        self.color = color;
        let texture_name = format!("block-{}-{}", self.points, self.color);
        if let Some(cube) = &mut self.cube {
            cube.set_texture_with_name(&texture_name);
        }
    }
}

pub struct Lava {
    pub position: Vec3,
    pub node: Option<SceneNode3d>,
}

#[derive(Clone)]
pub struct BlockInfo {
    pub color: BlockColor,
    pub points: usize,
    pub above_lava: f32,
    pub red_player: bool,
    pub blue_player: bool,
}

impl BlockInfo {
    pub fn has_player(&self, color: PlayerColor) -> bool {
        match color {
            PlayerColor::Blue => self.blue_player,
            PlayerColor::Red => self.red_player,
        }
    }
}

pub struct Scene {
    pub world: Option<SceneNode3d>,
    pub lava: Lava,
    pub camera: Option<OrbitCamera3d>,
    pub red_score: usize,
    pub blue_score: usize,
    blocks: Vec<Block>,
    light: Option<SceneNode3d>,
}

impl Scene {
    pub fn headless() -> Self {
        let lava_pos = vec3(0.0, -5.0, -5.0);
        let mut blocks = vec![];
        let z = 0.0;
        for x in (-6..=6).step_by(2) {
            Scene::spawn_block(None, &mut blocks, x as f32, z);
        }

        let lava = Lava {
            position: lava_pos,
            node: None,
        };

        Self {
            world: None,
            light: None,
            lava,
            camera: None,
            blocks,
            red_score: 0,
            blue_score: 0,
        }
    }

    pub fn headed() -> Self {
        let lava_pos = vec3(0.0, -5.0, -5.0);

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
            Scene::spawn_block(Some(&mut world), &mut blocks, x as f32, z);
        }

        let lava_node = world
            .add_cylinder_with_subdiv(12.0, 0.1, 32)
            .set_position(lava_pos)
            //.set_texture_from_file(Path::new("assets/textures/lava.png"), "lava")
            .set_color(DARK_RED.with_alpha(0.5));

        let lava = Lava {
            position: lava_pos,
            node: Some(lava_node),
        };

        Self {
            world: Some(world),
            light: Some(light),
            lava,
            camera: Some(camera()),
            blocks,
            red_score: 0,
            blue_score: 0,
        }
    }

    pub fn update(&mut self, clock: &Clock, players: &[&Player]) {
        let d = vec3(0.0, 1.0, 1.0) * (LAVA_SPEED * clock.dt);
        self.lava.position += d;
        if let Some(node) = &mut self.lava.node {
            node.set_position(self.lava.position);
        }
        if let Some(light) = &mut self.light {
            light.set_position(light.position() + d);
        }

        let player_pos: Vec3 = players.iter().map(|p| p.position).sum();
        if let Some(camera) = &mut self.camera {
            let camera_at = camera.at().lerp(player_pos / 2.0, clock.dt);
            camera.look_at(camera.eye() + d, camera_at);
        }

        let lava_z = self.lava.position.z;
        let mut max_z: Option<f32> = None;
        self.blocks.retain_mut(|block| {
            let pos = block.position;
            if pos.z < lava_z {
                if let Some(cube) = &mut block.cube {
                    cube.remove();
                }
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
                    Scene::spawn_block(self.world.as_mut(), &mut self.blocks, x as f32, new_z);
                }
            }
        }
    }

    pub fn is_standing_on_block(&mut self, target_xz: Vec2) -> Option<&mut Block> {
        self.blocks.iter_mut().find(|block| {
            let pos = block.position;
            (pos.x - target_xz.x).abs() <= 0.05 && (pos.z - target_xz.y).abs() <= 0.05
        })
    }

    fn spawn_block(world: Option<&mut SceneNode3d>, blocks: &mut Vec<Block>, x: f32, z: f32) {
        let points = rand::rng().random_range(1..=9);
        let position = vec3(x, z - 0.75, z);
        let cube = world.map(|w| {
            let texture_name = format!("block-{points}-gray");
            w.add_geom_with_name("block_cube", Vec3::splat(1.5))
                .expect("block_cube mesh not registered")
                .set_position(position)
                .set_texture_with_name(&texture_name)
        });

        let block = Block {
            position,
            cube,
            points,
            color: BlockColor::Neutral,
        };
        blocks.push(block);
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn blocks_matrix(
        &self,
        blue_player: &Player,
        red_player: &Player,
    ) -> Vec<Vec<Option<BlockInfo>>> {
        if self.blocks.is_empty() {
            return Vec::new();
        }

        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        let mut min_x = -6.0f32;
        let mut max_x = 6.0f32;

        for block in &self.blocks {
            let pos = block.position;
            min_z = min_z.min(pos.z);
            max_z = max_z.max(pos.z);
            min_x = min_x.min(pos.x);
            max_x = max_x.max(pos.x);
        }

        let num_rows = ((max_z - min_z) / STEP_SIZE).round() as usize + 1;
        let num_cols = ((max_x - min_x) / STEP_SIZE).round() as usize + 1;
        let mut matrix = vec![vec![None; num_cols]; num_rows];

        let lava_level = self.lava.position.z;

        for block in &self.blocks {
            let pos = block.position;
            let row = ((pos.z - min_z) / STEP_SIZE).round() as usize;
            let col = ((pos.x - min_x) / STEP_SIZE).round() as usize;
            if row < num_rows && col < num_cols {
                let is_player_on = |p: &Player| {
                    p.standing_xz().is_some_and(|player_xz| {
                        (pos.x - player_xz.x).abs() <= 0.05 && (pos.z - player_xz.y).abs() <= 0.05
                    })
                };

                matrix[row][col] = Some(BlockInfo {
                    color: block.color,
                    points: block.points,
                    above_lava: pos.z - lava_level,
                    blue_player: is_player_on(blue_player),
                    red_player: is_player_on(red_player),
                });
            }
        }

        matrix
    }
}

pub enum ClockMode {
    Realtime { start: Instant, last_time: Instant },
    Fixed { dt: f32 },
}

pub struct Clock {
    mode: ClockMode,
    pub dt: f32,
    pub game_time: f32,
}

impl Clock {
    pub fn realtime() -> Self {
        let now = Instant::now();
        Self {
            mode: ClockMode::Realtime {
                start: now,
                last_time: now,
            },
            dt: 0.0,
            game_time: 0.0,
        }
    }

    pub fn fixed(dt: f32) -> Self {
        Self {
            mode: ClockMode::Fixed { dt },
            dt,
            game_time: 0.0,
        }
    }

    pub fn tick(&mut self) {
        match &mut self.mode {
            ClockMode::Realtime { start, last_time } => {
                let now = Instant::now();
                self.game_time = now.duration_since(*start).as_secs_f32();
                self.dt = now.duration_since(*last_time).as_secs_f32().min(0.05);
                *last_time = now;
            }
            ClockMode::Fixed { dt } => {
                self.dt = *dt;
                self.game_time += *dt;
            }
        }
    }
}
