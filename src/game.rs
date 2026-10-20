use crate::{
    controller::{Controller, KeyboardController},
    player::{Player, PlayerColor},
    scene::Scene,
};

pub struct Game {
    pub scene: Scene,
    pub red_player: Player,
    pub blue_player: Player,
    pub red_controller: Box<dyn Controller>,
    pub blue_controller: Box<dyn Controller>,
}

impl Default for Game {
    fn default() -> Self {
        let mut scene = Scene::new();
        let blue_player = Player::new(&mut scene.world, PlayerColor::Blue);
        let red_player = Player::new(&mut scene.world, PlayerColor::Red);

        Self {
            scene,
            red_player,
            blue_player,
            red_controller: Box::new(KeyboardController::wasd()),
            blue_controller: Box::new(KeyboardController::arrows()),
        }
    }
}
