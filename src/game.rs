use crate::{
    controller::{Controller, KeyboardController, SimpleAiController},
    player::{Player, PlayerColor},
    scene::{Clock, Scene},
};

pub const STEP_SIZE: f32 = 2.0;

pub struct Game {
    pub scene: Scene,
    pub red_player: Player,
    pub blue_player: Player,
    pub red_controller: Box<dyn Controller>,
    pub blue_controller: Box<dyn Controller>,
}

impl Game {
    pub fn new(red_player_ai: bool, blue_player_ai: bool, headless: bool) -> Self {
        let mut scene = if headless {
            Scene::headless()
        } else {
            Scene::headed()
        };
        let blue_player = Player::new(scene.world.as_mut(), PlayerColor::Blue);
        let red_player = Player::new(scene.world.as_mut(), PlayerColor::Red);

        Self {
            scene,
            red_player,
            blue_player,
            red_controller: if red_player_ai {
                Box::new(SimpleAiController)
            } else {
                Box::new(KeyboardController::wasd())
            },
            blue_controller: if blue_player_ai {
                Box::new(SimpleAiController)
            } else {
                Box::new(KeyboardController::arrows())
            },
        }
    }

    pub fn step_simulation(&mut self, clock: &Clock) {
        if let Some(a) =
            self.blue_controller
                .get_action(None, &self.scene, &self.blue_player, &self.red_player)
        {
            self.blue_player.apply_action(a, clock);
        }
        if let Some(a) =
            self.red_controller
                .get_action(None, &self.scene, &self.red_player, &self.blue_player)
        {
            self.red_player.apply_action(a, clock);
        }

        let mut players: [&mut Player; 2] = [&mut self.blue_player, &mut self.red_player];
        if rand::random::<bool>() {
            players.swap(0, 1);
        }
        for player in players {
            player.update(clock, &mut self.scene);
        }
        self.scene
            .update(clock, &[&self.red_player, &self.blue_player]);
    }
}
