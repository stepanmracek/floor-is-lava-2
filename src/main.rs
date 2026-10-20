mod game_state;
mod player;
mod scene;

use kiss3d::prelude::*;
use player::{Player, PlayerColor};
use scene::Scene;

use crate::game_state::{GameState, StartState};

#[kiss3d::main]
async fn main() {
    let mut window = Window::new("Floor is lava!").await;
    window.set_shadows_enabled(false);
    window.set_samples(NumSamples::One);
    //window.set_vsync(false);

    let mut scene = Scene::new();
    let mut blue_player = Player::new(&mut scene.world, PlayerColor::Blue);
    let mut red_player = Player::new(&mut scene.world, PlayerColor::Red);
    let mut state: Box<dyn GameState> = Box::new(StartState::default());

    while window.render_3d(&mut scene.world, &mut scene.camera).await {
        if let Some(new_state) =
            state.step(&mut window, &mut scene, &mut red_player, &mut blue_player)
        {
            state = new_state;
        }
    }
}
