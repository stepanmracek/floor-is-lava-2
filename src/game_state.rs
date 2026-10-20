use crate::{
    player::Player,
    scene::{Clock, Scene},
};
use kiss3d::{glamx::vec2, prelude::*};
use std::sync::Arc;

pub trait GameState {
    fn step(
        &mut self,
        window: &mut Window,
        scene: &mut Scene,
        red_player: &mut Player,
        blue_player: &mut Player,
    ) -> Option<Box<dyn GameState>>;
}

pub struct StartState {
    font: Arc<Font>,
}

impl Default for StartState {
    fn default() -> Self {
        Self {
            font: Font::default(),
        }
    }
}

impl GameState for StartState {
    fn step(
        &mut self,
        window: &mut Window,
        _scene: &mut Scene,
        _red_player: &mut Player,
        _blue_player: &mut Player,
    ) -> Option<Box<dyn GameState>> {
        let text = "Press space to start";
        let font_size = 40.0;
        let text_width = font_size * 0.5 * text.len() as f32;
        window.draw_text(
            text,
            vec2(
                (window.width() as f32 - text_width) / 2.0,
                (window.height() as f32 - font_size) / 2.0,
            ),
            font_size,
            &self.font,
            WHITE,
        );

        if window.get_key(Key::Space) == Action::Press {
            Some(Box::new(RunState::default()))
        } else {
            None
        }
    }
}

pub struct RunState {
    clock: Clock,
    font: Arc<Font>,
    fps: f32,
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            clock: Clock::default(),
            font: Font::default(),
            fps: 0.0,
        }
    }
}

impl GameState for RunState {
    fn step(
        &mut self,
        window: &mut Window,
        scene: &mut Scene,
        red_player: &mut Player,
        blue_player: &mut Player,
    ) -> Option<Box<dyn GameState>> {
        self.clock.tick();
        let current_fps = if self.clock.dt > 0.0 {
            1.0 / self.clock.dt
        } else {
            0.0
        };
        self.fps = self.fps * 0.9 + current_fps * 0.1;
        window.draw_text(
            &format!("FPS: {:.0}", self.fps),
            vec2(0.0, 0.0),
            12.0,
            &self.font,
            WHITE,
        );

        window.draw_text(
            &format!("{}", scene.red_score),
            vec2(20.0, 20.0),
            30.0,
            &self.font,
            RED,
        );
        window.draw_text(
            &format!("{}", scene.blue_score),
            vec2(window.width() as f32 - 60.0, 20.0),
            30.0,
            &self.font,
            BLUE,
        );

        blue_player.control(window, &self.clock);
        red_player.control(window, &self.clock);

        let mut players: [&mut Player; 2] = [blue_player, red_player];
        if rand::random::<bool>() {
            players.swap(0, 1);
        }
        for player in players {
            player.update(&self.clock, scene);
        }
        scene.update(&self.clock, &[red_player, blue_player]);

        None
    }
}
