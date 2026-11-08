use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use clap::Parser;
use floor_is_lava_lib::{Clock, Controller, Game, PlayerColor, SimpleAiController};
use floor_is_lava_policy::PolicyController;

/// Sweep of the simple AI's urgency and lookahead decay (`-` = no lookahead).
///
/// Every configuration plays a round robin against all other configurations and every
/// given policy. Every pairing plays `games` seeded games in each colour assignment
/// (same game seeds).
#[derive(Parser)]
struct Cli {
    /// Seed for the first game; game g uses seed + g.
    seed: u64,
    /// Games per pairing and colour assignment.
    games: u64,
    /// Policies (.safetensors) used as fixed opponents.
    policies: Vec<String>,
    /// Urgencies; the first one with no lookahead is the reference.
    #[arg(long, value_delimiter = ',', default_value = "1,0.5,0.3,0.2,0.1")]
    urgencies: Vec<f32>,
    /// Lookahead decays; every urgency is also played without lookahead.
    #[arg(long, value_delimiter = ',', default_value = "0.9,0.95,0.99,1")]
    decays: Vec<f32>,
}

enum Entrant {
    Simple {
        urgency: f32,
        decay: Option<f32>,
    },
    /// SafeTensors bytes, parsed per game.
    Policy(Vec<u8>),
}

impl Entrant {
    fn controller(&self, color: PlayerColor, seed: u64) -> Box<dyn Controller> {
        match self {
            Entrant::Simple { urgency, decay } => {
                let ai = SimpleAiController::new(color, seed).with_urgency(*urgency);
                Box::new(match decay {
                    Some(decay) => ai.with_lookahead_decay(*decay),
                    None => ai,
                })
            }
            Entrant::Policy(bytes) => {
                Box::new(PolicyController::from_bytes(bytes, color, seed).unwrap())
            }
        }
    }
}

/// Plays one game; returns red score minus blue score.
fn play(seed: u64, red: &Entrant, blue: &Entrant) -> i64 {
    let mut red = red.controller(PlayerColor::Red, seed);
    let mut blue = blue.controller(PlayerColor::Blue, seed.wrapping_add(1 << 32));
    let mut game = Game::with_seed(seed);
    let mut clock = Clock::fixed(1.0 / 60.0);
    for _ in 0..60 * 60 {
        clock.tick();
        game.step(
            &clock,
            red.get_action(game.cells()),
            blue.get_action(game.cells()),
        );
    }
    game.red_player.score as i64 - game.blue_player.score as i64
}

/// Total margin of `a` against `b` over both colour assignments.
fn play_pair(seed: u64, games: u64, a: &Entrant, b: &Entrant) -> i64 {
    (0..games)
        .map(|g| play(seed + g, a, b) - play(seed + g, b, a))
        .sum()
}

fn main() {
    let Cli {
        seed,
        games,
        policies,
        urgencies,
        decays,
    } = Cli::parse();

    let mut entrants = vec![];
    for &urgency in &urgencies {
        entrants.push(Entrant::Simple {
            urgency,
            decay: None,
        });
        for &decay in &decays {
            entrants.push(Entrant::Simple {
                urgency,
                decay: Some(decay),
            });
        }
    }
    let simple = entrants.len();
    for path in &policies {
        entrants.push(Entrant::Policy(std::fs::read(Path::new(path)).unwrap()));
    }

    // simple vs simple (round robin) and simple vs policy; no policy vs policy
    let jobs: Vec<(usize, usize)> = (0..simple)
        .flat_map(|i| (i + 1..entrants.len()).map(move |j| (i, j)))
        .collect();
    let next = AtomicUsize::new(0);
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let done: Vec<Vec<(usize, i64)>> = thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut out = vec![];
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(i, j)) = jobs.get(k) else { break };
                        out.push((k, play_pair(seed, games, &entrants[i], &entrants[j])));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut margins = vec![0i64; jobs.len()];
    for (k, m) in done.into_iter().flatten() {
        margins[k] = m;
    }

    // per configuration: against all other configurations, the reference (entrant 0)
    // and the policies
    let mut vs_simple = vec![0i64; simple];
    let mut vs_reference = vec![0i64; simple];
    let mut vs_policy = vec![0i64; simple];
    for (&(i, j), &m) in jobs.iter().zip(&margins) {
        if j < simple {
            vs_simple[i] += m;
            vs_simple[j] -= m;
            if i == 0 {
                vs_reference[j] -= m;
            }
        } else {
            vs_policy[i] += m;
        }
    }
    let per_game = |total: i64, opponents: usize| {
        total as f32 / (2 * games as usize * opponents.max(1)) as f32
    };

    let mut ranking: Vec<usize> = (0..simple).collect();
    ranking.sort_by_key(|&i| -vs_simple[i]);
    println!("margin per game ({games} games per pairing and colour assignment)");
    println!(
        "{:>8}{:>8}{:>12}{:>14}{:>12}",
        "urgency", "decay", "vs simple", "vs reference", "vs policy"
    );
    for i in ranking {
        let Entrant::Simple { urgency, decay } = entrants[i] else {
            unreachable!()
        };
        let decay = decay.map_or("-".to_string(), |d| d.to_string());
        println!(
            "{urgency:>8}{decay:>8}{:>12.2}{:>14.2}{:>12.2}",
            per_game(vs_simple[i], simple - 1),
            per_game(vs_reference[i], 1),
            per_game(vs_policy[i], policies.len()),
        );
    }
}
