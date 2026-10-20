# Floor is lava

This game is still under heavy development!

Simple remake of [3D Spryjinx](https://www.myabandonware.com/game/3d-spryjinx-li7) game 

# How to build and run

### Prerequisites

- [Rust](https://www.rust-lang.org/) (Rust 1.85 or newer, as this project uses the Rust 2024 edition)

### Running

Run the game from the repository root directory (so relative paths to `assets/` resolve properly):

```sh
cargo run --release
```

If you run the executable directly, ensure your working directory contains the `assets/` folder.

# How to play

Press **Space** on the starting screen to begin the match.

### Objective

Compete as two players (**Red** vs **Blue**) to score the most points while escaping the rising lava:

1. **Hop across blocks**: Jump between floating blocks to stay ahead of the rising lava.
2. **Claim blocks**: Landing on a neutral (gray) or opponent's block paints it with your color.
3. **Score points**: When the lava rises and submerges a claimed block, the player who owns that color receives the points displayed on that block (values range from 1 to 9).
4. **Avoid the lava**: Jumping off the grid into empty space or letting the lava reach you causes you to fall in and perish. Fallen players will automatically respawn on a safe block above the lava after a short delay.

### Controls

| Action | Red Player | Blue Player |
|---|---|---|
| **Move / Jump Forward** | `W` | `Up Arrow` |
| **Move / Jump Backward** | `S` | `Down Arrow` |
| **Move / Jump Left** | `A` | `Left Arrow` |
| **Move / Jump Right** | `D` | `Right Arrow` |

- **Start Game**: `Space`
