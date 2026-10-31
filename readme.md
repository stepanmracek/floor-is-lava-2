# Floor is Lava

A two-player game: jump between blocks, claim them, and stay above the rising lava.
Points are banked only when a block you own sinks into the lava.

## Game Mechanics

Source of truth: `floor-is-lava-lib/src/game.rs`. Constants are in brackets.

### Board and lava

- The board is 7 columns wide (x = −3 … 3). Rows are stacked upwards; only the lowest
  `VISIBLE_ROWS` (4) rows that still exist form the visible grid (`Cells`).
- The game starts with a full row of 7 blocks at y = 0. Red stands at x = −2, blue at x = 2.
- The lava rises continuously at `LAVA_SPEED` (0.25 rows per second).
- New rows spawn above the top row whenever it is less than 3 rows above the lava. In
  each new row every column gets a block with probability 0.7, and one random column is
  guaranteed a block, so a row is never empty.
- Every block has 1–9 points, chosen at random.

### Moving and claiming

- An `Idle` player can jump one cell up, down, left or right. A jump takes
  `JUMP_DURATION` (1 s) and can't be cancelled or redirected. Actions given in any other
  state are ignored.
- On landing on a block, the player claims it: the block takes the player's colour.
  **Whoever lands last owns the block**, which is how blocks are stolen.
- Landing where there is no block (a hole, off the board, or a block that sank during
  the jump) starts a fall.

### Scoring

- When the lava passes a block (block y < lava height), the block is removed and its
  points go to the player whose colour it has. Neutral blocks score nothing.
- **Scoring is delayed:** points from a claimed block arrive only when it sinks, up to
  ~16 s later for a block high on the board. Until then it can be stolen. Stealing near
  the lava is the core of the game.

### Dying and respawning

- An idle player standing below the lava starts `Dying`. A falling player
  (`FALLING_SPEED` 8 rows/s) starts dying when it reaches the lava.
- After `DYING_DURATION` (3 s) the player respawns on a random block more than one row
  above the lava. **Respawning is effectively a free teleport upwards.** An RL agent
  exploited this when far-future points were discounted too strongly.

### Timing details within one `step`

These matter for close calls like the "chicken game": both players on a sinking block,
where whoever jumps first gets the block they jump to stolen by the one who follows.

1. Actions are applied first. An idle player with an action starts jumping on this tick.
2. The lava rises.
3. Blocks below the lava are removed and scored.
4. A new row spawns if needed.
5. Players are updated **in random order** each step, so simultaneous landings are
   resolved randomly:
   - an idle player below the lava starts dying;
   - a finished jump lands (claim) or misses (fall);
   - falling and dying advance.

Consequences:

- **Jumping players are never checked against the lava.** Starting a jump on the very
  last tick before the block sinks saves the player, because the action is applied
  before the lava rises.
- Jumping second wins the chicken game: the follower lands later and takes the block.

### Players' view (`Cells`)

`Game::cells()` exposes the visible grid as `[[Option<Cell>; 7]; VISIBLE_ROWS]`. Each cell
carries:

- points, colour and distance from the lava;
- per player, a *standing* value: 1 when idle on the cell, `1 − progress` on the cell a
  jump started from;
- per player, a *landing* value: the jump progress on the target cell.

A missing block is `None`. Controllers (`SimpleAiController`, keyboard, learned policy)
decide only from `Cells`.
