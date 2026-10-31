//! Learned policy without any ML framework: the observation encoding and a plain-Rust
//! forward pass of the policy network. Shared by the training (fast rollouts) and by
//! the game front-ends.

use std::error::Error;
use std::path::Path;

use floor_is_lava_lib::{BlockColor, Cells, PlayerAction, PlayerColor, VISIBLE_ROWS};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

const COLUMNS: usize = 7;
const FEATURES_PER_CELL: usize = 6;

pub const INPUTS: usize = VISIBLE_ROWS * COLUMNS * FEATURES_PER_CELL;
pub const HIDDEN: usize = 128;
pub const ACTIONS: [Option<PlayerAction>; 5] = [
    Some(PlayerAction::Up),
    Some(PlayerAction::Down),
    Some(PlayerAction::Left),
    Some(PlayerAction::Right),
    None,
];

/// Observation of the game from the perspective of player `me`, `INPUTS` floats.
pub fn observation(cells: &Cells, me: PlayerColor) -> Vec<f32> {
    let my_color = BlockColor::from(me);

    let mut data = Vec::with_capacity(INPUTS);
    for cell in cells.iter().flatten() {
        match cell {
            Some(cell) => {
                let owner = match cell.color {
                    BlockColor::Neutral => 0.0,
                    color if color == my_color => 1.0,
                    _ => -1.0,
                };
                let (me_standing, other_standing, other_landing) = match me {
                    PlayerColor::Red => (cell.red_standing, cell.blue_standing, cell.blue_landing),
                    PlayerColor::Blue => (cell.blue_standing, cell.red_standing, cell.red_landing),
                };
                data.extend([
                    cell.points as f32 / 10.0,
                    owner,
                    cell.distance_from_lava,
                    me_standing,
                    other_standing,
                    other_landing,
                ]);
            }
            None => data.extend([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
    }
    data
}

/// Fully connected layer `output = input · weight + bias`.
///
/// `weight` is row-major `[inputs, outputs]`, the same layout as burn's `Linear`.
pub struct Layer {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Layer {
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.bias.clone();
        let outputs = output.len();
        for (&x, row) in input.iter().zip(self.weight.chunks_exact(outputs)) {
            // inputs are often 0 (missing features, ReLU), skip the whole row
            if x == 0.0 {
                continue;
            }
            for (o, &w) in output.iter_mut().zip(row) {
                *o += x * w;
            }
        }
        output
    }
}

fn relu(mut values: Vec<f32>) -> Vec<f32> {
    for v in &mut values {
        *v = v.max(0.0);
    }
    values
}

/// INPUTS → HIDDEN → ReLU → HIDDEN → ReLU → ACTIONS logits
pub struct PlainPolicy {
    pub hidden1: Layer,
    pub hidden2: Layer,
    pub output: Layer,
}

impl PlainPolicy {
    /// Output of the trunk, `HIDDEN` floats.
    pub fn features(&self, observation: &[f32]) -> Vec<f32> {
        let x = relu(self.hidden1.forward(observation));
        relu(self.hidden2.forward(&x))
    }

    pub fn logits(&self, observation: &[f32]) -> Vec<f32> {
        self.output.forward(&self.features(observation))
    }

    pub fn probabilities(&self, observation: &[f32]) -> Vec<f32> {
        let logits = self.logits(observation);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    /// Samples an action index; `uniform` is a random number from [0, 1).
    pub fn sample_action(&self, observation: &[f32], uniform: f32) -> usize {
        sample(&self.probabilities(observation), uniform)
    }

    fn layers(&self) -> [(&'static str, &Layer); 3] {
        [
            ("hidden1", &self.hidden1),
            ("hidden2", &self.hidden2),
            ("output", &self.output),
        ]
    }

    /// Saves the weights as SafeTensors, f32 little-endian:
    /// `{layer}.weight` with shape `[inputs, outputs]` (row-major) and
    /// `{layer}.bias` with shape `[outputs]`, for `hidden1`, `hidden2` and `output`.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let mut tensors = vec![];
        for (name, layer) in self.layers() {
            let outputs = layer.bias.len();
            let inputs = layer.weight.len() / outputs;
            tensors.push((
                format!("{name}.weight"),
                vec![inputs, outputs],
                to_bytes(&layer.weight),
            ));
            tensors.push((format!("{name}.bias"), vec![outputs], to_bytes(&layer.bias)));
        }

        let views = tensors
            .iter()
            .map(|(name, shape, data)| {
                Ok((name, TensorView::new(Dtype::F32, shape.clone(), data)?))
            })
            .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
        safetensors::serialize_to_file(views, None, path)?;
        Ok(())
    }

    /// Loads weights saved by [`PlainPolicy::save`].
    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let buffer = std::fs::read(path)?;
        Self::from_bytes(&buffer)
    }

    pub fn from_bytes(buffer: &[u8]) -> Result<Self, Box<dyn Error>> {
        let tensors = SafeTensors::deserialize(&buffer)?;
        let tensor = |name: &str, expected_shape: &[usize]| -> Result<Vec<f32>, Box<dyn Error>> {
            let view = tensors.tensor(name)?;
            if view.dtype() != Dtype::F32 || view.shape() != expected_shape {
                return Err(format!(
                    "tensor {name}: expected F32 {expected_shape:?}, got {:?} {:?}",
                    view.dtype(),
                    view.shape()
                )
                .into());
            }
            Ok(from_bytes(view.data()))
        };
        let layer = |name: &str, inputs: usize, outputs: usize| -> Result<Layer, Box<dyn Error>> {
            Ok(Layer {
                weight: tensor(&format!("{name}.weight"), &[inputs, outputs])?,
                bias: tensor(&format!("{name}.bias"), &[outputs])?,
            })
        };

        Ok(Self {
            hidden1: layer("hidden1", INPUTS, HIDDEN)?,
            hidden2: layer("hidden2", HIDDEN, HIDDEN)?,
            output: layer("output", HIDDEN, ACTIONS.len())?,
        })
    }
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Samples an index from a probability distribution; `uniform` is a random number from [0, 1).
pub fn sample(probabilities: &[f32], uniform: f32) -> usize {
    let mut threshold = uniform;
    for (index, prob) in probabilities.iter().enumerate() {
        if threshold < *prob {
            return index;
        }
        threshold -= prob;
    }
    probabilities.len() - 1
}
