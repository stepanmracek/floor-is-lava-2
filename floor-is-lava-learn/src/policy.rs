use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use floor_is_lava_policy::{Layer, Mlp};

pub use floor_is_lava_policy::{ACTIONS, HIDDEN, INPUTS};

#[derive(Module, Debug)]
pub struct Policy {
    hidden1: Linear,
    hidden2: Linear,
    output: Linear,
    activation: Relu,
}

impl Policy {
    pub fn new(device: &Device) -> Self {
        Self {
            hidden1: LinearConfig::new(INPUTS, HIDDEN).init(device),
            hidden2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            output: LinearConfig::new(HIDDEN, ACTIONS.len()).init(device),
            activation: Relu::new(),
        }
    }

    /// [batch, INPUTS] -> [batch, ACTIONS] logits
    pub fn forward(&self, observations: Tensor<2>) -> Tensor<2> {
        self.head(self.features(observations))
    }

    /// [batch, INPUTS] -> [batch, HIDDEN] output of the shared trunk
    pub fn features(&self, observations: Tensor<2>) -> Tensor<2> {
        let x = self.activation.forward(self.hidden1.forward(observations));
        self.activation.forward(self.hidden2.forward(x))
    }

    /// [batch, HIDDEN] -> [batch, ACTIONS] logits
    pub fn head(&self, features: Tensor<2>) -> Tensor<2> {
        self.output.forward(features)
    }

    /// Copy of the weights for fast inference without burn.
    pub fn to_plain(&self) -> Mlp {
        Mlp {
            hidden1: plain_layer(&self.hidden1),
            hidden2: plain_layer(&self.hidden2),
            output: plain_layer(&self.output),
        }
    }
}

/// Converts a burn policy snapshot (`.bpk`) into SafeTensors for inference without burn.
pub fn export(snapshot: &str, output: &str) {
    Policy::new(&Device::default())
        .load_file(snapshot)
        .to_plain()
        .save(std::path::Path::new(output))
        .unwrap();
}

pub fn plain_layer(linear: &Linear) -> Layer {
    Layer {
        weight: linear.weight.val().try_into_vec_as::<f32>().unwrap(),
        bias: linear
            .bias
            .as_ref()
            .unwrap()
            .val()
            .try_into_vec_as::<f32>()
            .unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// deterministic pseudo-random observation with some zeros, like the real ones
    fn test_observation() -> Vec<f32> {
        (0..INPUTS)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    ((i * 37 % 19) as f32 - 9.0) / 10.0
                }
            })
            .collect()
    }

    fn assert_matches_burn(policy: &Policy, plain: &Mlp) {
        let observation = test_observation();
        let input = Tensor::<1>::from_data(observation.as_slice(), &Device::default()).unsqueeze();
        let expected = policy.forward(input).try_into_vec_as::<f32>().unwrap();
        let actual = plain.logits(&observation);

        for (e, a) in expected.iter().zip(&actual) {
            assert!(
                (e - a).abs() < 1e-4,
                "burn {expected:?} != plain {actual:?}"
            );
        }
    }

    #[test]
    fn plain_policy_matches_burn() {
        let policy = Policy::new(&Device::default());
        assert_matches_burn(&policy, &policy.to_plain());
    }

    #[test]
    fn exported_policy_matches_burn() {
        let dir = std::env::temp_dir().join(format!("floor-is-lava-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let snapshot = dir.join("policy.bpk");
        let exported = dir.join("policy.safetensors");

        let policy = Policy::new(&Device::default());
        policy.clone().save_file(&snapshot).unwrap();
        export(snapshot.to_str().unwrap(), exported.to_str().unwrap());
        let loaded = Mlp::load(&exported).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_matches_burn(&policy, &loaded);
    }
}
