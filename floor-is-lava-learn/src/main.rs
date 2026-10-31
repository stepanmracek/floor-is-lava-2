mod actor_critic;
mod policy;
mod reinforce;

/// Usage:
///   floor-is-lava-learn reinforce <seed>
///   floor-is-lava-learn actor-critic <seed> [policy.bpk]
///   floor-is-lava-learn export <policy.bpk> <policy.safetensors>
fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "reinforce" => reinforce::train(10_000, args[2].parse().unwrap()),
        "actor-critic" => actor_critic::train(
            args.get(3).map(String::as_str),
            10_000,
            args[2].parse().unwrap(),
        ),
        "export" => policy::export(&args[2], &args[3]),
        other => panic!("unknown command: {other}"),
    }
}
