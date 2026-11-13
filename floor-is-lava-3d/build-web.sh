#!/usr/bin/env bash
# Builds the 3D client for the web into floor-is-lava-3d/web/pkg.
#
#   ./build-web.sh          build
#   ./build-web.sh serve    build, then serve on http://localhost:8000 (open ?ai-hard etc.)
#
# Needs: rustup target add wasm32-unknown-unknown
#        cargo install wasm-bindgen-cli --version <version of wasm-bindgen in Cargo.lock>
set -euo pipefail

crate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$crate_dir")"
cd "$root"

wasm_bindgen="$(command -v wasm-bindgen || echo "$HOME/.cargo/bin/wasm-bindgen")"
if [[ ! -x "$wasm_bindgen" ]]; then
    echo "wasm-bindgen not found; install it with cargo install wasm-bindgen-cli" >&2
    exit 1
fi

# The CLI must match the wasm-bindgen crate version exactly.
lock_version="$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"$/\1/p')"
cli_version="$("$wasm_bindgen" --version | awk '{print $2}')"
if [[ "$lock_version" != "$cli_version" ]]; then
    echo "wasm-bindgen CLI is $cli_version but Cargo.lock has $lock_version;" >&2
    echo "run: cargo install wasm-bindgen-cli --version $lock_version" >&2
    exit 1
fi

cargo build --release -p floor-is-lava-3d --target wasm32-unknown-unknown

"$wasm_bindgen" --target web --out-dir "$crate_dir/web/pkg" \
    target/wasm32-unknown-unknown/release/floor-is-lava-3d.wasm

echo "Built $crate_dir/web"

if [[ "${1:-}" == "serve" ]]; then
    python3 -m http.server 8000 -d "$crate_dir/web"
fi
