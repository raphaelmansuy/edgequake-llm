# Running CI Checks Locally

This guide shows how to reproduce every CI job from `.github/workflows/ci.yml`
and `.github/workflows/python-ci.yml` on your local machine before pushing,
so you never need to wait for GitHub Actions to tell you something is broken.

---

## Prerequisites

```bash
# Rust toolchain (stable + MSRV)
rustup toolchain install stable
rustup toolchain install 1.83.0          # MSRV declared in Cargo.toml
rustup component add rustfmt clippy      # for stable

# Security auditing
cargo install cargo-audit --locked

# Python bindings (edgequake-litellm)
pip install maturin ruff mypy pytest
# or with uv:
uv pip install maturin ruff mypy pytest
```

---

## 1. Rust CI (`ci.yml`)

### One-shot: run everything in a clean environment

The most important habit: **strip your shell's provider API keys before running
tests**. Many auto-detection tests rely on a completely empty environment.

```bash
# Run the full test suite exactly as CI does (no API keys leaking in)
env -i HOME="$HOME" PATH="$PATH" \
  CARGO_TERM_COLOR=always \
  RUST_BACKTRACE=1 \
  cargo test --locked --verbose
```

`env -i` creates a subprocess with an empty environment, then adds only
`HOME`, `PATH`, and the two Cargo variables that CI also sets. This catches
a whole class of test failures that are invisible when you have
`MISTRAL_API_KEY`, `OPENAI_API_KEY`, etc. in your shell.

---

### Job 1 — Rustfmt

```bash
cargo fmt --all -- --check
```

Fix automatically:
```bash
cargo fmt --all
```

---

### Job 2 — Security audit

```bash
cargo audit
```

`audit.toml` at the repo root tells `cargo-audit` which unmaintained
advisories to acknowledge (RUSTSEC-2025-0012, RUSTSEC-2024-0384). The
command exits 0 even with "allowed warnings" printed.

To see the raw advisory list:
```bash
cargo audit --no-fetch    # use cached advisory DB
```

---

### Job 3 — Clippy

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

`--all-targets` is critical — it includes integration test files in `tests/`
and example code in `examples/`. Clippy on `--lib` alone can silently miss
issues in those.

---

### Job 4 — MSRV check

The minimum supported Rust version is declared in `Cargo.toml`:

```bash
# Read MSRV
grep '^rust-version' Cargo.toml

# Install it (once)
rustup toolchain install 1.83.0

# Run the MSRV check
cargo +1.83.0 test --locked
```

---

### Job 5 — Documentation

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

The `-D warnings` flag promotes every rustdoc warning (missing docstrings,
broken intra-doc links, etc.) to a hard error — exactly as CI does.

To open the docs in a browser after building:
```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --open
```

---

### Job 6 — Build examples

```bash
cargo build --examples --locked
```

This compiles every file under `examples/` without running them. It catches
struct literal exhaustiveness errors (like missing `thought_signature`) that
are in example code but not reachable through `--lib` or `--tests`.

---

### Job 7 — Unit + integration tests

```bash
# Mimic CI exactly (locked deps, no stray env vars)
env -i HOME="$HOME" PATH="$PATH" \
  CARGO_TERM_COLOR=always \
  cargo test --locked
```

#### Sub-suites you can run individually

```bash
# Unit tests only (fast, no I/O)
cargo test --locked --lib

# All integration tests (tests/ directory)
cargo test --locked --tests

# A specific integration test file
cargo test --locked --test e2e_provider_factory

# A specific test by name
cargo test --locked test_provider_priority_chain

# Provider doctests (src/**/*.rs)
cargo test --locked --doc
```

#### Gotcha: provider API keys pollute auto-detection tests

`tests/e2e_provider_factory.rs` tests the `from_env()` priority chain. If
your shell has `MISTRAL_API_KEY` set, the test that expects `"openai"` will
see `"mistral"` instead and fail.

```bash
# Always run these tests with a clean provider environment
env -i HOME="$HOME" PATH="$PATH" \
  cargo test --locked --test e2e_provider_factory
```

---

### Job 8 — Azure / Gemini / Vertex AI E2E tests (optional)

These only run in CI when the corresponding repo variable is set to `true`.
Locally, provide the required secret as an env var:

```bash
# Azure
AZURE_OPENAI_CONTENTGEN_API_KEY=<key> \
AZURE_OPENAI_CONTENTGEN_API_ENDPOINT=<url> \
AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT=<name> \
AZURE_OPENAI_CONTENTGEN_API_VERSION=2024-10-01-preview \
  cargo test --locked --test e2e_azure -- --nocapture

# Gemini
GEMINI_API_KEY=<key> \
  cargo test --locked --test e2e_gemini -- --nocapture

# Vertex AI
GOOGLE_CLOUD_PROJECT=<project> \
GOOGLE_CLOUD_REGION=us-central1 \
GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token) \
  cargo test --locked --test e2e_gemini_vertex -- --nocapture
```

---

### Makefile shortcuts

`make` wraps the most common commands:

| Make target | Equivalent |
|---|---|
| `make fmt` | `cargo fmt --all` |
| `make fmt-check` | `cargo fmt --all -- --check` |
| `make lint` | `cargo clippy --all-targets --all-features -- -D warnings` |
| `make test` | `cargo test --lib --tests` |
| `make test-unit` | `cargo test --lib` |
| `make test-integration` | `cargo test --tests` |
| `make docs` | `cargo doc --no-deps --open` |
| `make publish-dry` | `cargo publish --dry-run` |

> **Note**: `make test` does not set `--locked` and does not strip env vars.
> Use the `env -i` form above to faithfully reproduce CI.

---

## 2. Python CI (`python-ci.yml`)

All commands below are run from inside the `edgequake-litellm/` subdirectory
unless otherwise noted.

### Job 1 — Version consistency

```bash
# Both files must declare the same version
python_ver=$(grep '^version' edgequake-litellm/pyproject.toml | cut -d'"' -f2)
cargo_ver=$(grep '^version' edgequake-litellm/Cargo.toml | head -n1 | cut -d'"' -f2)
[ "$python_ver" = "$cargo_ver" ] && echo "OK: $python_ver" || echo "MISMATCH: pyproject=$python_ver Cargo=$cargo_ver"
```

### Job 2 — Clippy (edgequake-litellm)

```bash
cd edgequake-litellm
cargo clippy --all-features -- -D warnings
```

This compiles the whole workspace (including the parent crate) from the
`edgequake-litellm` directory. It finds issues in `src/types.rs` that the
top-level `cargo clippy` might not flag with the same flags.

### Job 3 — Ruff lint

```bash
cd edgequake-litellm
ruff check .
ruff format --check .
```

Fix automatically:
```bash
ruff check --fix .
ruff format .
```

### Job 4 — Mypy type check

```bash
cd edgequake-litellm
mypy .
```

### Job 5 — Python tests

```bash
cd edgequake-litellm

# Build the native extension in-place (development mode)
maturin develop          # uses system pip
# or
maturin develop --uv     # uses uv (faster)

# Run the Python test suite
python -m pytest
```

### Job 6 — ARM64 / cross build (Linux aarch64)

This uses QEMU emulation in CI and is impractical to run locally on non-ARM
hardware. On an Apple Silicon Mac you can approximate it:

```bash
cd edgequake-litellm
maturin build --target aarch64-apple-darwin --release
# or for Linux aarch64:
cross build --target aarch64-unknown-linux-gnu
```

---

## 3. Running the Full CI Locally with `act`

[`act`](https://github.com/nektos/act) runs GitHub Actions workflows locally
using Docker. It is the closest approximation to what GitHub runs.

```bash
# Install (macOS)
brew install act

# Run the Rust CI workflow
act pull_request -W .github/workflows/ci.yml

# Run the Python CI workflow
act pull_request -W .github/workflows/python-ci.yml

# Run a specific job only
act pull_request -W .github/workflows/ci.yml -j fmt
act pull_request -W .github/workflows/ci.yml -j clippy
act pull_request -W .github/workflows/ci.yml -j test

# Pass secrets for E2E jobs
act pull_request -W .github/workflows/ci.yml \
  --secret GEMINI_API_KEY=<your-key> \
  -j test-gemini
```

> **Limitation**: `act` with the default `ubuntu-latest` image may be missing
> some runner dependencies (e.g., OpenSSL headers for aws-lc-sys). Use the
> `-P ubuntu-latest=catthehacker/ubuntu:act-22.04` platform override for a
> more complete image.

---

## 4. Quick Pre-push Checklist

Run this block before every `git push` to a PR branch:

```bash
#!/usr/bin/env bash
set -e

# ── Formatting ─────────────────────────────────────────────────────────────
cargo fmt --all -- --check

# ── Linting ────────────────────────────────────────────────────────────────
cargo clippy --all-targets --all-features -- -D warnings

# ── Documentation ──────────────────────────────────────────────────────────
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# ── Security audit ─────────────────────────────────────────────────────────
cargo audit

# ── Full test suite (clean env = CI-accurate) ──────────────────────────────
env -i HOME="$HOME" PATH="$PATH" \
  CARGO_TERM_COLOR=always \
  cargo test --locked

# ── examples compile ───────────────────────────────────────────────────────
cargo build --examples --locked

echo "✅ All CI checks passed locally"
```

Save as `scripts/ci-local.sh`, make it executable (`chmod +x`), and run it
before pushing.

---

## 5. Why Tests Can Pass Locally but Fail in CI (and Vice Versa)

| Scenario | Local | CI | Cause |
|---|---|---|---|
| Auto-detection picks wrong provider | `"mistral"` | `"mock"` | `MISTRAL_API_KEY` set in shell |
| Doctest fails in CI only | passes | fails | Struct field added without updating doc example |
| MSRV failure in CI | passes | fails | Local toolchain newer than 1.83.0, accepting syntax not in 1.83 |
| Audit advisory fails in CI | passes | fails | `cargo-audit` DB stale locally, new advisory published since last fetch |
| Feature flag test fails | passes | fails | Local feature enabled by default (cargo workspace feature unification) |

The `env -i` technique at the top of this guide eliminates the first two
rows. The rest require discipline: always test with `--locked` and MSRV, and
always let `cargo audit` fetch a fresh advisory database before a release.
