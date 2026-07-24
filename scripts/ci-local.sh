#!/usr/bin/env bash
# ci-local.sh — Run every CI check locally before pushing.
#
# Usage:
#   ./scripts/ci-local.sh              # full suite
#   ./scripts/ci-local.sh fmt          # formatting only
#   ./scripts/ci-local.sh clippy       # linting only
#   ./scripts/ci-local.sh test         # tests only (clean env)
#   ./scripts/ci-local.sh audit        # security audit only
#   ./scripts/ci-local.sh docs         # documentation only
#   ./scripts/ci-local.sh python       # Python CI checks (edgequake-litellm)
#   ./scripts/ci-local.sh msrv         # MSRV (Rust 1.95.0) check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; exit 1; }
step() { echo -e "\n${BOLD}${YELLOW}── $1 ──${NC}"; }

# ─── Individual checks ────────────────────────────────────────────────────────

run_fmt() {
    step "Rustfmt"
    cargo fmt --all -- --check && pass "Formatting OK" || fail "Formatting failed — run: cargo fmt --all"
}

run_clippy() {
    step "Clippy"
    cargo clippy --all-targets --all-features -- -D warnings && pass "Clippy OK"
}

run_docs() {
    step "Documentation"
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features && pass "Docs OK"
}

run_audit() {
    step "Security audit"
    cargo audit && pass "Audit OK"
}

run_msrv() {
    step "MSRV check (Rust 1.95.0)"
    MSRV=$(grep '^rust-version' Cargo.toml | cut -d'"' -f2)
    echo "  MSRV: $MSRV"
    if ! rustup toolchain list | grep -q "$MSRV"; then
        echo "  Installing Rust $MSRV …"
        rustup toolchain install "$MSRV"
    fi
    cargo "+$MSRV" test --locked && pass "MSRV OK"
}

run_test() {
    step "Tests (clean environment — simulates CI)"
    # Strip all provider API keys so auto-detection tests work correctly.
    # This is the #1 source of 'passes locally, fails in CI' bugs.
    #
    # Also isolate `.env` — dotenvy loads it from CWD even under env -i when
    # HOME/PATH are preserved, which would otherwise run live e2e against real
    # credentials and hang on slow streaming cases.
    local dotenv_backup=""
    if [[ -f .env ]]; then
        dotenv_backup="$(mktemp "${TMPDIR:-/tmp}/edgequake-llm.env.XXXXXX")"
        mv .env "$dotenv_backup"
        echo "  Isolated .env → $dotenv_backup (restored after tests)"
    fi
    local status=0
    env -i \
        HOME="$HOME" \
        PATH="$PATH" \
        CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" \
        RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}" \
        CARGO_TERM_COLOR=always \
        RUST_BACKTRACE=1 \
        cargo test --locked \
    && pass "All tests OK" || status=$?
    if [[ -n "$dotenv_backup" && -f "$dotenv_backup" ]]; then
        mv "$dotenv_backup" .env
    fi
    return "$status"
}

run_examples() {
    step "Build examples"
    cargo build --examples --locked && pass "Examples build OK"
}

run_python() {
    step "Python CI (edgequake-litellm)"
    cd edgequake-litellm

    # Version consistency
    python_ver=$(grep '^version' pyproject.toml | head -n1 | cut -d'"' -f2)
    cargo_ver=$(grep '^version' Cargo.toml | head -n1 | cut -d'"' -f2)
    if [ "$python_ver" != "$cargo_ver" ]; then
        fail "Version mismatch: pyproject.toml=$python_ver Cargo.toml=$cargo_ver"
    fi
    pass "Version consistency: $python_ver"

    # Clippy (from edgequake-litellm directory)
    cargo clippy --all-features -- -D warnings && pass "Clippy (edgequake-litellm) OK"

    # Ruff lint
    if command -v ruff &>/dev/null; then
        ruff check . && ruff format --check . && pass "Ruff OK"
    else
        echo "  ⚠️  ruff not found — skipping (pip install ruff)"
    fi

    # Mypy
    if command -v mypy &>/dev/null; then
        mypy . && pass "Mypy OK"
    else
        echo "  ⚠️  mypy not found — skipping (pip install mypy)"
    fi

    # Python tests
    if command -v maturin &>/dev/null; then
        maturin develop --uv 2>/dev/null || maturin develop
        python -m pytest && pass "Python tests OK"
    else
        echo "  ⚠️  maturin not found — skipping Python tests (pip install maturin)"
    fi

    cd "$REPO_ROOT"
}

# ─── Dispatch ─────────────────────────────────────────────────────────────────

MODE="${1:-all}"

case "$MODE" in
    fmt)     run_fmt ;;
    clippy)  run_clippy ;;
    docs)    run_docs ;;
    audit)   run_audit ;;
    msrv)    run_msrv ;;
    test)    run_test ;;
    examples) run_examples ;;
    python)  run_python ;;
    all)
        run_fmt
        run_clippy
        run_docs
        run_audit
        run_test
        run_examples
        echo -e "\n${GREEN}${BOLD}✅ All CI checks passed locally!${NC}"
        ;;
    *)
        echo "Unknown target: $MODE"
        echo "Usage: $0 [fmt|clippy|docs|audit|msrv|test|examples|python|all]"
        exit 1
        ;;
esac
