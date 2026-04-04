# Release Cycle Guide

> **High-signal reference** for the `edgequake-llm` dual-package release process.
> Covers the Rust crate (`edgequake-llm` → crates.io) and the Python bindings
> (`edgequake-litellm` → PyPI), ensuring every release is reproducible,
> security-audited, and fully automated by CI/CD.

---

## 1. Release Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RELEASE PIPELINE                                   │
│                                                                         │
│  Developer                GitHub                 Registries             │
│  ─────────                ──────                 ──────────             │
│                                                                         │
│  ┌──────────┐   push      ┌──────────────────┐   ┌────────────────┐   │
│  │ feature  │ ──────────> │  PR + CI checks  │   │  crates.io     │   │
│  │ branch   │             │  (ci.yml)        │   │  (Rust crate)  │   │
│  └──────────┘             │  (python-ci.yml) │   └────────┬───────┘   │
│                           └────────┬─────────┘            │            │
│  ┌──────────┐   merge     ┌────────▼─────────┐            │ cargo      │
│  │  main    │ <────────── │  PR review &     │            │ publish    │
│  │  branch  │             │  approval        │            │            │
│  └────┬─────┘             └──────────────────┘   ┌────────▼───────┐   │
│       │                                           │  PyPI          │   │
│       │ bump version +                            │  (Python pkg)  │   │
│       │ update CHANGELOG                          └────────┬───────┘   │
│       │                                                    │            │
│  ┌────▼─────┐  v-tag      ┌──────────────────┐   maturin  │            │
│  │  release │ ──────────> │  publish.yml     │ ───────────┘            │
│  │  commit  │             │  python-         │                          │
│  └──────────┘  py-v-tag   │  publish.yml     │                          │
│                           └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Versioning Scheme

Both packages follow [Semantic Versioning](https://semver.org/) independently:

```
┌──────────────────────────────────────────────────────────────┐
│  Package              Tag format      Example                │
│  ─────────────────    ──────────────  ───────────────────── │
│  edgequake-llm        v{MAJOR}.{MINOR}.{PATCH}  v0.5.0      │
│  edgequake-litellm    py-v{MAJOR}.{MINOR}.{PATCH} py-v0.4.0 │
└──────────────────────────────────────────────────────────────┘

SemVer decision guide:
  PATCH  Bug fixes, doc/CI improvements, dependency bumps
  MINOR  New providers, new features, new public API (backward-compatible)
  MAJOR  Breaking changes to public traits / types
```

---

## 3. Step-by-Step Release Process

### 3.1 Pre-flight Checks

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Verify main is green                                   │
│                                                                 │
│  $ open https://github.com/raphaelmansuy/edgequake-llm/actions │
│                                                                 │
│  ci.yml (Rust CI)        ──────────────────> ✅ success         │
│  python-ci.yml (Py CI)   ──────────────────> ✅ success         │
│                                                                 │
│  Both must be green on the HEAD commit before tagging.          │
└─────────────────────────────────────────────────────────────────┘
```

```bash
# Confirm locally
git checkout main && git pull origin main
cargo test --locked
```

### 3.2 Update CHANGELOG

Edit `CHANGELOG.md`:

```
┌─────────────────────────────────────────────────────────────────┐
│  CHANGELOG.md structure                                         │
│                                                                 │
│  ## [Unreleased]             ← keep empty (or staging ground)  │
│                                                                 │
│  ## [0.5.0] - 2026-04-04    ← NEW: add this section            │
│  ### Added                                                      │
│  ### Fixed                                                      │
│  ### Changed                                                    │
│                                                                 │
│  ## [0.4.0] - 2026-04-04    ← previous release                 │
│  ...                                                            │
└─────────────────────────────────────────────────────────────────┘
```

Rules:
- Follow [Keep a Changelog](https://keepachangelog.com) format
- Reference GitHub issue/PR numbers: `(fixes #31)`
- Group by `Added / Fixed / Changed / Deprecated / Removed / Security`
- **Date = today in UTC** (`YYYY-MM-DD`)

### 3.3 Bump Version Numbers

All four places must be updated atomically in one commit:

```
┌────────────────────────────────────────────────────────────────────┐
│  Files to update                         Field                     │
│  ──────────────────────────────────      ──────────────────────── │
│  Cargo.toml                              version = "0.5.0"         │
│  edgequake-litellm/Cargo.toml            version = "0.4.0"         │
│  edgequake-litellm/Cargo.toml            edgequake-llm = { ..      │
│                                          version = "0.5.0" }        │
│  edgequake-litellm/pyproject.toml        version = "0.4.0"         │
└────────────────────────────────────────────────────────────────────┘
```

```bash
# Quick inline sed alternative (or edit manually):
sed -i '' 's/^version = "0.4.0"/version = "0.5.0"/' Cargo.toml
sed -i '' 's/^version = "0.3.0"/version = "0.4.0"/' edgequake-litellm/Cargo.toml
sed -i '' 's/version = "0.4.0"$/version = "0.5.0"/' edgequake-litellm/Cargo.toml  # dep pin
sed -i '' 's/version = "0.3.0"/version = "0.4.0"/' edgequake-litellm/pyproject.toml
```

### 3.4 Commit the Release Prep

```bash
git add CHANGELOG.md Cargo.toml Cargo.lock \
        edgequake-litellm/Cargo.toml \
        edgequake-litellm/pyproject.toml

git commit -m "release: edgequake-llm v0.5.0 + edgequake-litellm v0.4.0"
git push origin main
```

> Wait for both CI workflows to go green on this commit before tagging.

### 3.5 Tag and Push — Triggers Publish

```
┌─────────────────────────────────────────────────────────────────────┐
│  Tagging triggers automated publish workflows                       │
│                                                                     │
│   git tag v0.5.0                                                    │
│   git tag py-v0.4.0                                                 │
│   git push origin v0.5.0 py-v0.4.0                                 │
│                                                                     │
│                    ┌────────────────────────────────────────┐       │
│   Tag v0.5.0  ──── │  publish.yml                           │       │
│                    │  1. Pre-flight: fmt, clippy, tests, doc│       │
│                    │  2. cargo audit                         │       │
│                    │  3. Environment gate: crates-io         │       │
│                    │  4. cargo publish --locked              │       │
│                    │  5. GitHub Release + .crate artifact    │       │
│                    └────────────────────────────────────────┘       │
│                                                                     │
│                    ┌────────────────────────────────────────┐       │
│   Tag py-v0.4.0 ── │  python-publish.yml                    │       │
│                    │  1. Pre-flight: clippy, ruff, mypy      │       │
│                    │  2. Build wheels (manylinux, musllinux, │       │
│                    │     macOS, windows) in parallel         │       │
│                    │  3. PyPI upload via OIDC or token       │       │
│                    │  4. maturin upload → PyPI (OIDC)        │       │
│                    │  5. GitHub Release + wheel artifacts    │       │
│                    └────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Required Secrets & Environments

### 4.1 Rust / crates.io

```
┌───────────────────────────────────────────────────────────┐
│  Secret                Where                   Value       │
│  ──────────────────    ──────────────────────  ─────────  │
│  CARGO_REGISTRY_TOKEN  Repo secret (Actions)   crates.io  │
│                        Settings → Secrets →    API token  │
│                        Actions → New secret    (scoped)   │
│                                                            │
│  Environment: crates-io                                    │
│  (optional protection: require manual review before        │
│   the publish step runs)                                   │
└───────────────────────────────────────────────────────────┘
```

Generate token at `https://crates.io/settings/tokens` → scope: `publish-update`.

### 4.2 Python / PyPI

```
┌───────────────────────────────────────────────────────────┐
│  Method: OIDC Trusted Publishing (no long-lived secret)   │
│                                                            │
│  PyPI project settings:                                    │
│    Publisher   → GitHub                                    │
│    Owner       → raphaelmansuy                             │
│    Repository  → edgequake-llm                             │
│    Workflow    → python-publish.yml                        │
│    Environment → (blank)                                   │
│                                                            │
│  GitHub: no environment is required for OIDC.              │
│  Optional fallback: configure PYPI_API_TOKEN as a secret.  │
└───────────────────────────────────────────────────────────┘
```

---

## 5. Wheel Build Matrix

```
┌──────────────────────────────────────────────────────────────────────┐
│  python-publish.yml builds wheels for:                              │
│                                                                      │
│  Platform        Architecture   Runner               Notes          │
│  ─────────────   ────────────   ──────────────────   ────────────── │
│  Linux           x86_64         ubuntu-latest        manylinux auto │
│  Linux           aarch64        ubuntu-latest        manylinux auto │
│  Linux (musl)    x86_64         ubuntu-latest        musllinux 1.2  │
│  Linux (musl)    aarch64        ubuntu-latest        musllinux 1.2  │
│  macOS           x86_64         macos-latest         cross-compile  │
│  macOS           aarch64        macos-latest         native arm64   │
│  Windows         x86_64         windows-latest       msvc           │
│                                                                      │
│  Python ABI: abi3-py39 (supports Python 3.9+)                       │
│  Tag format: edgequake_litellm-{ver}-cp39-abi3-{platform}.whl       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. End-to-End Timeline

```
Hour 0                       Hour 0.5                    Hour 1+
  │                             │                             │
  ▼                             ▼                             ▼
[PR merged]              [Version bump          [Tags pushed]
  │                        committed & CI         │
  ├─ Rust CI #N (push)     passes]               ├─ publish.yml starts
  │   ~ 4 min                                    │   preflight (6 min)
  │                                              │   cargo publish
  └─ Python CI #N (push)                         │   crates.io live
      ~ 15 min (aarch64)                         │
                                                 └─ python-publish.yml
                                                     builds 7 wheel targets
                                                     in parallel (~15 min)
                                                     PyPI live
```

---

## 7. Post-Release Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│  VERIFY                                                         │
│                                                                 │
│  [ ] crates.io: https://crates.io/crates/edgequake-llm         │
│      → version 0.5.0 visible, README rendered                   │
│                                                                 │
│  [ ] PyPI: https://pypi.org/project/edgequake-litellm/         │
│      → version 0.4.0 visible, all 7 wheel variants present     │
│                                                                 │
│  [ ] GitHub Release: create via UI or gh CLI                   │
│      gh release create v0.5.0 --generate-notes --verify-tag   │
│                                                                 │
│  [ ] Downstream: bump edgequake-llm dep in edgecrab/           │
│      Cargo.toml from git path → crates.io version              │
│                                                                 │
│  [ ] Announce (optional): README badge update, blog post        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Emergency Rollback

```
┌────────────────────────────────────────────────────────────────────┐
│  If a critical bug is found immediately after publish:             │
│                                                                    │
│  Rust (crates.io):                                                 │
│    cargo yank --version 0.5.0 edgequake-llm                       │
│    # Yanked versions remain downloadable but blocked from           │
│    # fresh installs. Fix and publish 0.5.1.                        │
│                                                                    │
│  Python (PyPI):                                                    │
│    pip install twine                                               │
│    # Use PyPI web UI: Manage → Delete release (only within 1h)    │
│    # Alternatively: yank via pip yank (if pip >= 23.3)             │
│    # Fix and publish 0.4.1.                                        │
│                                                                    │
│  Git tag rollback (before publish completes):                      │
│    git tag -d v0.5.0 && git push origin :v0.5.0                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. Quick Reference Commands

```bash
# ─── Pre-release ──────────────────────────────────────────────────
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked
cargo audit
cargo doc --no-deps --all-features

# ─── Version bump (edit files, then verify) ───────────────────────
grep '^version' Cargo.toml
grep '^version' edgequake-litellm/Cargo.toml
grep '^version' edgequake-litellm/pyproject.toml

# ─── Commit + tag ─────────────────────────────────────────────────
git add -A
git commit -m "release: edgequake-llm v0.5.0 + edgequake-litellm v0.4.0"
git push origin main

# Wait for CI green, then:
git tag v0.5.0
git tag py-v0.4.0
git push origin v0.5.0 py-v0.4.0

# ─── Monitor ──────────────────────────────────────────────────────
open https://github.com/raphaelmansuy/edgequake-llm/actions
open https://crates.io/crates/edgequake-llm
open https://pypi.org/project/edgequake-litellm/

# ─── Create GitHub Release ────────────────────────────────────────
gh release create v0.5.0 \
  --title "edgequake-llm v0.5.0 — image generation, provider parity, post-0.4.0 fixes" \
  --generate-notes \
  --verify-tag
```
