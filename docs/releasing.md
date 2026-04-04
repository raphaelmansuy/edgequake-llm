# Release & Publishing Guide

This document covers the complete process for publishing `edgequake-llm` to
**crates.io** and `edgequake-litellm` (the Python bindings) to **PyPI**,
including the one-time GitHub configuration required for each.

---

## Overview

| Release track | Trigger | Destination |
|---|---|---|
| Rust crate | `git tag v<X>.<Y>.<Z>` | [crates.io/crates/edgequake-llm](https://crates.io/crates/edgequake-llm) |
| Python wheels | `git tag py-v<X>.<Y>.<Z>` | [pypi.org/project/edgequake-litellm](https://pypi.org/project/edgequake-litellm) |

Both workflows live in `.github/workflows/`:

- `publish.yml` — Rust publish pipeline (preflight → security audit → publish)
- `python-publish.yml` — Python publish pipeline (preflight → sdist → 7-platform wheel matrix → smoke-test → publish)

---

## 1. One-Time GitHub Repository Setup

### 1.1 Create the `crates-io` Environment (Rust releases)

The Rust publish job requires a [GitHub Environment][gh-env] named **`crates-io`**
so that you can add required reviewers (manual approval gate) before any code
is published.

1. Go to **Settings → Environments → New environment**.
2. Name it exactly `crates-io`.
3. Under **Deployment protection rules**, tick **Required reviewers** and add
   yourself (or a trusted team).
4. Click **Save protection rules**.

> **Why?** The `publish` job in `publish.yml` specifies `environment: crates-io`.
> Without a matching environment the job runs with no gate; with one, GitHub
> pauses for a human approval before executing `cargo publish`.

### 1.2 Add the `CARGO_REGISTRY_TOKEN` Secret

1. Log in to [crates.io](https://crates.io) with your GitHub account.
2. Click your avatar → **Account Settings → API Tokens → New Token**.
3. Give it a meaningful name (e.g. `github-actions-edgequake-llm`), select
   **Publish new crates** and **Publish updates**, then click **Create**.
4. Copy the token — you will only see it once.
5. In the GitHub repo: **Settings → Secrets and variables → Actions →
   New repository secret**.
6. Name: `CARGO_REGISTRY_TOKEN`, value: the token you just copied.

> The `publish.yml` workflow reads this as
> `env: CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}`.

### 1.3 PyPI: Trusted Publisher (recommended, no token required)

[PyPI Trusted Publishing][tp] lets GitHub Actions authenticate via OIDC —
no long-lived token ever touches your repository secrets.

#### On PyPI

1. Log in to [pypi.org](https://pypi.org).
2. Navigate to the project (create it first with a manual upload if it does
   not yet exist) → **Manage → Publishing → Add a new publisher**.
3. Fill in:
   | Field | Value |
   |---|---|
   | Owner | `raphaelmansuy` |
   | Repository | `edgequake-llm` |
   | Workflow name | `python-publish.yml` |
   | Environment name | *(leave blank)* |
4. Click **Add**.

#### In GitHub

No secret is needed. The `python-publish.yml` workflow already requests the
`id-token: write` permission that OIDC requires:

```yaml
permissions:
  id-token: write
```

When `PYPI_API_TOKEN` is **not** set the workflow auto-selects OIDC:

```bash
# Determine publish method (inside the workflow)
if [ -n "$PYPI_API_TOKEN" ]; then
  echo "method=token"
else
  echo "method=oidc"   # ← this path when using Trusted Publishers
fi
```

### 1.4 PyPI: API Token (fallback alternative)

If you prefer a classic token instead of (or in addition to) Trusted Publishing:

1. Log in to [pypi.org](https://pypi.org) → **Account settings → API tokens →
   Add API token**.
2. Scope it to the `edgequake-litellm` project.
3. Copy the token (shown only once).
4. In GitHub: **Settings → Secrets and variables → Actions → New repository
   secret**.
5. Name: `PYPI_API_TOKEN`, value: the token.

When `PYPI_API_TOKEN` is present it takes precedence over OIDC automatically.

---

## 2. Rust Release Process (`edgequake-llm`)

### 2.1 Pre-release Checklist

```
[ ] All tests pass locally:   cargo test --locked
[ ] No clippy warnings:        cargo clippy --all-targets --all-features -- -D warnings
[ ] No fmt issues:             cargo fmt --all -- --check
[ ] Documentation builds:      cargo doc --no-deps --all-features
[ ] Security audit clean:      cargo audit
[ ] CHANGELOG.md updated
[ ] Cargo.toml `version` bumped to X.Y.Z
[ ] README examples updated if API changed
```

### 2.2 Bump the Version

Edit `Cargo.toml` in the repo root:

```toml
[package]
version = "X.Y.Z"   # ← update this
```

Commit:

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: bump version to X.Y.Z"
git push
```

### 2.3 Tag and Push

```bash
git tag v X.Y.Z          # must match Cargo.toml exactly
git push origin vX.Y.Z
```

> The CI pipeline automatically rejects the tag if the version does not match
> `Cargo.toml`, so there is no risk of publishing a stale or mismatched version.

### 2.4 Approve the Publish Gate

1. GitHub → **Actions → Publish to crates.io** → click the pending run.
2. The `Publish to crates.io` job waits for approval in the `crates-io`
   environment.
3. Click **Review pending deployments → Approve and deploy**.

The crate is live on crates.io within a few minutes.

---

## 3. Python Release Process (`edgequake-litellm`)

### 3.1 Pre-release Checklist

```
[ ] Rust unit tests pass:    cargo test --locked  (from repo root)
[ ] Python clippy clean:     cd edgequake-litellm && cargo clippy --all-features -- -D warnings
[ ] Ruff lint:               cd edgequake-litellm && ruff check .
[ ] Mypy types:              cd edgequake-litellm && mypy .
[ ] Python tests pass:       maturin develop --uv && python -m pytest
[ ] CHANGELOG.md updated
[ ] Version bumped in BOTH:
    edgequake-litellm/Cargo.toml  (version = "...")
    edgequake-litellm/pyproject.toml  (version = "...")
```

### 3.2 Bump the Version

Both files must agree (the CI `version-check` job enforces this):

```bash
# edgequake-litellm/Cargo.toml
[package]
version = "X.Y.Z"

# edgequake-litellm/pyproject.toml
[project]
version = "X.Y.Z"
```

Commit:

```bash
git add edgequake-litellm/Cargo.toml edgequake-litellm/pyproject.toml Cargo.lock
git commit -m "chore(python): bump edgequake-litellm to X.Y.Z"
git push
```

### 3.3 Tag and Push

The Python publish workflow triggers on `py-v*` tags (distinct from the Rust
`v*` tag series to allow independent versioning):

```bash
git tag py-vX.Y.Z
git push origin py-vX.Y.Z
```

### 3.4 Wheel Build Matrix

The workflow automatically builds **7 platform/architecture combinations**:

| Platform | Arch | Libc |
|---|---|---|
| Linux | x86_64 | manylinux (glibc) |
| Linux | aarch64 | manylinux (glibc) |
| Linux | x86_64 | musl (Alpine) |
| Linux | aarch64 | musl (Alpine) |
| macOS | x86_64 | — |
| macOS | arm64 | — |
| Windows | x86_64 | — |

All wheels target Python ≥ 3.9 via `abi3-py39`.

### 3.5 Dry-Run Without Tagging

You can trigger the publish workflow manually to build all wheels and verify
everything works before actually uploading:

1. GitHub → **Actions → Publish Python wheels to PyPI → Run workflow**.
2. Set **Dry run** to `true` (the default).
3. Click **Run workflow**.

All wheels are built and uploaded as GitHub Actions artifacts — nothing is
sent to PyPI.

To upload after a successful dry run, re-run with **Dry run** = `false`, or
simply push a `py-v*` tag.

---

## 4. Repository Variables for E2E Tests

The CI pipeline gates provider-specific end-to-end tests behind repository
variables. Set these in **Settings → Secrets and variables → Actions →
Variables** (not Secrets — these are not sensitive):

| Variable | Value to enable | Tests gated |
|---|---|---|
| `AZURE_E2E_ENABLED` | `true` | Azure OpenAI live tests |
| `GEMINI_E2E_ENABLED` | `true` | Gemini live tests |
| `VERTEXAI_E2E_ENABLED` | `true` | Vertex AI live tests |

When a variable is absent or set to any other value, the corresponding test
job is skipped (exit 0).

### 4.1 Provider Secrets for E2E Tests

Add these as **repository secrets** when enabling the corresponding variable:

#### Azure OpenAI
| Secret | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | API key from Azure Portal |
| `AZURE_OPENAI_ENDPOINT` | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Deployment name (e.g. `gpt-4o`) |

#### Google Gemini
| Secret | Description |
|---|---|
| `GEMINI_API_KEY` | API key from [aistudio.google.com](https://aistudio.google.com) |

#### Google Vertex AI
| Secret | Description |
|---|---|
| `GCP_SA_KEY_JSON` | JSON key of a service account with `roles/aiplatform.user` |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID |
| `GOOGLE_CLOUD_REGION` | Region (e.g. `us-central1`) |

---

## 5. Security Audit Configuration

`audit.toml` at the repo root configures which advisories are acknowledged:

```toml
[advisories]
ignore = [
    "RUSTSEC-2025-0012",  # backoff – unmaintained, transitive via async-openai
    "RUSTSEC-2024-0384",  # instant – unmaintained, transitive via async-openai
]
```

`cargo audit` reads this file automatically. Both entries are **unmaintained**
(not vulnerability) advisories that have no upstream fix — they come from
`async-openai` and cannot be removed until that crate updates its dependencies.

When a new advisory appears in CI that is not in `audit.toml`, determine:

1. Is there a fixed version of the affected crate? → Run `cargo update <crate>` and commit `Cargo.lock`.
2. Is the advisory unfixable (unmaintained, no replacement exists)? → Add it to `audit.toml` with a comment explaining why.

---

## 6. Quick Reference

```bash
# Rust release
cargo test --locked && cargo fmt --all -- --check && cargo audit
git add Cargo.toml && git commit -m "chore: bump to vX.Y.Z"
git tag vX.Y.Z && git push && git push origin vX.Y.Z
# → approve publish gate on GitHub Actions

# Python release
git add edgequake-litellm/Cargo.toml edgequake-litellm/pyproject.toml
git commit -m "chore(python): bump edgequake-litellm to X.Y.Z"
git tag py-vX.Y.Z && git push && git push origin py-vX.Y.Z
# → CI builds all 7 wheel platforms and publishes to PyPI automatically
```

[gh-env]: https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment
[tp]: https://docs.pypi.org/trusted-publishers/
