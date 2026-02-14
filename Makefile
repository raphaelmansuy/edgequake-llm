.PHONY: help build test test-unit test-integration clean publish run-example examples docs lint fmt check install

# Default target
help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  EdgeQuake LLM - Build & Development Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  🏗️  BUILD & CHECK"
	@echo "    make build              Build the project in debug mode"
	@echo "    make build-release      Build the project in release mode"
	@echo "    make check              Check without building (fast)"
	@echo "    make clean              Remove build artifacts"
	@echo ""
	@echo "  🧪 TESTING"
	@echo "    make test               Run all tests"
	@echo "    make test-unit          Run unit tests only"
	@echo "    make test-integration   Run integration tests only"
	@echo ""
	@echo "  📦 PUBLISHING"
	@echo "    make publish            Publish to crates.io (requires access)"
	@echo "    make publish-dry        Simulate publish without uploading"
	@echo ""
	@echo "  🚀 EXAMPLES"
	@echo "    make examples           List all available examples"
	@echo "    make run-basic          Run basic_completion example"
	@echo "    make run-multi          Run multi_provider example"
	@echo ""
	@echo "  📚 DOCUMENTATION & CODE QUALITY"
	@echo "    make docs               Generate and open API documentation"
	@echo "    make lint               Run clippy for linting"
	@echo "    make fmt                Format code with rustfmt"
	@echo "    make fmt-check          Check code formatting without changes"
	@echo ""
	@echo "  🔧 UTILITIES"
	@echo "    make install            Install the package locally"
	@echo "    make update-deps        Update dependencies to latest version"
	@echo "    make info               Display project information"
	@echo ""
	@echo "  📌 VERSION MANAGEMENT"
	@echo "    make version-show       Display current version"
	@echo "    make version-major      Bump major version (e.g., 1.0.0 → 2.0.0)"
	@echo "    make version-minor      Bump minor version (e.g., 1.0.0 → 1.1.0)"
	@echo "    make version-patch      Bump patch version (e.g., 1.0.0 → 1.0.1)"
	@echo "    make version-set V=x.y.z Set version to specific number"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""

# BUILD TARGETS
build:
	@echo "🔨 Building project..."
	cargo build

build-release:
	@echo "🚀 Building release version..."
	cargo build --release

check:
	@echo "✓ Checking project..."
	cargo check

clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	@echo "✓ Done!"

# TESTING TARGETS
test:
	@echo "🧪 Running all tests..."
	cargo test --lib --tests

test-unit:
	@echo "🧪 Running unit tests..."
	cargo test --lib

test-integration:
	@echo "🧪 Running integration tests..."
	cargo test --tests

# PUBLISHING TARGETS
publish-dry:
	@echo "📦 Running dry-run publish..."
	cargo publish --dry-run

publish:
	@echo "📤 Publishing to crates.io..."
	@echo "⚠️  Make sure you have updated CHANGELOG.md and version in Cargo.toml"
	@read -p "Press enter to continue publishing..." dummy
	cargo publish

# EXAMPLE TARGETS
examples:
	@echo "📚 Available Examples:"
	@echo ""
	@echo "  1. basic_completion       - Basic LLM completion example"
	@echo "  2. multi_provider         - Multi-provider configuration example"
	@echo ""
	@echo "Usage: make run-basic  or  make run-multi"
	@echo ""

run-basic:
	@echo "▶️  Running basic_completion example..."
	cargo run --example basic_completion --

run-multi:
	@echo "▶️  Running multi_provider example..."
	cargo run --example multi_provider --

run-example:
	@echo "Please specify which example to run:"
	@echo "  make run-basic    - Basic completion example"
	@echo "  make run-multi    - Multi-provider example"

# DOCUMENTATION & CODE QUALITY
docs:
	@echo "📖 Generating documentation..."
	cargo doc --no-deps --open

lint:
	@echo "🔍 Running clippy (linter)..."
	cargo clippy --all-targets --all-features -- -D warnings

fmt:
	@echo "✨ Formatting code..."
	cargo fmt

fmt-check:
	@echo "✓ Checking code formatting..."
	cargo fmt -- --check

# UTILITIES
install:
	@echo "📦 Installing package locally..."
	cargo install --path .

update-deps:
	@echo "🔄 Updating dependencies..."
	cargo update

info:
	@echo "📋 Project Information:"
	@cargo --version
	@rustc --version
	@echo ""
	@echo "Package: edgequake-llm"
	@echo "Description: Multi-provider LLM abstraction library with caching, rate limiting, and cost tracking"
	@echo "Repository: https://github.com/raphaelmansuy/edgequake-llm"
	@echo ""

# VERSION MANAGEMENT
version-show:
	@echo "📌 Current version:"
	@grep '^version' Cargo.toml | head -n1 | cut -d'"' -f2

version-major:
	@echo "📌 Bumping major version..."
	@current=$$(grep '^version' Cargo.toml | head -n1 | cut -d'"' -f2); \
	major=$$(echo $$current | cut -d'.' -f1); \
	minor=0; \
	patch=0; \
	new_version="$$((major + 1)).$$minor.$$patch"; \
	echo "  $$current → $$new_version"; \
	sed -i '' 's/^version = "'"$$current"'"/version = "'"$$new_version"'"/' Cargo.toml; \
	echo "✓ Version updated in Cargo.toml"

version-minor:
	@echo "📌 Bumping minor version..."
	@current=$$(grep '^version' Cargo.toml | head -n1 | cut -d'"' -f2); \
	major=$$(echo $$current | cut -d'.' -f1); \
	minor=$$(echo $$current | cut -d'.' -f2); \
	patch=0; \
	new_version="$$major.$$((minor + 1)).$$patch"; \
	echo "  $$current → $$new_version"; \
	sed -i '' 's/^version = "'"$$current"'"/version = "'"$$new_version"'"/' Cargo.toml; \
	echo "✓ Version updated in Cargo.toml"

version-patch:
	@echo "📌 Bumping patch version..."
	@current=$$(grep '^version' Cargo.toml | head -n1 | cut -d'"' -f2); \
	major=$$(echo $$current | cut -d'.' -f1); \
	minor=$$(echo $$current | cut -d'.' -f2); \
	patch=$$(echo $$current | cut -d'.' -f3); \
	new_version="$$major.$$minor.$$((patch + 1))"; \
	echo "  $$current → $$new_version"; \
	sed -i '' 's/^version = "'"$$current"'"/version = "'"$$new_version"'"/' Cargo.toml; \
	echo "✓ Version updated in Cargo.toml"

version-set:
	@if [ -z "$(V)" ]; then \
		echo "❌ Error: Please specify version with V=x.y.z"; \
		echo "   Example: make version-set V=1.2.3"; \
		exit 1; \
	fi
	@echo "📌 Setting version..."
	@current=$$(grep '^version' Cargo.toml | head -n1 | cut -d'"' -f2); \
	echo "  $$current → $(V)"; \
	sed -i '' 's/^version = "'"$$current"'"/version = "$(V)"/' Cargo.toml; \
	echo "✓ Version updated in Cargo.toml"

# Phony targets that don't create files
.PHONY: help build build-release check clean test test-unit test-integration publish-dry publish examples run-basic run-multi run-example docs lint fmt fmt-check install update-deps info version-show version-major version-minor version-patch version-set
