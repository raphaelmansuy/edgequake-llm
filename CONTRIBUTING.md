# Contributing to EdgeQuake LLM

Thank you for your interest in contributing! We welcome contributions from the community.

## Development Setup

1. **Install Rust** (1.75.0 or later)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/raphaelmansuy/edgequake-llm.git
   cd edgequake-llm
   ```

3. **Build the project**
   ```bash
   cargo build
   ```

4. **Run tests**
   ```bash
   cargo test
   ```

## Code Style

We follow the standard Rust style guidelines:

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix all warnings
- Write documentation for public APIs
- Add tests for new features

## Testing

- Unit tests: `cargo test`
- Integration tests: `cargo test --test '*'`
- Doc tests: `cargo test --doc`
- All features: `cargo test --all-features`

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with conventional commits format
6. Push to your fork
7. Open a Pull Request

### Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Example:
```
feat: add support for Claude 4 models

- Add Claude 4 model definitions
- Update provider to handle new API
- Add integration tests

Closes #123
```

## Adding a New Provider

1. Create provider file in `src/providers/`
2. Implement `LLMProvider` trait
3. Add to `src/providers/mod.rs`
4. Export in `src/lib.rs`
5. Add integration tests
6. Document usage in README
7. Update CHANGELOG

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers
- Follow Rust community guidelines

## Questions?

Open an issue or join our discussions!
