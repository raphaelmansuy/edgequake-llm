# OODA-65 Orient: README Structure

## Documentation Section Categories

### Current
| Category | Docs |
|----------|------|
| API Documentation | Rust docs.rs |
| Guides | 4 (providers, architecture, etc) |
| Features | 5 (caching, cost, rate-limiting) |
| Reference | 3 (testing, migration, faq) |

### Proposed
Add new "Operations" category between Features and Reference:
- Performance Tuning
- Security

## Rationale
- Performance and security are operational concerns
- Separate from features (which are about library capabilities)
- Naturally fits between functional docs and reference docs
