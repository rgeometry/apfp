# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the library code. Key modules: `ap64/` (adaptive arithmetic), `expansion.rs` (low-level kernels), and `geometry/` (public predicates + `Coord`).
- `tests/` holds property and integration tests; QuickCheck-based geometry checks live in `tests/geometry_tests.rs`.
- `benches/` includes Criterion benchmarks (`orient2d_bench.rs`).
- `SHEWCHUK.md` captures research notes; `README.md` describes high-level usage.

## Build, Test, and Development Commands
- `cargo fmt` – format the entire workspace.
- `cargo clippy --all-targets` – run lint checks for lib, tests, and benches.
- `cargo test` – execute unit, property, and integration tests.
- `cargo bench --bench orient2d_bench` – benchmark `orient2d` variants.

## Coding Style & Naming Conventions
- Rust 2024 edition; follow `rustfmt` defaults (run `cargo fmt` before committing).
- Keep modules small and well-named (`ap64`, `geometry`, etc.); prefer snake_case for files/functions.
- Leverage `Ap64` for adaptive numeric work; expose ergonomics (e.g., `Coord` with `f64`).

## Testing Guidelines
- Unit tests with `#[test]`, property tests via QuickCheck. Place module-specific tests near implementation or under `tests/`.
- Maintain magnitude guards (`MAG_LIMIT`) in random properties to avoid overflow.
- All new features require `cargo test` and `cargo clippy --all-targets` to pass.

## Commit & Pull Request Guidelines
- Use concise, descriptive commit messages in lowercase semantic style (`feat: add geometry predicates`, `fix: handle orient2d edge case`).
- PRs should link relevant issues, describe major changes, and note test/benchmark results (`cargo test`, `cargo bench`). Screenshots unnecessary unless UI-affecting docs/assets are added.

## Additional Tips
- Benchmarks depend on Criterion 0.5; keep sample sizes reasonable to preserve runtime in CI.
- When introducing new predicates, mirror existing patterns: offer `Coord`-friendly APIs, rational cross-checks, and QuickCheck comparisons with external references (`robust`).
