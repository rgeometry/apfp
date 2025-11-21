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
- `nix flake check` – runs the full Crane QA stack (fmt, clippy, doc, Taplo, nextest, audit) using the pinned toolchain.
- `nix run .#format` – one-touch formatter that chains `cargo fmt`, `taplo fmt`, and `alejandra .`.

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
- Use the `gh` tool to open PRs: `gh pr create --title "feat: your feature description" --body "Description of changes..."`.
- **Never use `git commit --no-verify`**: Pre-commit verification (hooks) is mandatory for code quality and consistency.

## Additional Tips
- Benchmarks depend on Criterion 0.5; keep sample sizes reasonable to preserve runtime in CI.
- When introducing new predicates, mirror existing patterns: offer `Coord`-friendly APIs, rational cross-checks, and QuickCheck comparisons with external references (`robust`).

## Performance & Benchmarking
- Profile `orient2d` variants with existing Criterion benches; inspect generated machine code via `cargo asm --lib apfp::geometry::predicates::orient2d`.
- Avoid runtime memory allocations whenever possible; prefer fixed-size stack buffers for intermediate storage.
- Keep adaptive kernels allocation-free so tight loops remain cache-friendly and deterministic.
- Distinguish the adaptive code paths: quick-exit when points are obviously non-co-linear, use the near-co-linear expansion lift when magnitudes suggest cancellation, and fall back to the exact co-linear resolution path that computes and returns the signed area directly.
