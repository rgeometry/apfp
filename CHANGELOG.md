# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fix `int_signum!` macro incorrectly handling minimum signed integer values (e.g., `i8::MIN`, `i16::MIN`, etc.) ([#24](https://github.com/rgeometry/apfp/pull/24))

## [0.1.0] - 2026-01-02

Initial release of apfp - Adaptive Precision Floating-Point arithmetic for robust geometric predicates.

### Added

- `apfp_signum!` macro for computing exact signs of arithmetic expressions
- `square()` helper for efficient squared terms in expressions
- Geometric predicates for f64 coordinates:
  - `orient2d` - orientation test for three points
  - `orient2d_vec` - orientation test using point and direction vector
  - `cmp_dist` - distance comparison between points
- Geometric predicates for integer coordinates (i8, i16, i32, i64)
- Adaptive precision with three-stage evaluation:
  - Fast f64 path with error bounds
  - Double-double arithmetic fallback (~106 bits)
  - Exact floating-point expansion for degenerate cases
- Allocation-free implementation using fixed stack buffers
