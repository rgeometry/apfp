# Performance Analysis Guide

This document provides guidance on analyzing the performance of functions in the APFP library, including how to run benchmarks and inspect generated assembly code.

## Table of Contents

1. [Using Benchmarks](#using-benchmarks)
2. [Viewing Assembly Code](#viewing-assembly-code)
3. [Assembly Code Inspection](#assembly-code-inspection)
4. [Performance Optimization Workflow](#performance-optimization-workflow)

## Using Benchmarks

The APFP library uses [Criterion](https://github.com/bheisler/criterion.rs) for benchmarking. Benchmarks are located in the `benches/` directory and can be run to measure the performance of various functions.

### Running Benchmarks

To run all benchmarks:

```bash
cargo bench
```

To run a specific benchmark suite:

```bash
cargo bench --bench orient2d_bench
```

This will run the `orient2d_bench` benchmark suite, which includes performance tests for various `orient2d` implementations.

### Quick Benchmark Mode

For faster iteration during development, you can use Criterion's `--quick` flag when 2-3% measurement variation is acceptable:

```bash
cargo bench --bench orient2d_bench -- --quick
```

The `--quick` flag:
- Reduces the number of measurement samples
- Uses shorter warm-up periods
- Provides results in seconds rather than minutes
- Trades accuracy for speed during development and debugging

### Analyzing Benchmark Results

The benchmark suite includes several variants of the `orient2d` function:

- `orient2d_apfp` - The main adaptive precision implementation
- `orient2d_inexact_baseline` - Basic floating-point implementation
- `orient2d_inexact_interval` - Interval arithmetic version
- `orient2d_fixed` - Fixed-precision implementation
- `orient2d_rational` - Exact rational arithmetic reference
- `orient2d_robust` - External robust library reference

Each benchmark runs 5,000 samples with coordinates limited to the range [-1000, 1000] to avoid overflow issues.

### Finding Performance of Specific Functions

To analyze the performance of a specific function like `apfp::analysis::orient2d_fast`:

1. **Check if it's already benchmarked**: Look in `benches/orient2d_bench.rs` to see if the function is included in an existing benchmark group.

2. **Add a new benchmark if needed**: If the function isn't benchmarked, add it to the benchmark file:

```rust
fn orient2d_fast_batch(samples: &[(Coord, Coord, Coord)]) {
    for (a, b, c) in samples {
        black_box(orient2d_fast(a, b, c));
    }
}

// In the benchmark function:
c.bench_function("orient2d_fast", |b| {
    b.iter(|| orient2d_fast_batch(black_box(&samples)))
});
```

3. **Run and compare**: Execute the benchmarks and compare the throughput (operations per second) across different implementations.

### Interpreting Benchmark Output

Criterion provides detailed statistics including:
- **Mean execution time** per operation
- **Standard deviation** to assess measurement stability
- **Outlier analysis** to identify performance anomalies
- **Relative performance** comparisons between implementations

Look for:
- Consistent timing across runs (low standard deviation)
- Expected performance hierarchies (fast filters should be much faster than exact arithmetic)
- Any unexpected performance regressions

## Viewing Assembly Code

The project uses `cargo-asm` to generate and inspect assembly code for performance-critical functions.

### Prerequisites

Install `cargo-asm`:

```bash
cargo install cargo-asm
```

### Generating Assembly for a Function

To view the assembly code for `apfp::analysis::orient2d_fast`:

```bash
cargo asm --lib apfp::analysis::orient2d_fast
```

This generates assembly for the function in debug mode. For release optimizations:

```bash
cargo asm --release --lib apfp::analysis::orient2d_fast
```

### Using Nix for Consistent Assembly Output

The project includes Nix-based tooling for consistent assembly generation across different platforms. The flake configuration targets `aarch64-unknown-linux-gnu` for reproducible output.

To generate assembly using Nix:

```bash
# Build the assembly output
nix build .#cargoAsmOutput

# View the generated assembly
cat result/apfp_analysis_orient2d_fast.s
```

### Assembly Output Options

- **Intel syntax**: `cargo asm --intel --lib function_name`
- **Specific target**: `cargo asm --target aarch64-unknown-linux-gnu --lib function_name`
- **Function context**: `cargo asm --context 5 --lib function_name` (shows 5 lines before/after)

## Assembly Code Inspection

When analyzing assembly code, focus on performance characteristics, correctness guarantees, and optimization opportunities.

### Common Things to Look For

#### 1. Memory Allocations

**What to check**: Look for calls to memory allocation functions
**Why it matters**: Allocations are expensive and can introduce unpredictability
**Patterns to avoid**:
- `malloc`, `calloc`, `realloc`
- `__rust_alloc`, `__rust_realloc`, `__rust_dealloc`
- `alloc::alloc`

The project's assembly checks automatically verify that critical functions contain no allocation calls.

#### 2. Assertions and Panics

**What to check**: Look for panic-related code
**Why it matters**: Assertions can cause unexpected control flow and performance overhead
**Patterns to avoid**:
- `panic`
- `assert`
- `__rust_start_panic`
- `rust_begin_unwind`

#### 3. Control Flow Complexity

**What to check**: Branch instructions and conditional jumps
**Why it matters**: Complex control flow can impact branch prediction
**Look for**:
- Number of conditional branches (`jne`, `je`, `cmp`, etc.)
- Branch target alignment
- Predictable vs. unpredictable branches

#### 4. Register Usage

**What to check**: How registers are utilized
**Why it matters**: Efficient register usage minimizes memory accesses
**Look for**:
- Register spilling to stack
- SIMD instruction usage
- Floating-point register allocation

#### 5. Instruction Selection

**What to check**: Choice of CPU instructions
**Why it matters**: Some instructions are faster or more appropriate for specific data types
**Look for**:
- Use of fused multiply-add (`fma`) for floating-point operations
- Vectorization opportunities
- Memory access patterns (aligned vs. unaligned)

#### 6. Function Call Overhead

**What to check**: Function call instructions
**Why it matters**: Function calls introduce overhead
**Look for**:
- Direct vs. indirect calls
- Inlining opportunities
- Call sequence patterns

### Assembly Quality Checks

The project includes automated checks that validate:

1. **No assertions**: Ensures performance-critical code doesn't contain debug assertions
2. **No allocations**: Verifies that functions don't perform heap allocations
3. **Instruction patterns**: Checks for expected optimization patterns

Run the assembly checks:

```bash
# Using Nix
nix run .#asmCheckTable

# Or manually
./nix/check-no-assertions.sh path/to/assembly.s
./nix/check-no-allocations.sh path/to/assembly.s
```

### Platform-Specific Considerations

Different CPU architectures generate different assembly. The project standardizes on `aarch64-unknown-linux-gnu` for consistent analysis, but consider:

- **x86-64**: Look for SSE/AVX instructions for vectorization
- **ARM64**: Check for NEON SIMD usage
- **RISC-V**: Examine instruction scheduling and branch prediction

## Performance Optimization Workflow

### 1. Establish Baselines

```bash
# Run comprehensive benchmarks
cargo bench --bench orient2d_bench

# Generate assembly for key functions
cargo asm --release --lib apfp::analysis::orient2d_fast > orient2d_fast.s
```

### 2. Profile Hotspots

Use the benchmark results to identify:
- Functions with unexpectedly slow performance
- High variance in timing (potential cache issues)
- Outliers that might indicate degenerate cases

### 3. Analyze Assembly

For each performance-critical function:

```bash
# Generate and inspect assembly
cargo asm --release --lib function_name

# Run automated checks
./nix/check-no-assertions.sh function_name.s
./nix/check-no-allocations.sh function_name.s
```

### 4. Identify Optimization Opportunities

Common optimization targets:

- **Reduce branches**: Simplify conditional logic
- **Improve cache locality**: Reorder operations for better memory access patterns
- **Vectorization**: Use SIMD instructions where applicable
- **Inlining**: Ensure small functions are inlined
- **Constant propagation**: Precompute values where possible

### 5. Implement and Measure

After code changes:

```bash
# Verify correctness
cargo test

# Check performance impact
cargo bench --bench orient2d_bench

# Inspect assembly changes
cargo asm --release --lib modified_function
```

### 6. Regression Testing

Ensure optimizations don't break functionality:

```bash
# Run full test suite
cargo test

# Run property-based tests
cargo test --features quickcheck

# Check clippy and formatting
cargo clippy --all-targets
cargo fmt --check
```

## Best Practices

### Benchmarking
- Always run benchmarks in release mode for accurate performance measurements
- Use `black_box` to prevent compiler optimizations from eliminating benchmarked code
- Run benchmarks multiple times to account for system variability
- Compare against known-good baselines

### Assembly Analysis
- Use consistent targets (e.g., `aarch64-unknown-linux-gnu`) for reproducible results
- Compare assembly before and after optimizations
- Look for unexpected code generation (allocations where none expected)
- Consider both instruction count and instruction efficiency

### Performance Goals
- Maintain deterministic performance (no allocations in hot paths)
- Ensure fast-path code is truly fast (no unnecessary overhead)
- Preserve exact arithmetic fallbacks for correctness
- Keep code maintainable and readable

## Troubleshooting

### Benchmarks Not Running
```bash
# Ensure Criterion is available
cargo check --benches

# Run with verbose output
cargo bench --verbose
```

### Assembly Generation Failing
```bash
# Check function name syntax
cargo check

# Try with explicit target
cargo asm --target x86_64-unknown-linux-gnu --lib function_name
```

### Inconsistent Results
- Ensure system is idle during benchmarking
- Disable CPU frequency scaling
- Use consistent Rust toolchain versions
- Compare results across multiple runs
