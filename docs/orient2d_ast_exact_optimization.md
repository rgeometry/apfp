# Performance Analysis: `orient2d_ast_exact` Optimization

## Executive Summary

Current baseline performance shows `orient2d_ast_exact` is **180x slower** than `geometry_predicates` for regular points and **60x slower** for collinear points. The goal is to reduce this gap while maintaining correctness.

## Baseline Performance

### Random Points Benchmark
- `orient2d_fast`: 4.66 µs
- `orient2d_geometry_predicates`: 5.23 µs
- `orient2d_ast_exact`: 939.63 µs (**180x slower than geometry_predicates**)

### Collinear Points Benchmark (challenging cases)
- `orient2d_fast`: 4.95 µs
- `orient2d_geometry_predicates`: 24.98 µs (**5x regression vs random points**)
- `orient2d_ast_exact`: 1.52 ms (**3x regression vs random points, 60x slower than geometry_predicates**)

## Analysis Areas

### 1. Assembly Code Analysis
### 2. Algorithm Analysis
### 3. Optimization Opportunities

## Current Implementation Issues

The current `orient2d_exact` function:
1. Uses a massive 4KB+ stack buffer (`BUFFER_SIZE: usize = 512`)
2. Performs complex expression tree evaluation with multiple expansion arithmetic operations
3. Calls heap-allocated `expansion_product` for multiplications
4. Has deeply nested control flow for sign determination

### Key Problems Identified

1. **Stack Allocation Size**: 4KB buffer allocation per call
2. **Heap Allocations**: `expansion_product` allocates vectors on the heap
3. **Complex Expression Evaluation**: Multiple nested operations instead of optimized determinant calculation
4. **Inefficient Sign Computation**: Full expansion arithmetic when simpler comparisons might suffice

## Optimization Strategies

### Strategy 1: Reduce Buffer Usage
- Pre-calculate required buffer sizes for each subexpression
- Use smaller, more targeted buffers instead of monolithic allocation

### Strategy 2: Eliminate Heap Allocations
- Implement fully stack-allocated `expansion_product_stack`
- Avoid all heap allocations in hot path

### Strategy 3: Simplify Expression Tree
- Compute determinant more directly instead of building complex AST
- Fuse operations where possible

### Strategy 4: Fast Path for Common Cases
- Add early exit for obviously non-collinear points
- Use floating-point filters before expansion arithmetic

## Implementation Plan

1. **Phase 1**: Implement stack-allocated product operations
2. **Phase 2**: Optimize buffer allocation strategy
3. **Phase 3**: Simplify determinant computation
4. **Phase 4**: Add fast paths and filters

## Benchmark Tracking

## Optimization Results

### Phase 1: Stack-Allocated Product ✅ COMPLETED
**Result**: 2.4x speedup (939.63 µs → 389.13 µs for random points)
**New performance**: ~74x slower than geometry_predicates (vs 180x before)
**Collinear points**: 3.3x speedup (1.52 ms → 466.79 µs)

### Phase 2: Fast Path Filtering ✅ COMPLETED
**Result**: 38x speedup (389.13 µs → 10.28 µs for random points)
**New performance**: Only ~1.8x slower than geometry_predicates (10.28 µs vs 5.67 µs)
**Collinear points**: Maintained performance (458.68 µs vs geometry_predicates 25.53 µs)

### Phase 3: Optimized Buffer Allocation ✅ COMPLETED
**Result**: Only allocate large buffers when actually needed for exact computation
**Impact**: Reduces stack usage for fast-path cases

## Final Performance Summary

### Random Points (easy cases)
- **Before optimization**: 939.63 µs (~166x slower than geometry_predicates)
- **After optimization**: 10.28 µs (~1.8x slower than geometry_predicates)
- **Total improvement**: **91x speedup**

### Collinear Points (hard cases)
- **Before optimization**: 1.52 ms (~60x slower than geometry_predicates)
- **After optimization**: 458.68 µs (~18x slower than geometry_predicates)
- **Total improvement**: **3.3x speedup**

### Key Insights
- **Adaptive precision is crucial**: Using fast floating-point filters before expensive exact arithmetic gives massive speedups
- **Most cases are easy**: The fast filter succeeds for almost all random points, avoiding expensive computation
- **Stack allocation matters**: Eliminating heap allocations provided significant speedup
- **Collinear cases remain challenging**: When points are nearly collinear, full expansion arithmetic is still needed

### Optimization Techniques Applied
1. **Stack-allocated expansion arithmetic**: Eliminated heap allocations in product operations
2. **Fast path filtering**: Use floating-point filter before expensive exact computation
3. **Lazy buffer allocation**: Only allocate large buffers when actually needed
4. **Maintained correctness**: All optimizations preserve exact arithmetic guarantees

## Detailed Analysis

### Assembly Code Issues

The current assembly shows:
- Massive stack frame setup (128 bytes of register saves)
- Complex buffer management with multiple sub-buffer splits
- Heap allocation calls (`__rustc::__rust_alloc`, `__rustc::__rust_dealloc`)
- Deeply nested conditional logic for sign determination

### Algorithm Issues

The current algorithm:
1. Builds full expression tree: `(ax - cx) * (by - cy) - (ay - cy) * (bx - cx)`
2. Evaluates each subexpression to full expansions
3. Computes products using heap-allocated arithmetic
4. Only then determines the sign

**Opportunity**: We can compute the determinant more directly using expansion arithmetic primitives without building the full expression tree.

### Comparison with geometry_predicates

The `geometry_predicates` crate likely:
- Uses more efficient expansion arithmetic implementation
- Has better optimized primitives for common operations
- Avoids unnecessary heap allocations
- Has more aggressive fast paths

## Next Steps

1. Implement fully stack-allocated expansion product
2. Profile the impact
3. Explore direct determinant computation
4. Add floating-point fast paths
5. Compare with geometry_predicates assembly
