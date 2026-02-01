# Epylabel Migration Plan: Pure Python and Pure R Versions

## Overview

This document outlines the plan to split the current hybrid Python/R implementation into two independent, pure implementations:
- **epylabel-py**: Pure Python implementation
- **epylabel-r**: Pure R implementation

Both implementations must produce identical results for the same inputs.

## Current Architecture

```
epylabel/
├── labeler.py          # All labeling algorithms (Python + rpy2 for BCP)
├── pipeline.py         # Pipeline orchestration (Python)
├── utils.py            # Data transformation utilities (Python)
├── metrics.py          # Summary statistics (Python)
├── plot.py             # Visualization (Python/Plotly)
└── wavefinder/         # Wave detection (Python, ported from R)
```

### Current Dependencies on R
- **BCP (Bayesian Change Point)**: Uses `rpy2` to call R's `bcp` package
- **WaveFinder**: Already pure Python (originally ported from R)

## Migration Strategy

### Phase 1: Test Infrastructure (Current Sprint)
- [ ] Create comprehensive test datasets with known properties
- [ ] Implement golden master (snapshot) tests
- [ ] Generate reference outputs for all algorithms
- [ ] Create cross-validation framework

### Phase 2: Pure Python Implementation
- [ ] Replace R's `bcp` with Python alternative:
  - Option A: `ruptures` library (change point detection)
  - Option B: `bayesian-changepoint` or custom implementation
  - Option C: Port BCP algorithm directly to Python/NumPy
- [ ] Validate Python BCP produces equivalent results
- [ ] Remove rpy2 dependency

### Phase 3: Pure R Implementation
- [ ] Port all Python algorithms to R:
  - Changerate
  - ExponentialGrowth
  - Shapelet
  - GapFiller
  - Ensemble
  - WaveFinder (port back to R)
  - Pipeline
  - Metrics
- [ ] Create R package structure
- [ ] Validate R implementation produces equivalent results

### Phase 4: Synchronization & Maintenance
- [ ] Shared test data repository
- [ ] Automated cross-validation in CI/CD
- [ ] Version synchronization protocol

## Algorithm-by-Algorithm Migration

### 1. Changerate
**Current**: Python (NumPy/Pandas)
**Complexity**: Low
**Migration Notes**: Simple rolling calculation, straightforward to implement in both languages.

```python
# Python implementation exists
def transform(data):
    # Rolling change rate with ceiling
```

```r
# R equivalent needed
changerate <- function(data, n_days, ceiling) {
  # Implementation
}
```

### 2. BCP (Bayesian Change Point)
**Current**: R via rpy2
**Complexity**: High
**Migration Notes**: Core algorithm uses MCMC sampling. Consider:

**Python Options:**
1. `ruptures` library - different algorithm but similar purpose
2. Custom implementation using PyMC or NumPyro
3. Direct port of R's bcp C++ code

**R**: Already implemented (bcp package)

### 3. Shapelet
**Current**: Python (SciPy correlation)
**Complexity**: Medium
**Migration Notes**: Uses exponential curve fitting and correlation.

### 4. WaveFinder
**Current**: Python (ported from R)
**Complexity**: High
**Migration Notes**: Complex algorithm with multiple sub-algorithms. Already Python, needs R port.

### 5. ExponentialGrowth
**Current**: Python (SciPy linregress)
**Complexity**: Low
**Migration Notes**: Linear regression on log-transformed data.

### 6. GapFiller
**Current**: Python (NumPy)
**Complexity**: Low
**Migration Notes**: Simple gap filling logic.

### 7. Ensemble
**Current**: Python
**Complexity**: Low
**Migration Notes**: Majority voting, straightforward in both languages.

## Test Strategy

### Test Categories

1. **Unit Tests**: Individual function behavior
2. **Integration Tests**: Pipeline combinations
3. **Golden Master Tests**: Snapshot comparison against reference outputs
4. **Cross-Validation Tests**: Python vs R comparison
5. **Property-Based Tests**: Invariants that must hold

### Test Data Requirements

| Dataset | Purpose | Characteristics |
|---------|---------|-----------------|
| `simple_wave.parquet` | Basic wave detection | Single clear wave |
| `multi_wave.parquet` | Multiple waves | 3-5 distinct waves |
| `noisy.parquet` | Noise handling | High variance data |
| `edge_cases.parquet` | Boundary conditions | Empty, single point, etc. |
| `real_covid_sample.parquet` | Real-world validation | Actual COVID-19 data subset |

### Golden Master Format

```
test_data/
├── inputs/
│   ├── simple_wave.parquet
│   ├── multi_wave.parquet
│   └── ...
├── expected_outputs/
│   ├── python/
│   │   ├── bcp/
│   │   │   ├── simple_wave_labels.parquet
│   │   │   └── simple_wave_summary.parquet
│   │   ├── shapelet/
│   │   └── ...
│   └── r/
│       └── ...
└── cross_validation/
    └── comparison_report.json
```

## Acceptance Criteria

### For Each Algorithm

1. **Functional Equivalence**: Given identical inputs, outputs must match within tolerance
2. **API Compatibility**: Same function signatures and parameter names
3. **Performance**: Comparable execution time (within 2x)
4. **Documentation**: Equivalent docstrings/roxygen

### Tolerance Definitions

| Output Type | Tolerance |
|-------------|-----------|
| Boolean labels | Exact match |
| Numeric values | Relative error < 1e-6 |
| Dates/timestamps | Exact match |
| Summary statistics | Relative error < 1e-4 |

## Repository Structure (Future)

```
epylabel/                    # Monorepo
├── python/                  # Pure Python package
│   ├── epylabel/
│   ├── tests/
│   └── pyproject.toml
├── r/                       # Pure R package
│   ├── R/
│   ├── tests/
│   └── DESCRIPTION
├── test_data/               # Shared test data
│   ├── inputs/
│   └── expected_outputs/
├── cross_validation/        # Cross-language validation
│   └── compare.py
└── docs/                    # Shared documentation
```

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Test Infrastructure | 1-2 weeks | None |
| Phase 2: Pure Python | 2-4 weeks | Phase 1 |
| Phase 3: Pure R | 4-6 weeks | Phase 1 |
| Phase 4: Synchronization | Ongoing | Phase 2, 3 |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| BCP algorithm differences | High | Extensive validation, document acceptable variance |
| Floating point precision | Medium | Define tolerance, use deterministic seeds |
| Missing R packages | Low | Document dependencies, use CRAN only |
| Maintenance burden | Medium | Automated cross-validation, shared tests |

## Next Steps

1. **Immediate**: Implement comprehensive test suite (this PR)
2. **Short-term**: Research Python BCP alternatives
3. **Medium-term**: Begin R package structure
4. **Long-term**: CI/CD pipeline for cross-validation
