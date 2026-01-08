# Knowledge Editing Experiments - Summary Report

Generated: 2026-01-06 07:28:31

================================================================================

## Executive Summary
--------------------------------------------------------------------------------
- **Total Heuristics Analyzed:** 5
- **Completed Experiments:** 4
- **Incomplete Experiments:** 1


## BASELINE
--------------------------------------------------------------------------------
**Status:** ⚠ Baseline only (no editing results)

### Baseline Performance
- **Correctness:** 34.7%
- **Top-1 Share:** 74.0%
- **Entropy:** 1.166
- **Diversity:** 17.3%
- **Problems with Top-1 Correct:** 2/5

## MODULAR MULT
--------------------------------------------------------------------------------
**Status:** ✓ Complete

### Performance Metrics

| Metric | Baseline | After Editing | Change |
|--------|----------|---------------|--------|
| **Correctness** | 37.5% | 27.5% | ✗ -10.0% |
| **Top-1 Share** | 77.5% | 53.8% | ✗ -23.8% |
| **Entropy** | 0.964 | 1.831 | ✗ -0.867 |
| **Diversity** | 23.8% | 37.5% | ↑ +13.8% |

**Problems Tested:** 5
**Top-1 Correct (Baseline):** 2/5
**Top-1 Correct (Edited):** 2/5

### Interpretation
- ✗ **Significant degradation** in correctness (-10.0%)
- ✗ **Increased entropy** (less consistent responses)
- ✗ **Lower consensus** on most frequent response

## MODULAR ADD
--------------------------------------------------------------------------------
**Status:** ✓ Complete

### Performance Metrics

| Metric | Baseline | After Editing | Change |
|--------|----------|---------------|--------|
| **Correctness** | 38.8% | 23.2% | ✗ -15.6% |
| **Top-1 Share** | 77.6% | 45.2% | ✗ -32.4% |
| **Entropy** | 1.170 | 2.475 | ✗ -1.304 |
| **Diversity** | 14.8% | 24.8% | ↑ +10.0% |

**Problems Tested:** 5
**Top-1 Correct (Baseline):** 2/5
**Top-1 Correct (Edited):** 2/5

### Interpretation
- ✗ **Significant degradation** in correctness (-15.6%)
- ✗ **Increased entropy** (less consistent responses)
- ✗ **Lower consensus** on most frequent response

## AM GM
--------------------------------------------------------------------------------
**Status:** ✓ Complete

### Performance Metrics

| Metric | Baseline | After Editing | Change |
|--------|----------|---------------|--------|
| **Correctness** | 38.4% | 27.2% | ✗ -11.2% |
| **Top-1 Share** | 79.2% | 52.4% | ✗ -26.8% |
| **Entropy** | 1.092 | 2.483 | ✗ -1.391 |
| **Diversity** | 14.4% | 28.0% | ↑ +13.6% |

**Problems Tested:** 5
**Top-1 Correct (Baseline):** 2/5
**Top-1 Correct (Edited):** 2/5

### Interpretation
- ✗ **Significant degradation** in correctness (-11.2%)
- ✗ **Increased entropy** (less consistent responses)
- ✗ **Lower consensus** on most frequent response

## CAUCHY
--------------------------------------------------------------------------------
**Status:** ✓ Complete

### Performance Metrics

| Metric | Baseline | After Editing | Change |
|--------|----------|---------------|--------|
| **Correctness** | 37.6% | 32.0% | ✗ -5.6% |
| **Top-1 Share** | 80.0% | 54.4% | ✗ -25.6% |
| **Entropy** | 0.939 | 2.264 | ✗ -1.325 |
| **Diversity** | 11.2% | 23.6% | ↑ +12.4% |

**Problems Tested:** 5
**Top-1 Correct (Baseline):** 2/5
**Top-1 Correct (Edited):** 2/5

### Interpretation
- ✗ **Significant degradation** in correctness (-5.6%)
- ✗ **Increased entropy** (less consistent responses)
- ✗ **Lower consensus** on most frequent response

================================================================================
## Summary Statistics
--------------------------------------------------------------------------------

**Completed Experiments:** 4
- Improved correctness: 0
- Degraded correctness: 4
- No change: 0

**Average correctness change:** -10.60%

**Experiments with Degraded Correctness:**
- modular_mult: -10.0%
- modular_add: -15.6%
- am_gm: -11.2%
- cauchy: -5.6%

================================================================================

**Note:** This summary is based on aggregate metrics across multiple problems for each heuristic.
