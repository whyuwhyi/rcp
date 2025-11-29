# RCPFP32 - Hardware Reciprocal Function Implementation

A high-performance hardware implementation of the reciprocal function (`1/x`) for single-precision floating-point numbers (FP32), designed using Chisel HDL.

## Overview

This project implements **RCPFP32**, which computes `1/x` for FP32 inputs using a pipelined architecture combining lookup tables (LUT) and polynomial approximation to achieve efficient hardware implementation with high accuracy.

This project reuses the **XiangShan Fudian** floating-point unit library for basic FP32 arithmetic operations (multiplication, fused multiply-add).

## Algorithm

$$
\begin{equation}
\begin{aligned}
f               &= \frac{1}{x} \\
\\
\frac{1}{-x}    &= -\frac{1}{x}\\
\\
x               &= 2^{E} \times (1 + M) \\
\\
\frac{1}{x}     &= 2^{-E} \times \frac{1}{1 + M} \\
\\
M               &= M_{\text{high}} + M_{\text{low}} \\
\\
\frac{1}{1 + M} &= \frac{1}{1+M_{\text{high}}} \times \frac{1}{1 + \frac{M_{\text{low}}}{1+M_{\text{high}}}} \\
\\
T               &= \frac{1}{1+M_{\text{high}}} \\
                &= \text{LUT}[M_{\text{high}}] \\
\\
r               &= \frac{M_{\text{low}}}{1+M_{\text{high}}} \\
                &= M_{\text{low}} \times T \\
\\
\frac{1}{1+r}   &= \text{poly}(r) \\
\\
\text{poly}(r)  &= 1 - r + r^2 \\
                &= 1 + r(-1 + r)
\end{aligned}
\end{equation}
$$

### Algorithm Breakdown

- Extract sign bit and exponent $E$ from input
- Decompose mantissa $M$ into high 7 bits ($M_{\text{high}}$) and low 16 bits ($M_{\text{low}}$)
- Use $M_{\text{high}}$ to index lookup table: $T = \frac{1}{1+M_{\text{high}}/128}$ (128 entries)
- Compute normalized residual: $r = M_{\text{low}} \times T$
- Approximate $\frac{1}{1+r}$ using second-order Taylor polynomial, where $r \in [0, \frac{1}{2^{m+1}}]$
- Final result: $\frac{1}{x} = 2^{-E} \times T \times \text{poly}(r)$, with sign preserved

### Key Features

- **Sign handling**: Preserves input sign through computation
- **LUT-based coarse approximation**: 128-entry table for $\frac{1}{1+M_{\text{high}}/128}$
- **Polynomial refinement**: Second-order polynomial corrects for $M_{\text{low}}$
- **Special value support**: Handles NaN, Inf, and Zero correctly

## Hardware Design

### Pipeline Structure

```
S0: Input Filtering and Special Value Handling (FilterFP32)
    - Handle NaN, Inf, and Zero inputs
    - NaN → NaN
    - Zero → ±Inf (preserves sign)
    - ±Inf → ±Zero (preserves sign)

S1: Mantissa Decomposition (DecomposeRCP)
    - Extract sign, exponent E, mantissa M
    - Split M into:
      * M_high: High 7 bits (LUT index)
      * M_low: Low 16 bits converted to FP32

S2: Lookup Table (RCPLUT)
    - T = LUT[M_high] = 1/(1 + M_high/128)
    - 128-entry lookup table

S3: Compute Normalized Residual (MULFP32 - mulA)
    - r = M_low × T
    - Represents normalized error term

S4: Polynomial First Stage (CMAFP32 - cma0)
    - tmp = -1 + r
    - First stage of polynomial evaluation

S5: Polynomial Second Stage (CMAFP32 - cma1)
    - poly = 1 + r × tmp = 1 - r + r²
    - Completes polynomial approximation of 1/(1+r)

S6: Mantissa Reciprocal (MULFP32 - mulFinal)
    - rcpM = T × poly
    - Produces 1/(1+M)

S7: Exponent Adjustment and Result Assembly
    - Compute final exponent: 254 - E + rcpM_exp
    - Implements multiplication by 2^(-E)
    - Restore sign bit
    - If bypass condition met, output bypassVal; otherwise output result
```

### Key Modules

- **FilterFP32**: Input validation and special case handling
- **DecomposeRCP**: Mantissa decomposition into high and low parts
- **RCPLUT**: Lookup table for coarse reciprocal approximation
- **CMAFP32**: Fused multiply-add for polynomial computation
- **MULFP32**: FP32 multiplication using Fudian library

## Performance Results

### Accuracy

Verification against both CPU and GPU references on 1,000,000 test cases:

#### CPU Reference (C math library `1.0/x`)

```
=== CPU_Ref Statistics ===
Total: 1,000,000 test cases
Pass:  940,921 (94.09%)
Fail:  59,079 (5.91%)

Average Error: 5.887073e-08
Maximum Error: 4.991844e-07

Average ULP: 0.73
Maximum ULP: 8

Total Cycles: 1,000,089
Throughput:   1 result/cycle
```

#### GPU Reference (NVIDIA RTX 5060 with `-use_fast_math`)

```
=== GPU_Ref Statistics ===
Total: 1,000,000 test cases
Pass:  940,921 (94.09%)
Fail:  59,079 (5.91%)

Average Error: 5.887073e-08
Maximum Error: 4.991844e-07

Average ULP: 0.73
Maximum ULP: 8
```

### Timing

- **Maximum Frequency**: 1410 MHz
- **Timing Violations**: None
- **Critical Path**: (To be measured)

### Area

- **Total Area**: (To be measured)
- **LUT Count**: (To be measured)
- **Register Count**: (To be measured)

## Dependencies

### Required

- **Chisel 6.6.0**: Hardware description language
- **Scala 2.13.15**: Programming language for Chisel
- **Mill**: Build tool for Scala/Chisel projects
- **Verilator**: For simulation and verification
- **XiangShan Fudian**: Floating-point arithmetic library (included as git submodule)

### Optional

- **CUDA/NVCC**: For GPU-accelerated reference implementation (NVIDIA GPU required)
- **Synopsys Design Compiler**: For ASIC synthesis (TSMC N22 process)

## Building

### Initialize Dependencies

```bash
make init
```

This will initialize the XiangShan Fudian submodule.

### Generate SystemVerilog

```bash
# Generate RCPFP32 RTL
./mill --no-server RCPFP32.run
```

The generated SystemVerilog will be placed in `rtl/RCPFP32.sv`.

### Build and Run Simulation

```bash
make run
```

The build system automatically detects CUDA availability:

- **Without CUDA**: Uses CPU reference only (standard C library `1.0/x`)
- **With CUDA**: Uses both CPU and GPU references simultaneously
  - CPU Reference: Standard C library `1.0/x`
  - GPU Reference: NVIDIA CUDA math library with `-use_fast_math` flag
  - Both error statistics are computed and displayed for comparison

### Clean Build Artifacts

```bash
make clean
```

## Testing and Verification

### Simulation

Verilator-based testbench with:

- Comprehensive test vector generation (1M test cases)
- Random input generation across full FP32 range
- Special value testing (NaN, Inf, zero, subnormals)
- ULP (Unit in Last Place) error measurement
- Waveform generation (FST format) for debugging

### Reference Models

The testbench automatically uses available reference implementations:

- **CPU Reference**: Standard C library (`1.0/x`) - always available
- **GPU Reference**: NVIDIA CUDA math library with `-use_fast_math` - automatically enabled if CUDA is detected

When both references are available, error statistics are computed against both to provide comprehensive verification.

### Accuracy Metrics

- **ULP Error**: Measures floating-point accuracy in terms of "units in the last place"
- **Relative Error**: Standard floating-point error metrics
- **Pass/Fail**: Bit-exact comparison against reference implementation

## Future Improvements

- [ ] Reduce maximum ULP error to < 1 ULP average
- [ ] Optimize polynomial coefficients using Remez algorithm
- [ ] Add configurable rounding mode support

## Credits

- **XiangShan Fudian FPU Library**: Provides high-quality floating-point arithmetic components
  - Repository: <https://github.com/OpenXiangShan/fudian>
  - Used for: FMUL, FCMA_ADD, RawFloat utilities

## References

- IEEE Standard for Floating-Point Arithmetic (IEEE 754-2008)
- XiangShan Fudian FPU: <https://github.com/OpenXiangShan/fudian>
- Chisel/FIRRTL Documentation: <https://www.chisel-lang.org/>
- Handbook of Floating-Point Arithmetic (Muller et al.)
- "Elementary Functions: Algorithms and Implementation" (Muller, 2006)

## License

This project reuses the XiangShan Fudian library. Please refer to the respective license files in the `dependencies/fudian` directory for licensing terms.
