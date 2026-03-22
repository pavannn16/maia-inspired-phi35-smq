# Scale Metadata Quantization (SMQ) — Formal Specification

Frozen: 2026-03-22.  Do not modify after C3/C4/C5 runs begin.

---

## Motivation

Standard per-group W4 (AWQ, GPTQ, bitsandbytes NF4) stores per-group scales as
FP16 — effectively 14 mantissa bits.  For large MLP weight matrices the number
of scale values is `(out × in) / group_size`.  For Phi-3.5 MLP layers
(typical `out=3072, in=3072, group_size=128`), this is 73,728 FP16 scale values
= **144 KB per layer** of metadata.

Research question: how many mantissa bits do per-group scales actually need?

---

## Method: E5Mx Scale Representation

Scale values are quantized to a mini-float with:
- 1 sign bit
- 5 exponent bits  (same as FP8-E5 family; saturates at ~57344)
- `scale_mbits` mantissa bits

Total bits per scale: `6 + scale_mbits`.

### Scale quantization algorithm

For a group scale $s > 0$:

$$e = \lfloor \log_2 s \rfloor, \quad m = s / 2^e - 1 \in [0, 1)$$

$$\hat{m} = \frac{\lfloor m \cdot 2^{k} + 0.5 \rfloor}{2^k}, \quad k = \text{scale\_mbits}$$

$$\hat{s} = 2^{\hat{e}} \cdot (1 + \hat{m})$$

where $\hat{e} = e + 1$ if $\hat{m} \geq 1$ (carry), else $\hat{e} = e$.

For `scale_mbits=0`: $\hat{m} = 0$, so $\hat{s} = 2^e$ (binary rounding).
For `scale_mbits=-1`: $\hat{s} = s$ (exact, no quantization).

### Weight quantization (symmetric int4, per group)

Given weight matrix $W \in \mathbb{R}^{M \times N}$, group size $G = 128$:

$$s_g = \frac{\max |W_g|}{7} \quad \text{(group } g \text{ contains } G \text{ consecutive elements along } N\text{)}$$

$$\hat{s}_g = \mathrm{SMQ}(s_g, k) \quad \text{(apply E5Mx quantization)}$$

$$q_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{w_i}{\hat{s}_g}\right), -8, 7\right)$$

### Packing convention

Two int4 values packed per uint8 byte:
- Low nibble (bits 0–3) → even column index
- High nibble (bits 4–7) → odd column index

Signed representation: unsigned nibble $u \in [0,15]$ maps to signed
$q = u - 16$ if $u \geq 8$, else $q = u$.  (Two's complement lower nibble.)

### Dequantization

$$\hat{w}_i = q_i \cdot \hat{s}_g$$

---

## Fixed hyperparameters (frozen)

| Parameter    | Value | Rationale |
|---|---|---|
| `group_size` | 128   | Standard for W4; balances accuracy and overhead |
| `quant_target` | MLP layers (default), all layers (C7) | Stability; C7 explores full-model quantization |
| Rounding     | Round-nearest-even | PyTorch default |
| Saturation   | Clamp to $[-8, 7]$ | int4 symmetric range |

---

## Ablation axis

| Config | `scale_mbits` | Scale bits | Scale storage vs FP16 |
|---|---|---|---|
| C3     | -1 (exact)    | 32 (FP32)  | +2× baseline |
| C5-a   | 0             | 6          | -57% |
| C5-b   | 3             | 9          | -38% |
| C4/C5-c| 5             | 11         | -21% |

---

## Expected error bounds (from unit tests)

- `scale_mbits=5`: relative MSE < 0.5% vs exact; cosine similarity > 0.99
- `scale_mbits=3`: relative MSE < 1.5% vs exact
- `scale_mbits=0`: relative MSE < 5%; quality impact measured in C5 ablation

---

## Relationship to prior work

- **AWQ / GPTQ**: use exact FP16 or FP32 per-group scales.  SMQ is orthogonal —
  it can be applied on top of any calibration-based quant.
- **FP8 scales in Maia / NV Blackwell**: hardware-native E4M3/E5M2 scale tensors;
  SMQ is a software proxy for this concept on A100 BF16 paths.
- **Double quantization (bitsandbytes, C6)**: quantizes scales in 8-bit blocks;
  SMQ uses a formal floating-point framework (E5Mx) for principled ablation.
  C6 is included as a direct comparison baseline.
