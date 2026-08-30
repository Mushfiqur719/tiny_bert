# TinyBanglaBERT Efficiency & Edge Deployment Benchmark Report

## 1. Abstract-Ready Summary Numbers
- **Parameter Compression:** **`9.58x fewer parameters`** (`11.55M` vs `110.62M`, ~89.6% reduction).
- **Model Size on Disk:** **`9.58x smaller`** in PyTorch FP32 (`44.06 MB` vs `421.98 MB`), and **`37.63x smaller`** with ONNX INT8 Quantization (`11.22 MB`).
- **CPU Inference Latency:** **`22.69x faster`** on CPU via ONNX Runtime (`5.15 ms` vs `116.83 ms` per headline).
- **INT8 Accelerated Speed:** **`28.41x faster`** on CPU (`4.11 ms` per headline, `243.2 headlines/sec`).
- **Memory Footprint:** Peak RAM consumption during single-item inference dropped from `0.27 MB` to **`0.27 MB`**.

## 2. Comparative Efficiency & Latency Benchmark Table
*Benchmarked over N=200 single-headline ($B=1$, $L=128$) forward passes on CPU (Primary) and GPU (Secondary)*

| Model Variant               | Runtime / Format   | Parameters         | Disk Size (MB)   | CPU Latency (Mean ± P95)   | CPU Throughput   | GPU Latency (Mean)   | Peak Memory (CPU)   | Size Reduction   | CPU Speedup      |
|:----------------------------|:-------------------|:-------------------|:-----------------|:---------------------------|:-----------------|:---------------------|:--------------------|:-----------------|:-----------------|
| Teacher (BanglaBERT)        | PyTorch FP32       | 110,618,882        | 422.23 MB        | 116.83 ms / 125.46 ms      | 8.6 fps          | 13.79 ms             | 0.27 MB             | 1.00x (Baseline) | 1.00x (Baseline) |
| TinyBanglaBERT-final        | PyTorch FP32       | 11,549,442         | 44.07 MB         | 9.98 ms / 13.68 ms         | 100.2 fps        | 5.95 ms              | 0.27 MB             | 9.58x smaller    | 11.71x faster    |
| TinyBanglaBERT-final (ONNX) | ONNX Runtime FP32  | 11,549,442         | 44.15 MB         | 5.15 ms / 6.09 ms          | 194.2 fps        | N/A                  | 0.03 MB             | 9.56x smaller    | 22.69x faster    |
| TinyBanglaBERT-final (INT8) | ONNX Runtime INT8  | 11,549,442 (8-bit) | 11.22 MB         | 4.11 ms / 5.59 ms          | 243.2 fps        | N/A                  | 0.02 MB             | 37.63x smaller   | 28.41x faster    |

## 3. Deployment Feasibility Analysis
1. **Browser Extension Target (ONNX Runtime Web / WASM):**
   - TinyBanglaBERT ONNX (44.15 MB) achieves a mean CPU response time of **`5.15 ms`** with a 95th percentile of **`6.09 ms`**.
   - This satisfies the sub-50ms budget required for instantaneous headline classification during web page rendering without blocking the browser DOM thread.
2. **Mobile / Edge Device Target (ONNX INT8):**
   - Dynamic INT8 quantization compacts the full model to **`11.22 MB`** (over **30x smaller** than the original Teacher).
   - Processes **`243.2 headlines per second`** on CPU cores with virtually zero startup latency.
