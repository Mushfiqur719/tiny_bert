# Phase L — Unified ONNX Efficiency & Cascade RTT Report

**Hardware:** `AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD` | **OS:** `Windows 11` | **ONNX Runtime:** `1.29.0`  
**Protocol:** Batch = 1, `max_length = 64`, 200 warm-up + 1000 timed runs.

## 1. Model × Precision × Thread Benchmark

| model                        | precision   |   threads |   p50_ms |   p95_ms |   mean_ms |   disk_mb | cpu                                                | os         | ort_version   |
|:-----------------------------|:------------|----------:|---------:|---------:|----------:|----------:|:---------------------------------------------------|:-----------|:--------------|
| BanglaBERT-Base (110.6M)     | FP32        |         1 |   190.56 |   211.9  |    183.52 |    422.25 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| BanglaBERT-Base (110.6M)     | FP32        |         4 |    74.46 |    88.78 |     74.51 |    422.25 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| BanglaBERT-Base (110.6M)     | INT8        |         1 |    99.67 |   133.66 |     98.82 |    177.75 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| BanglaBERT-Base (110.6M)     | INT8        |         4 |    28.79 |    35.67 |     29.39 |    177.75 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| BanglaBERT-Small-12L (13.7M) | FP32        |         1 |    15.06 |    16.89 |     15.29 |     52.68 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| BanglaBERT-Small-12L (13.7M) | FP32        |         4 |     8.72 |    10.36 |      8.86 |     52.68 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| BanglaBERT-Small-12L (13.7M) | INT8        |         1 |    10.18 |    10.93 |     10.26 |     25.59 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| BanglaBERT-Small-12L (13.7M) | INT8        |         4 |     5.56 |     6.97 |      5.69 |     25.59 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| TinyBanglaBERT-4L (7.42M)    | FP32        |         1 |     5.17 |     5.95 |      5.26 |     28.41 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| TinyBanglaBERT-4L (7.42M)    | FP32        |         4 |     2.89 |     3.77 |      2.95 |     28.41 | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| TinyBanglaBERT-4L (7.42M)    | INT8        |         1 |     3.48 |     4.07 |      3.55 |     19.2  | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |
| TinyBanglaBERT-4L (7.42M)    | INT8        |         4 |     1.77 |     2.39 |      1.86 |     19.2  | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Windows 11 | 1.29.0        |

## 2. Edge–Cloud Cascade Effective Latency (with 50 ms & 150 ms Network RTT)

| escalation_rate   |   network_rtt_ms |   edge_p50_ms |   cloud_p50_ms |   effective_avg_ms |
|:------------------|-----------------:|--------------:|---------------:|-------------------:|
| 0%                |                0 |          1.77 |          74.46 |               1.77 |
| 0%                |               50 |          1.77 |          74.46 |               1.77 |
| 0%                |              150 |          1.77 |          74.46 |               1.77 |
| 10%               |                0 |          1.77 |          74.46 |               9.22 |
| 10%               |               50 |          1.77 |          74.46 |              14.22 |
| 10%               |              150 |          1.77 |          74.46 |              24.22 |
| 25%               |                0 |          1.77 |          74.46 |              20.38 |
| 25%               |               50 |          1.77 |          74.46 |              32.88 |
| 25%               |              150 |          1.77 |          74.46 |              57.88 |
| 50%               |                0 |          1.77 |          74.46 |              39    |
| 50%               |               50 |          1.77 |          74.46 |              64    |
| 50%               |              150 |          1.77 |          74.46 |             114    |
| 100%              |                0 |          1.77 |          74.46 |              76.23 |
| 100%              |               50 |          1.77 |          74.46 |             126.23 |
| 100%              |              150 |          1.77 |          74.46 |             226.23 |
