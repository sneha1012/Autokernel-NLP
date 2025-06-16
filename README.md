# AutoKernel++

AutoKernel++ is a profiler-guided kernel tuning framework for ML workloads. It optimizes CUDA kernel launch configurations using Bayesian search, visualizes performance improvements, and is extendable to real-world kernels like FlashAttention and LayerNorm from transformer-based models.

---

## Features

- Profile and optimize custom CUDA kernels (e.g., GEMM, conv2d)
- Apply Bayesian Optimization over launch configs (block/grid/shared mem)
- Visualize throughput, latency, and occupancy using Nsight Compute
- Tune FlashAttention and LayerNorm kernels from BERT/LLM models
- Containerized pipeline with full reproducibility (Docker)
- Interactive dashboard for upload, profiling, and tuning

---

## Modules

1. **Kernels**
   - `matmul.cu`, `conv2d.cu`, `layernorm.cu`, `flashattn.cu` (sourced or custom)

2. **Profiler**
   - Uses Nsight CLI or `nv-nsight-cu` to log:
     - occupancy, memory throughput, L2 cache hit, execution time

3. **Tuner**
   - Generates launch configs (blockDim, gridDim)
   - Bayesian optimization using `scikit-optimize`
   - Reward = throughput / latency, regularized by memory use

4. **Dashboard**
   - Streamlit frontend
   - Upload kernels, choose parameter ranges
   - Output: config-vs-performance heatmaps, best config logs

5. **Docker**
   - CUDA SDK, Nsight, Python packages
   - One-command setup for remote/GPU machines

---

## Extension: NLP Kernel Tuning

New Kernels:
- **LayerNorm**: widely used in BERT, GPT
- **FlashAttention**: key kernel in efficient transformer inference

Tuning Targets:
- Optimize block sizes for LayerNorm on different sequence lengths
- Profile memory coalescing + shared memory for FlashAttention

Visual Output:
- Comparison: baseline vs tuned GPU time (per layer)
- Visual dashboard for transformer-layer optimization

---

## Learning Outcomes

- Understand GPU launch config space for ML kernels
- Use Bayesian Optimization for systems-level tuning
- Learn Nsight Compute CLI for performance profiling
- Interpret memory throughput and latency tradeoffs in LLM kernels
- Extend tuning logic to real-world transformer workloads

---

## Timeline 

- Week 1: Baseline kernels (matmul, conv2d), profiler logging
- Week 2: Bayesian tuner loop, FlashAttention + LayerNorm integration
- Week 3: Streamlit UI, Docker pipeline
- Week 4: Evaluation, final benchmarks, blog + resume + GitHub polish

---

## Future Work

- RL-based tuner (PPO) to adapt at runtime
- AutoTVM-style scheduling + tiling engine
- Extend to custom model blocks via Triton + TVM integration
