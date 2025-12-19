# HPML Project: An Analysis of Radial Attention

## Team Information
- **Team Name**: Radial Attention
- **Members**:
  - Nandini Prasad (nnp2119)
  - Hanz Tan (yt2924)

---

## 1. Problem Statement
State-of-the-art Video Diffusion Transformers (DiTs), such as Wan 2.1, rely on "dense" self-attention mechanisms where every token attends to every other token. This results in quadratic computational complexity ($O(N^2)$) regarding sequence length, creating severe bottlenecks for memory and latency when generating long videos. This project investigates "Radial Attention," a sparse attention mechanism ($O(N \log N)$) based on spatiotemporal energy decay, to address these scaling limitations. We implemented and profiled this mechanism on the Wan 2.1 1.3B model to validate efficiency claims versus standard dense attention.

---

## 2. Model Description
We utilized the Wan 2.1 1.3B Text-to-Video (T2V) model.
- **Framework**: PyTorch, Hugging Face Diffusers Library.
- **Architecture**: Diffusion Transformer (DiT) with a 3D Causal Variational Autoencoder (VAE).
- **Customization**:
  - We patched the `wan2.1-radial` attention interface to ensure compatibility with the modern Hugging Face diffusers library, fixing tensor layout incompatibilities in the reference implementation.
  - We replaced the standard dense attention layers with a static masking strategy (Radial Attention) that limits attention to a "radial" neighborhood of adjacent tokens.

---

## 3. Final Results Summary

Benchmarking was conducted on a single NVIDIA GeForce RTX 5090. The comparison is between the baseline Dense Attention and our implemented Radial Attention.

| Metric (241 Frames) | Value (Radial) | Value (Dense) | Improvement |
|---------------------|----------------|---------------|-------------|
| Inference Latency   | 343.91 s | 627.43 s | **1.82x Faster** |
| Peak VRAM Usage     | 20.70 GB | 20.70 GB | Neutral |
| Speedup (59 Frames) | -              | -             |  Radial 1.37x Faster |
| SM Cycle Count      | ~9M | ~30M | ~70% Reduction |

**Key Findings:**
- **Latency:** Speedup scales positively with sequence length, peaking at 1.82x for 241 frames.
- **Memory:** Peak VRAM consumption remained identical.
- **Compute:** Profiling via NVIDIA Nsight Compute confirmed the speedup is driven by a reduction in FLOPs rather than memory bandwidth utilization improvements.

---

## 4. Reproducibility Instructions

### A. Requirements

1. **Environment**: We use a containerized environment. Download the NVIDIA CUDA container for CUDA 12.8 (Ubuntu).
2. **Setup**: Run the provided setup script to install dependencies.

```bash
# Inside the container
bash container_setup.sh
```
### Quickstart

```
# Standard lengths
bash scripts/wan_t2v_inference_1_3B.sh

# Long length
bash scripts/wan_t2v_inference_1_3B_long.sh
```



### Profling

Run the command inside the script of your choosing with ncu.

