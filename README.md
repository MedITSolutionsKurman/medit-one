# MedIT One

## A Blazingly Fast Single-Token Transformer with Hidden State Self-Attention

MedIT One is an innovative transformer architecture that dramatically improves inference speed and memory efficiency by leveraging a novel combination of techniques from both transformer and recurrent neural networks.

## Key Innovations

### Single-Token Prediction

Unlike traditional transformers that process and predict across the entire sequence length, MedIT One is optimized for single-token prediction:

- **Token-by-Token Processing**: Processes one token at a time during inference, eliminating the need for quadratic attention complexity across the entire sequence
- **Constant Memory Footprint**: Memory usage remains constant regardless of context length
- **State-Based Context Management**: Maintains rich contextual information without requiring the full sequence to be present in memory

### Hidden-State Self-Attention

MedIT One reimagines attention mechanisms to work at the hidden state level rather than the token level:

- **Attention on Hidden Representations**: Applies self-attention directly on hidden state representations rather than across tokens
- **Feature-Level Interactions**: Captures complex feature interactions without the computational burden of cross-token attention
- **Linear Scaling**: Computational complexity scales linearly with sequence length instead of quadratically

### Recurrent-Style State Management

The architecture maintains state vectors that evolve throughout the sequence:

- **State Vectors (hx, cx)**: Carries information forward like LSTM/GRU cells but enhanced with transformer-style processing
- **Continuous State Refinement**: States are continuously refined with each new token, preserving long-range dependencies
- **Context Compression**: Efficiently compresses contextual information into state representations

### Mixture of Experts Integration

MedIT One includes an optimized Mixture of Experts (MoE) implementation:

- **Dynamic Expert Selection**: Selectively activates different experts based on input features
- **Parallel Processing**: Processes expert computations in parallel for maximum efficiency
- **Numerical Stability**: Built-in safeguards ensure stable training and inference

### CUDA-Accelerated Operations

MedIT One now features custom CUDA kernels for maximum performance:

- **Optimized Flash Attention**: Specialized CUDA kernels achieve memory-efficient attention computation with numerical stability safeguards
- **Combined RoPE and Attention**: Fused kernels for rotary positional embeddings and attention calculation minimize memory transfers
- **Hardware-Optimized Computation**: Kernels are tuned for modern GPU architectures (Volta, Turing, Ampere, and Hopper)
- **Mixed Precision Support**: Full support for FP16/BF16 operations with automatic numerical stability handling

## Performance Benefits

- **Faster Inference**: Dramatically reduced computation needed per token
- **Constant Memory Usage**: Memory requirements don't increase with sequence length
- **Unbounded Context Length**: Theoretically unlimited context length with efficient state management
- **GPU Acceleration**: With CUDA Turbo mode enabled, achieves near-theoretical peak performance on supported hardware

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meditsolutions/one-small")
model = AutoModelForCausalLM.from_pretrained("meditsolutions/one-small")

# Generate text
input_text = "The single-token architecture provides"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=50, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

## How It Works

1. **Input Embedding**: Tokens are embedded and pooled to a fixed dimension
2. **State Initialization**: Initial hidden and cell states are created
3. **Hidden Processing**: The One module processes the states with self-attention on features
4. **State Update**: States are updated based on attention results
5. **Next Token Prediction**: The updated state is used to predict the next token
6. **State Preservation**: States are passed to the next inference step

## Technical Details

- **Architecture**: Hybrid transformer-recurrent design with MoE integration
- **Attention Mechanism**: Feature-level self-attention rather than token-level cross-attention
- **State Management**: Dual-state approach similar to LSTMs but with transformer-style processing
- **Numerical Stability**: Robust handling of numerical edge cases for reliable training
- **CUDA Kernels**: Custom-written high-performance CUDA kernels for critical operations:
  - Flash Attention with optimized memory access patterns
  - Rotary Position Embedding (RoPE) with fused operations
  - Mixture of Experts routing with parallel top-k selection
  - Combined RoPE + Attention operations for maximizing throughput

## Installation

```bash
# From PyPI
pip install medit-one

# From source with CUDA acceleration
git clone https://github.com/MedITSolutionsKurman/medit-one
cd medit-one
pip install -e .
```

### CUDA Acceleration

The CUDA Turbo backend is automatically enabled when installing from source if CUDA is available. To verify that the CUDA backend is active:

```python
from one.turbo_ops import TURBO_MODE
print(f"CUDA Turbo Mode: {'Enabled' if TURBO_MODE else 'Disabled'}")
```

## Citation

If you use MedIT One in your research or applications, please cite:

```bibtex
@software{medit_one,
  title = {MedIT One: A Blazingly Fast Single-Token Transformer with Hidden State Self-Attention},
  author = {MedIT Solutions Team},
  year = {2025},
  publisher = {MedIT Solutions},
  url = {https://github.com/MedITSolutionsKurman/medit-one}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.