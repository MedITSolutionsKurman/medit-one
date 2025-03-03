# coding=utf-8
# Copyright 2025 The MedIT Solutions Kurman i Wspolnicy Sp. z o. o. Team.
# https://meditsolutions.pl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import math

from torch import nn
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.activations import ACT2FN

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from one.flash_attention import (
    OneFlashAttention,
    apply_rotary_pos_emb_flash,
)
from one.configuration_one import OneConfig
from typing import Optional, Tuple, Union
from einops import rearrange
from logging import getLogger
from one.linear_moe import LinearMoE

# Import the turbo operations
try:
    from one.turbo_ops import (
        TurboFlashAttention,
        turbo_rotary_embedding,
        turbo_cumsum_and_normalize,
        TURBO_MODE,
    )
except ImportError:
    TURBO_MODE = False

logger = getLogger()


class OneRMSNorm(nn.Module):
    def __init__(self, dim: int, bias: bool = False, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else 0

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype) + self.bias

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# This is a copy of the LlamaRotaryEmbedding class from transformers
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Licensed under the Apache License, Version 2.0
class OneRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[OneConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Use turbo implementation if available
        # if TURBO_MODE and x.is_cuda:
        #     return turbo_rotary_embedding(x, position_ids, self.inv_freq)

        # Original implementation
        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# This is a copy of the LlamaMLP class from transformers
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Licensed under the Apache License, Version 2.0
class OneMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.transformer_dim
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class One(nn.Module):
    def __init__(
        self,
        config: OneConfig,
        dropout: float = 0.1,
        layer_idx: int = 0,
        with_hx: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.hidden_size = config.transformer_dim
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale_factor = config.scale_factor
        self.layer_idx = layer_idx
        self.with_hx = with_hx
        self.use_single_token = config.use_single_token

        self.scaled_dim = self.num_attention_heads * self.head_dim

        self.norm1 = OneRMSNorm(self.hidden_size)
        self.rotary_pos_emb = OneRotaryEmbedding(config=config)

        self.lin1 = nn.Linear(self.hidden_size, self.scaled_dim)
        self.lin2 = nn.Linear(self.hidden_size, self.scaled_dim)
        self.lin3 = nn.Linear(self.hidden_size, self.scaled_dim)

        # Add proper scaling factor for attention stability
        self.attention_scale = 1.0 / math.sqrt(self.head_dim)

        self.lin_q = nn.Linear(self.head_dim, self.num_attention_heads)
        self.lin_k = nn.Linear(self.head_dim, self.num_attention_heads)
        self.lin_v = nn.Linear(self.head_dim, self.num_attention_heads)

        self.gate1 = nn.Linear(self.hidden_size, self.scaled_dim)

        # Check if MoE should be used
        use_moe = config.num_experts is not None and config.num_experts > 0

        # Replace standard Linear layers with LinearMoE where appropriate
        if use_moe:
            self.hx_moe = LinearMoE(
                int(self.num_attention_heads * self.num_attention_heads),
                self.hidden_size,
                bias=True,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                r=config.r,
            )
            self.cx_moe = LinearMoE(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                r=config.r,
            )
            self.x_moe = LinearMoE(
                int(self.num_attention_heads * self.num_attention_heads),
                self.hidden_size,
                bias=True,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                r=config.r,
            )
        else:
            self.hx_moe = nn.Linear(
                int(self.num_attention_heads * self.num_attention_heads),
                self.hidden_size,
                bias=True,
            )
            self.cx_moe = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.x_moe = nn.Linear(
                int(self.num_attention_heads * self.num_attention_heads),
                self.hidden_size,
                bias=True,
            )

        self.mlp = OneMLP(config)

        self.activation = nn.SiLU()
        self.norm = OneRMSNorm(self.hidden_size, bias=True, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.use_flash_attention = getattr(config, "use_flash_attention", False)

        # Use TurboFlashAttention if available and flash attention is enabled
        if self.use_flash_attention:
            if TURBO_MODE:
                self.flash_attention = TurboFlashAttention(causal=True, dropout=dropout)
                logger.info(
                    f"Layer {layer_idx}: Using TurboFlashAttention for maximum performance"
                )
                # Enable combined RoPE + attention operation for maximum performance
                self.use_turbo_combined = True
                logger.info(
                    f"Layer {layer_idx}: Using combined RoPE + Attention for maximum performance"
                )
            else:
                self.flash_attention = OneFlashAttention(config)
                logger.info(f"Layer {layer_idx}: Using standard FlashAttention")
                self.use_turbo_combined = False
        else:
            self.use_turbo_combined = False

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_state: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[DynamicCache] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ):
        _, s, _ = hidden_states.size()
        hx, cx = previous_state

        compute_sequence_len = 1 if self.use_single_token else s

        norm_x = self.norm1(hidden_states)

        # Pre-compute value transformation only once
        value = self.lin3(cx)
        value = rearrange(
            value, "b t (n k) -> b n t k", n=self.num_attention_heads, k=self.head_dim
        )

        # Compute RoPE embedding
        if position_embeddings is None:
            with torch.no_grad() if not self.training else torch.enable_grad():
                cos, sin = self.rotary_pos_emb(value, position_ids)
        else:
            cos, sin = position_embeddings

        value = value[:, :, -compute_sequence_len:, :].contiguous()

        # Update cache if needed
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            norm_x, _ = past_key_value.update(
                norm_x,
                cx[:, -1:, :],
                layer_idx=self.layer_idx,
                cache_kwargs=cache_kwargs,
            )

        # Process only relevant context lengths
        cx = cx[:, -s:, :].contiguous()
        hx = hx[:, -s:, :].contiguous()

        kv_seq = norm_x.shape[1]

        if s < kv_seq:
            kv_seq -= 1

        hidden_state_sum = torch.cumsum(norm_x, dim=1)[:, -s:, :].contiguous()

        # Safe normalization to prevent division by zero
        seq_range = torch.arange(
            1, kv_seq + 1, device=hidden_states.device, dtype=hidden_states.dtype
        ).view(1, -1, 1)

        # print(hidden_state_sum.shape, seq_range.shape)

        hidden_state_sum = hidden_state_sum / (
            seq_range[:, -s:, :] + 1e-10
        )  # Add small epsilon

        # Run transformations in parallel
        single_x = self.lin1(norm_x[:, -compute_sequence_len:, :].contiguous())
        key = self.lin2(hx[:, -compute_sequence_len:, :].contiguous())

        # Reshape with better memory access patterns
        single_x = rearrange(
            single_x,
            "b t (n k) -> b n t k",
            n=self.num_attention_heads,
            k=self.head_dim,
        )
        key = rearrange(
            key, "b t (n k) -> b n t k", n=self.num_attention_heads, k=self.head_dim
        )

        # Apply proper scaling for numerical stability
        single_x = single_x * self.attention_scale

        # Use Flash Attention if available
        if self.use_flash_attention:
            # Apply rotary embeddings using flash attention optimized function
            single_x_rotated, key_rotated = apply_rotary_pos_emb_flash(
                single_x, key, cos, sin
            )
            value_rotated = value[:, :, -compute_sequence_len:, :].contiguous()

            try:
                # Try Flash Attention but with error handling
                single_x_processed, attn_weights = self.flash_attention(
                    self.lin_q(single_x_rotated),
                    self.lin_k(key_rotated),
                    self.lin_v(value_rotated),
                )
            except Exception as e:
                # Fallback to standard attention with added numerical stability
                q = self.lin_q(single_x_rotated)
                k = self.lin_k(key_rotated)
                v = self.lin_v(value_rotated)

                # Compute attention scores with improved numerical stability
                attn_logits = q * k
                attn_max = torch.max(attn_logits, dim=-1, keepdim=True)[0]
                attn_logits = (
                    attn_logits - attn_max
                )  # Subtract max for numerical stability
                attn_weights = torch.softmax(attn_logits, dim=-1, dtype=torch.float32)

                # Check for NaNs in the attention weights
                if torch.isnan(attn_weights).any():
                    # Replace NaNs with uniform attention (fallback)
                    mask = torch.isnan(attn_weights)
                    attn_weights = attn_weights.masked_fill(
                        mask, 1.0 / attn_weights.size(-1)
                    )

                attn_weights = attn_weights.to(dtype=q.dtype)
                single_x_processed = attn_weights * v
        else:
            # Use either turbo combined operation or separate operations
            if (
                self.use_turbo_combined
                and TURBO_MODE
                and compute_sequence_len == 1
                and single_x.is_cuda
            ):
                # Use combined RoPE + attention CUDA kernel for maximum performance
                from one.turbo_ops import turbo_combined_rope_attention

                try:
                    # Fast path: Combined RoPE and attention in a single CUDA kernel
                    single_x_processed = turbo_combined_rope_attention(
                        single_x,
                        key,
                        value,
                        cos,
                        sin,
                        self.lin_q,
                        self.lin_k,
                        self.lin_v,
                        compute_sequence_len,
                    )
                except Exception as e:
                    # Fall back to standard path if CUDA kernel fails
                    logger.warning(
                        f"Turbo combined RoPE + attention failed: {e}. Falling back to standard path."
                    )

                    # Apply rotary embeddings
                    single_x_rotated, key_rotated = apply_rotary_pos_emb_flash(
                        single_x, key, cos, sin
                    )

                    key_rotated = key_rotated[
                        :, :, -compute_sequence_len:, :
                    ].contiguous()
                    value = value[:, :, -compute_sequence_len:, :].contiguous()

                    # Standard attention with numerical stability improvements
                    q = self.lin_q(
                        single_x_rotated[:, :, -compute_sequence_len:, :].contiguous()
                    )
                    k = self.lin_k(key_rotated)
                    v = self.lin_v(value)

                    # Compute attention with numerical stability
                    attn_logits = q * k
                    attn_max = torch.max(attn_logits, dim=-1, keepdim=True)[0].detach()
                    attn_logits = attn_logits - attn_max  # Subtract max for stability

                    # Use float32 for softmax stability and convert back to original dtype
                    attn_weights = torch.softmax(
                        attn_logits, dim=-1, dtype=torch.float32
                    ).to(dtype=q.dtype)

                    # Check for NaNs in the attention weights
                    if torch.isnan(attn_weights).any():
                        # Replace NaNs with uniform attention
                        mask = torch.isnan(attn_weights)
                        attn_weights = attn_weights.masked_fill(
                            mask, 1.0 / attn_weights.size(-1)
                        )

                    single_x_processed = attn_weights * v
            else:
                # Standard path
                # Apply rotary embeddings
                single_x_rotated, key_rotated = apply_rotary_pos_emb_flash(
                    single_x, key, cos, sin
                )

                key_rotated = key_rotated[:, :, -compute_sequence_len:, :].contiguous()
                value = value[:, :, -compute_sequence_len:, :].contiguous()

                # Standard attention with numerical stability improvements
                q = self.lin_q(
                    single_x_rotated[:, :, -compute_sequence_len:, :].contiguous()
                )
                k = self.lin_k(key_rotated)
                v = self.lin_v(value)

                # Compute attention with numerical stability
                attn_logits = q * k
                attn_max = torch.max(attn_logits, dim=-1, keepdim=True)[0].detach()
                attn_logits = attn_logits - attn_max  # Subtract max for stability

                # Use float32 for softmax stability and convert back to original dtype
                attn_weights = torch.softmax(
                    attn_logits, dim=-1, dtype=torch.float32
                ).to(dtype=q.dtype)

                # Check for NaNs in the attention weights
                if torch.isnan(attn_weights).any():
                    # Replace NaNs with uniform attention
                    mask = torch.isnan(attn_weights)
                    attn_weights = attn_weights.masked_fill(
                        mask, 1.0 / attn_weights.size(-1)
                    )

                single_x_processed = attn_weights * v

        # Reshape output from attention
        single_x = rearrange(
            single_x_processed,
            "b n t k -> b t (n k)",
            n=self.num_attention_heads,
            k=self.num_attention_heads,
            t=compute_sequence_len,
        )

        # Compute transformations in parallel with gradient clipping
        hx_moe = torch.clamp(self.hx_moe(single_x.contiguous()), min=-100, max=100)
        x_moe_result = torch.clamp(self.x_moe(single_x), min=-100, max=100)

        # Update hidden states
        hx = hx + hx_moe
        hx = self.norm(hx)
        cx = self.cx_moe(cx)

        # Compute final output
        output = hidden_state_sum * x_moe_result

        # Apply MLP and residual connection
        mlp_output = self.mlp(output)
        output = output + mlp_output
        output = self.norm(output)

        # Apply dropout only during training
        if self.training:
            output = self.dropout(output)

        outputs = (
            output[:, -s:].contiguous(),
            (hx[:, -s:].contiguous(), cx[:, -s:].contiguous()),
            past_key_value,
        )

        if output_attentions:
            outputs += (attn_weights,)
        else:
            outputs += (None,)

        return outputs


class OnePreTrainedModel(PreTrainedModel):
    config_class = OneConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["One"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class OneModel(OnePreTrainedModel):
    def __init__(self, config: OneConfig):
        super(OneModel, self).__init__(config)
        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.rotary_emb = OneRotaryEmbedding(config=config)

        self.num_attention_heads = config.num_attention_heads
        self.transformer_dim = config.transformer_dim

        self.pooling = nn.AdaptiveAvgPool1d(output_size=self.transformer_dim)

        self.decoder_layers = nn.ModuleList(
            [
                One(
                    config,
                    dropout=config.lstm_dropout,
                    layer_idx=i,
                    with_hx=config.with_hx,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.lhs_norm = OneRMSNorm(
            self.transformer_dim, bias=True, eps=config.rms_norm_eps
        )

        self.up_scaler = nn.Linear(
            self.transformer_dim,
            self.hidden_size,
            bias=True,
        )

        # self.activation = nn.SiLU()
        self.norm = OneRMSNorm(self.hidden_size, bias=True, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        b, s = input_ids.size()

        # Handle cache format and conversions
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class."
                )

        # Determine position handling
        past_seen_tokens = 0
        if past_key_values is not None:
            past_seen_tokens = past_key_values.get_seq_length()

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Manage positions
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Apply pooling
        hidden_states = self.pooling(inputs_embeds)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        b, s, _ = hidden_states.size()

        # Initialize hidden states for the decoder layers
        hx = torch.ones(
            (b, s, self.transformer_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        cx = hidden_states.clone()

        if self.training:
            hx = hx.requires_grad_()
            cx = cx.detach()
            cx.requires_grad = True

        next_decoder_cache = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.decoder_layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    (hx, cx),
                    past_key_values,
                    position_embeddings,
                    cache_position,
                    position_ids,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    (hx, cx),
                    past_key_values,
                    position_embeddings,
                    cache_position,
                    position_ids,
                    output_attentions=output_attentions,
                )

            output, (hx, cx), cache, attn_weights = layer_outputs

            hidden_states = hidden_states + output

            if use_cache:
                next_decoder_cache = cache

            if output_attentions:
                all_self_attns += (attn_weights,)

        hidden_states = self.lhs_norm(hidden_states)
        hidden_states = self.up_scaler(hidden_states)
        hidden_states = self.norm(hidden_states)

        if self.training:
            hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class OneForCausalLM(OnePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: OneConfig):
        super().__init__(config)

        self.config = config

        self.model = OneModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_state = outputs[0]

        logits = self.lm_head(hidden_state)

        if len(outputs) > 1 and not self.training:
            past_key_values = outputs[1]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
