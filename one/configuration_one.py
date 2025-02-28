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

from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from typing import Optional


@dataclass
class OneConfig(PretrainedConfig):
    vocab_size: int = 28886
    num_hidden_layers: int = 32
    hidden_size: int = 1024
    transformer_dim: int = 512
    intermediate_size: int = 4096
    conv_filters: int = 4
    num_attention_heads: int = 32
    scale_factor: int = 4
    dropout: float = 0.02
    lstm_dropout: float = 0.01
    conv_activation: str = "silu"
    rms_norm_eps: float = 1e-6
    use_cache: bool = False
    with_hx: bool = True
    head_dim: int = 32
    max_model_length: int = 32768
    pad_token_id: int = 0
    lm_head_embedding: bool = False
    num_of_tokens_to_predict: int = 1
    rope_theta = 10000.0
    rope_scaling = None
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True
    hidden_act: str = "gelu"
    mlp_bias: bool = False
    use_flash_attention: bool = False
    use_single_token: bool = False

    # Linear MoE
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = 2
    r: Optional[int] = 64

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )

    def __post_init__(self):
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
