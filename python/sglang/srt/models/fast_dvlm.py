# Copyright 2023-2024 SGLang Team
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
"""
Fast-dVLM model for SGLang.

This model combines the Fast-dLLM v2 diffusion language model with the
Qwen2.5-VL vision encoder to support multimodal (image + text) diffusion
generation. The language backbone uses bidirectional attention for block
diffusion decoding, while the vision encoder is identical to Qwen2.5-VL.

Architecture name: Fast_dLLM_Qwen2_5_VLForConditionalGeneration
"""

import logging
import re
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)

from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class FastDVLMForConditionalGeneration(nn.Module):
    """
    Fast-dVLM: Fast Diffusion Vision-Language Model for SGLang.

    Combines:
    - Qwen2.5-VL vision encoder (Qwen2_5_VisionTransformer)
    - Qwen2Model language backbone
    - Full logits return for diffusion decoding

    The dLLM algorithm dynamically switches attention between:
      - DECODER (causal) for prefill and KV cache writes
      - ENCODER_ONLY (bidirectional) for block denoising / draft iterations
    """

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_up_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Weight mapping: HF checkpoint names -> sglang names
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attn.qkv": "attn.qkv_proj",
        },
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.pp_group = get_pp_group()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        # Language model: Qwen2Model
        # NOTE: Do NOT patch attention to ENCODER_ONLY here.
        # The dLLM algorithm dynamically switches between:
        #   - DECODER (causal) for prefill and KV cache writes
        #   - ENCODER_ONLY (bidirectional) for block denoising iterations
        if not self.config.encoder_only:
            self.model = Qwen2Model(
                config,
                quant_config=quant_config,
                prefix=add_prefix("model", prefix),
            )

            # LM head
            if self.pp_group.is_last_rank:
                if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                    self.lm_head = self.model.embed_tokens
                else:
                    self.lm_head = ParallelLMHead(
                        self.config.vocab_size,
                        self.config.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix("lm_head", prefix),
                    )
            else:
                self.lm_head = PPMissingLayer()

            # PP weight tying
            if self.pp_group.world_size > 1 and config.tie_word_embeddings:
                if self.pp_group.is_first_rank:
                    self.pp_group.send(
                        self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                    )
                elif self.pp_group.is_last_rank:
                    emb_token_weight = self.pp_group.recv(
                        size=(config.vocab_size, config.hidden_size),
                        dtype=next(self.model.parameters()).dtype,
                        src=self.pp_group.first_rank,
                    )
                    self.lm_head.weight.copy_(emb_token_weight)
        else:
            self.lm_head = None

        # Vision encoder: same as Qwen2.5-VL
        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
            max_context_len=self.config.max_position_embeddings,
        )

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        # Return full logits for dLLM diffusion decoding
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)

        expected_dim = getattr(self.visual, "embed_dim", -1)
        if expected_dim == -1:
            vision_conf = self.config.vision_config
            expected_dim = getattr(
                vision_conf, "embed_dim", getattr(vision_conf, "hidden_size", -1)
            )

        raw_patch_dim = 1176

        if pixel_values.dim() == 2:
            current_dim = pixel_values.shape[-1]
            if current_dim == expected_dim:
                return pixel_values
            if current_dim != raw_patch_dim:
                return pixel_values

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, video_grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    def post_process(
        self,
        inputs_embeds,
        modalities: List[Modality],
        embeddings: List[torch.Tensor],
        indices: List[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        new_embeddings = []
        for i, (modality, embedding, index) in enumerate(
            zip(modalities, embeddings, indices)
        ):
            if embedding is None or index is None:
                continue
            new_embeddings.append(embedding)
        return new_embeddings, forward_batch

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Run forward pass for Fast-dVLM.

        Uses MRoPE positions (same as Qwen2.5-VL) for all modes.
        Prompt prefill (EXTEND): vision embeddings injected via general_mm_embed_routine.
        Block decode (DLLM_EXTEND): language model only, no multimodal processing.
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_dllm_extend()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        if (
            forward_batch.forward_mode.is_dllm_extend()
            and not forward_batch.contains_image_inputs()
        ):
            # dLLM block decode: vision tokens already prefilled in KV cache.
            # Only skip multimodal processing when there are no image inputs
            # in this batch (i.e., after the initial prompt prefill).
            hidden_states = self.model(
                input_ids,
                positions,
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        else:
            # Prompt prefill (with images) or decode: process multimodal inputs.
            # For the first DLLM_EXTEND pass that includes the prompt with
            # images, this path ensures vision embeddings are injected.
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    _lora_pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self._lora_pattern.match(module_name))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.pp_group.is_last_rank and "model.embed_tokens.weight" in name:
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if (
                    "visual" in name
                    and "up_proj" not in name
                    and "gate_proj" not in name
                ):
                    continue
                name = name.replace(weight_name, param_name)
                layer_id = get_layer_id(name)
                if (
                    layer_id is not None
                    and hasattr(self, "model")
                    and hasattr(self.model, "start_layer")
                    and (
                        layer_id < self.model.start_layer
                        or layer_id >= self.model.end_layer
                    )
                ):
                    continue

                if name.endswith(".bias") and name not in params_dict:
                    continue
                if (
                    self.config.encoder_only or self.config.language_only
                ) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                try:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                    else:
                        continue
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


# Register with HuggingFace architecture name from config.json
FastDVLMForConditionalGeneration.__name__ = (
    "Fast_dLLM_Qwen2_5_VLForConditionalGeneration"
)

EntryClass = FastDVLMForConditionalGeneration
