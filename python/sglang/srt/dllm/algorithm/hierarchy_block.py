import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class HierarchyBlock(DllmAlgorithm):
    """Fast dLLM v2 hierarchical block decoding with token inheritance.

    Attention type switching (matches original HF implementation):
      - Prefill (EXTEND mode):          DECODER (causal)   — handled by model default
      - Denoising iterations:            ENCODER_ONLY (bidirectional) — can use CUDA Graph
      - Final forward (KV cache write):  DECODER (causal)   — always eager (no CUDA Graph)
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.sub_block_size = config.algorithm_config.get("sub_block_size", 8)
        self.token_shift = config.algorithm_config.get("token_shift", 1)
        self.debug = config.algorithm_config.get("debug", False)

        self.last_inherited_token = None
        self.last_block_end_position = None

    @staticmethod
    def _set_attention_type(model_runner: ModelRunner, attn_type: AttentionType):
        """Switch attention type on all language model layers."""
        model = model_runner.model
        # FastDVLMForConditionalGeneration.model is Qwen2Model
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        if layers is None:
            return
        for layer in layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "attn"):
                layer.self_attn.attn.attn_type = attn_type

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        total_len = len(forward_batch.input_ids)
        tokens_per_req = self.block_size

        # Compute per-request block_start (number of non-mask tokens at front)
        start_list = []
        for i in range(batch_size):
            block_start_idx = i * tokens_per_req
            block_end_idx = block_start_idx + tokens_per_req
            block_ids = forward_batch.input_ids[block_start_idx:block_end_idx]
            num_masked = (block_ids == self.mask_id).sum().item()
            start_list.append(tokens_per_req - num_masked)

        # Detect new request (positions start from 0) and clear inheritance
        is_new_request = False
        if (
            hasattr(forward_batch, "positions")
            and forward_batch.positions is not None
        ):
            if forward_batch.positions[0] == 0:
                is_new_request = True
                self.last_inherited_token = None
                self.last_block_end_position = None

        # Handle token inheritance: replace first mask token with inherited AR token
        for i in range(batch_size):
            block_start_idx = i * tokens_per_req
            if (
                forward_batch.input_ids[block_start_idx] == self.mask_id
                and self.last_inherited_token is not None
                and not is_new_request
            ):
                forward_batch.input_ids[block_start_idx] = self.last_inherited_token

        num_sub_blocks = tokens_per_req // self.sub_block_size

        # === Denoising iterations: use bidirectional attention ===
        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)

        for sub_idx in range(num_sub_blocks):
            rel_start = sub_idx * self.sub_block_size
            rel_end = rel_start + self.sub_block_size

            while True:
                # Check if any masks remain in this sub-block across all batches
                any_mask = False
                for i in range(batch_size):
                    base = i * tokens_per_req
                    sub_mask = (
                        forward_batch.input_ids[base + rel_start : base + rel_end]
                        == self.mask_id
                    )
                    if sub_mask.sum().item() > 0:
                        any_mask = True
                        break
                if not any_mask:
                    break

                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                logits_output, can_run_cuda_graph = (
                    out.logits_output,
                    out.can_run_graph,
                )
                full_logits = logits_output.full_logits
                assert full_logits is not None

                # Token shift: shifted[i] predicts token at position i
                if self.token_shift > 0:
                    shifted_full = torch.cat(
                        [full_logits[:1], full_logits[:-1]], dim=0
                    )
                else:
                    shifted_full = full_logits

                # Process each batch item's sub-block
                for i in range(batch_size):
                    base = i * tokens_per_req
                    sub_ids = forward_batch.input_ids[
                        base + rel_start : base + rel_end
                    ]
                    sub_mask = sub_ids == self.mask_id
                    if sub_mask.sum().item() == 0:
                        continue

                    sub_logits = shifted_full[base + rel_start : base + rel_end, :]

                    preds = sub_logits.argmax(dim=-1)
                    probs = F.softmax(sub_logits, dim=-1)
                    conf = probs.gather(
                        dim=-1, index=preds.unsqueeze(-1)
                    ).squeeze(-1)
                    conf = torch.where(
                        sub_mask,
                        conf,
                        torch.tensor(-np.inf, device=conf.device),
                    )

                    unmask = conf > self.threshold
                    if unmask.sum().item() == 0:
                        unmask[conf.argmax()] = True
                    unmask = unmask & sub_mask

                    forward_batch.input_ids[
                        base + rel_start : base + rel_end
                    ] = torch.where(unmask, preds, sub_ids)

        # === Final forward: switch to causal attention for KV cache write ===
        self._set_attention_type(model_runner, AttentionType.DECODER)

        logits_output = model_runner.forward_extend(
            forward_batch, pp_proxy_tensors=None
        )
        if isinstance(logits_output, tuple):
            logits_output = logits_output[0]
        full_logits = logits_output.full_logits

        # Extract inherited token from the last position's AR prediction
        if full_logits is not None:
            # Use the last token of the last batch item
            self.last_inherited_token = full_logits[-1].argmax().item()

        # Update position tracking
        if (
            hasattr(forward_batch, "positions")
            and forward_batch.positions is not None
        ):
            self.last_block_end_position = forward_batch.positions[-1].item()

        # Restore ENCODER_ONLY for next block's denoising iterations
        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)

        # Build per-request next_token_ids list (matching official interface)
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = HierarchyBlock
