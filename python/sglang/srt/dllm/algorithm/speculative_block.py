"""Speculative block decoding for dLLM.

Two-phase design per block:
  Phase 1 — Draft (bidirectional):  ENCODER_ONLY attention, CUDA Graph
  Phase 2 — Verify (causal / AR):   DECODER attention, eager forward

After AR verification the longest matching prefix is accepted and
rejected KV slots are freed by the scheduler.
"""

import logging
from typing import List, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class SpeculativeBlock(DllmAlgorithm):
    """Speculative block decoding with draft-then-verify.

    Attention type switching (matches original HF implementation):
      - Prefill (EXTEND mode):   DECODER (causal)   — handled by model default
      - Draft phase:              ENCODER_ONLY (bidirectional) — CUDA Graph
      - Verify phase:             DECODER (causal)   — eager (forward_extend)
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.token_shift = config.algorithm_config.get("token_shift", 1)
        self.debug = config.algorithm_config.get("debug", False)
        self.last_inherited_token = None
        self.last_block_end_position = None

    @staticmethod
    def _set_attention_type(model_runner: ModelRunner, attn_type: AttentionType):
        """Switch attention type on all language model layers."""
        model = model_runner.model
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
        tokens_per_req = self.block_size
        total_len = len(forward_batch.input_ids)

        # Compute per-request block_start (number of non-mask tokens at front)
        start_list = []
        for i in range(batch_size):
            base = i * tokens_per_req
            block_ids = forward_batch.input_ids[base : base + tokens_per_req]
            num_masked = (block_ids == self.mask_id).sum().item()
            start_list.append(tokens_per_req - num_masked)

        # --- detect new request & clear state ---
        is_new_request = False
        if (
            hasattr(forward_batch, "positions")
            and forward_batch.positions is not None
        ):
            if forward_batch.positions[0] == 0:
                is_new_request = True
                self.last_inherited_token = None
                self.last_block_end_position = None

        # --- place inherited token at position 0 of an all-mask block ---
        for i in range(batch_size):
            base = i * tokens_per_req
            if (
                forward_batch.input_ids[base] == self.mask_id
                and self.last_inherited_token is not None
                and not is_new_request
            ):
                forward_batch.input_ids[base] = self.last_inherited_token

        # ==============================================================
        # Phase 1 – Draft  (bidirectional attention, CUDA Graph)
        # ==============================================================
        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        can_run_cuda_graph = out.can_run_graph
        draft_logits = out.logits_output.full_logits
        assert draft_logits is not None

        # token-shift: shifted[i] predicts token at position i
        if self.token_shift > 0:
            shifted = torch.cat([draft_logits[:1], draft_logits[:-1]], dim=0)
        else:
            shifted = draft_logits

        # materialize predictions before the output buffer is overwritten
        draft_preds = shifted.argmax(dim=-1)

        # fill mask positions with draft predictions
        mask_pos = forward_batch.input_ids == self.mask_id
        forward_batch.input_ids[mask_pos] = draft_preds[mask_pos]

        # ==============================================================
        # Phase 2 – Verify  (causal attention, eager forward)
        # ==============================================================
        self._set_attention_type(model_runner, AttentionType.DECODER)

        verify_out = model_runner.forward_extend(
            forward_batch, pp_proxy_tensors=None
        )
        if isinstance(verify_out, tuple):
            verify_out = verify_out[0]
        logits_output = verify_out
        verify_logits = logits_output.full_logits
        assert verify_logits is not None

        ar_tokens = verify_logits.argmax(dim=-1)

        # --- Restore ENCODER_ONLY for next block's draft phase ---
        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)

        # --- Per-request AR comparison and acceptance ---
        next_token_ids_list = []
        for i in range(batch_size):
            base = i * tokens_per_req
            block_start = base + start_list[i]

            # ar[j] should equal block[j+1] for sequential acceptance
            accepted_num = 0
            for j in range(base, base + tokens_per_req - 1):
                if ar_tokens[j] == forward_batch.input_ids[j + 1]:
                    accepted_num += 1
                else:
                    break
            accepted_num += 1  # correction / next-token prediction
            accepted_num = min(accepted_num, tokens_per_req)

            # determine output tokens
            if accepted_num >= tokens_per_req:
                output_count = tokens_per_req - start_list[i]
                self.last_inherited_token = ar_tokens[base + tokens_per_req - 1].item()
            else:
                output_count = max(accepted_num - start_list[i], 0)
                self.last_inherited_token = ar_tokens[base + accepted_num - 1].item()

            next_token_ids = forward_batch.input_ids[
                block_start : block_start + output_count
            ]
            next_token_ids_list.append(next_token_ids)

            if self.debug:
                logger.info(
                    f"[SpeculativeBlock] req={i} total={tokens_per_req} "
                    f"blk_start={start_list[i]} accepted={accepted_num} "
                    f"output={output_count}"
                )

        # --- position tracking ---
        if (
            hasattr(forward_batch, "positions")
            and forward_batch.positions is not None
        ):
            self.last_block_end_position = forward_batch.positions[-1].item()

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = SpeculativeBlock
