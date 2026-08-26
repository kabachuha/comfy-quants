"""Kandinsky 5 model adapter.

Authored from ComfyUI ``comfy/model_detection.py`` and validated against the 
Kandinsky 5 tensor architecture. Kandinsky 5 utilizes a split-transformer 
architecture with separate text and visual blocks.
"""

from __future__ import annotations

from comfy_quants.comfy.stock_dit_contract import stock_dit_artifact_contract_metadata
from comfy_quants.core.policy import QuantPolicy
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.stock_dit_contract import (
    BlockGroup,
    StockDitContract,
    build_stock_dit_graph,
    kept_component,
    linear,
    summarize_stock_dit_graph,
)

CONTRACT_SCHEMA_VERSION = "kandinsky5_static_contract.v1"

# Kandinsky 5 architecture constants
_HIDDEN = 4096           # model_dim
_TEXT_BLOCKS = 4         # num_text_blocks
_VISUAL_BLOCKS = 60      # num_visual_blocks
_FF_DIM = 16384          # ff_dim (feed_forward hidden size)
_OL_DIM = 36864          # ol_dim (outlayer)
_EXIT_DIM = 1024          # exit_dim (final dim)
_VISUAL_EMBED_DIM = 132  # visual_embed_dim


def _dims() -> dict[str, int]:
    return {
        "H": _HIDDEN,
        "FF": _FF_DIM,
        "OL": _OL_DIM,
        "EXIT": _EXIT_DIM,
        "VIS_EMBED": _VISUAL_EMBED_DIM,
        "TEXT_B": _TEXT_BLOCKS,
        "VIS_B": _VISUAL_BLOCKS,
    }


def _text_block_modules() -> tuple:
    p = "text_transformer_blocks.{block}"
    return (
        linear(f"{p}.feed_forward.in_layer", "FF", "H"),
        linear(f"{p}.feed_forward.out_layer", "H", "FF"),
        linear(f"{p}.self_attention.to_key", "H", "H"),
        linear(f"{p}.self_attention.to_query", "H", "H"),
        linear(f"{p}.self_attention.to_value", "H", "H"),
        linear(f"{p}.self_attention.out_layer", "H", "H"),
    )


def _visual_block_modules() -> tuple:
    p = "visual_transformer_blocks.{block}"
    return (
        # Cross Attention
        linear(f"{p}.cross_attention.to_key", "H", "H"),
        linear(f"{p}.cross_attention.to_query", "H", "H"),
        linear(f"{p}.cross_attention.to_value", "H", "H"),
        linear(f"{p}.cross_attention.out_layer", "H", "H"),
        # Feed Forward
        linear(f"{p}.feed_forward.in_layer", "FF", "H"),
        linear(f"{p}.feed_forward.out_layer", "H", "FF"),
        # Self Attention
        linear(f"{p}.self_attention.to_key", "H", "H"),
        linear(f"{p}.self_attention.to_query", "H", "H"),
        linear(f"{p}.self_attention.to_value", "H", "H"),
        linear(f"{p}.self_attention.out_layer", "H", "H"),
        # Visual Modulation
        linear(f"{p}.visual_modulation.out_layer", "OL", "EXIT"),
    )


def _extra_components() -> tuple:
    return (
        kept_component("pooled_text_embeddings", "Linear", "transformer", "text pooling kept high precision"),
        kept_component("text_embeddings", "Linear", "transformer", "text input projection kept high precision"),
        kept_component("time_embeddings", "MLPEmbedder", "transformer", "timestep embedding kept high precision"),
        kept_component("visual_embeddings", "Linear", "transformer", "visual input projection kept high precision"),
        kept_component("out_layer", "LastLayer", "transformer", "final layer kept high precision"),
    )


def build_kandinsky5_static_contract() -> StockDitContract:
    return StockDitContract(
        family="kandinsky5",
        schema_version=CONTRACT_SCHEMA_VERSION,
        preferred_format="fp8_e4m3",
        dims=_dims(),
        block_groups=(
            BlockGroup(prefix="text_transformer_blocks", count=_TEXT_BLOCKS, modules=_text_block_modules()),
            BlockGroup(prefix="visual_transformer_blocks", count=_VISUAL_BLOCKS, modules=_visual_block_modules()),
        ),
        extra_components=_extra_components(),
        metadata={
            "export_name": "Kandinsky 5",
            "architecture": "split_transformer",
            "hidden_size": _HIDDEN,
            "num_text_blocks": _TEXT_BLOCKS,
            "num_visual_blocks": _VISUAL_BLOCKS,
            "ff_dim": _FF_DIM,
            "visual_embed_dim": _VISUAL_EMBED_DIM,
        },
    )


class Kandinsky5Adapter:
    """Adapter for Kandinsky 5 Split-Transformer (Text/Visual blocks)."""

    family = "kandinsky5"
    supported_model_ids = ["kandinskylab/kandinsky-5-pro-video", "kandinskylab/kandinsky-5-lite-image", "kandinskylab/kandinsky-5-lite-video"]

    def inspect(self, source: ModelSource):
        contract = build_kandinsky5_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata("kandinsky5"),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3", *, mixed: bool = False) -> QuantPolicy:
        """Return the layer-selection policy for Kandinsky 5.
        
        If mixed=True, it protects the critical attention layers in the visual 
        transformer blocks to maintain spatial coherence.
        """
        return QuantPolicy(
            name="kandinsky5_mixed" if mixed else "kandinsky5_default",
            algorithm="fp8_static",
            target_dtype=target_dtype,
            include=["text_transformer_blocks.*", "visual_transformer_blocks.*"],
            exclude=["visual_transformer_blocks.*.cross_attention.*"] if mixed else [],
            keep_components=["pooled_text_embeddings", "visual_embeddings"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(Kandinsky5Adapter())
