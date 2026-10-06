"""Kandinsky 6 model adapter.

Authored from ComfyUI ``comfy/ldm/kandinsky6/model.py``, the release
``core_contract`` / ``detection`` and the Pro ``transformer_pro.json`` module
config, and validated against the joint video+audio tensor architecture.
Kandinsky 6 fuses one video and one audio transformer stream per block
(``visual_blocks``) around shared text-encoder streams.
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

CONTRACT_SCHEMA_VERSION = "kandinsky6_static_contract.v1"

# Kandinsky 6 (Pro) architecture constants, from DIT_CONFIG / transformer_pro.
_H = 4096            # model_dim (video)
_HA = 2048           # model_dim_a (audio)
_FF = 16384          # ff_dim (video)
_FFA = 7168          # ff_dim_a (audio)
_TD = 1024           # time_dim (video)
_TDA = 1024          # time_dim_a (audio)
_VIS_EMBED = 132     # visual_embed_dim = (2*16 + 1) * prod(patch_size=1,2,2)
_VIS_MOD = 9 * _H    # videoT.visual_modulation out = 9 * model_dim
_VIS_MOD_A = 9 * _HA  # audioT.visual_modulation out = 9 * model_dim_a
_VA_MOD = 2 * _H + _HA  # va_modulation out (cross_gates) = 2*H + HA
_AV_MOD = 2 * _HA + _H  # av_modulation out (cross_gates) = 2*HA + H
_TEXT_B = 4          # num_text_blocks
_VIS_B = 60          # num_visual_blocks


def _dims() -> dict[str, int]:
    return {
        "H": _H, "HA": _HA, "FF": _FF, "FFA": _FFA,
        "TD": _TD, "TDA": _TDA, "VIS_EMBED": _VIS_EMBED,
        "VIS_MOD": _VIS_MOD, "VIS_MOD_A": _VIS_MOD_A,
        "VA_MOD": _VA_MOD, "AV_MOD": _AV_MOD,
        "TEXT_B": _TEXT_B, "VIS_B": _VIS_B,
    }


def _video_text_block_modules() -> tuple:
    p = "video_text_transformer_blocks.{block}"
    return (
        linear(f"{p}.feed_forward.in_layer", "FF", "H"),
        linear(f"{p}.feed_forward.out_layer", "H", "FF"),
        linear(f"{p}.self_attention.to_query", "H", "H"),
        linear(f"{p}.self_attention.to_key", "H", "H"),
        linear(f"{p}.self_attention.to_value", "H", "H"),
        linear(f"{p}.self_attention.out_layer", "H", "H"),
    )


def _audio_text_block_modules() -> tuple:
    p = "audio_text_transformer_blocks.{block}"
    return (
        linear(f"{p}.feed_forward.in_layer", "FFA", "HA"),
        linear(f"{p}.feed_forward.out_layer", "HA", "FFA"),
        linear(f"{p}.self_attention.to_query", "HA", "HA"),
        linear(f"{p}.self_attention.to_key", "HA", "HA"),
        linear(f"{p}.self_attention.to_value", "HA", "HA"),
        linear(f"{p}.self_attention.out_layer", "HA", "HA"),
    )


def _visual_block_modules() -> tuple:
    p = "visual_blocks.{block}"
    return (
        # video decoder
        linear(f"{p}.videoT.self_attention.to_query", "H", "H"),
        linear(f"{p}.videoT.self_attention.to_key", "H", "H"),
        linear(f"{p}.videoT.self_attention.to_value", "H", "H"),
        linear(f"{p}.videoT.self_attention.out_layer", "H", "H"),
        linear(f"{p}.videoT.cross_attention.to_query", "H", "H"),
        linear(f"{p}.videoT.cross_attention.to_key", "H", "H"),
        linear(f"{p}.videoT.cross_attention.to_value", "H", "H"),
        linear(f"{p}.videoT.cross_attention.out_layer", "H", "H"),
        linear(f"{p}.videoT.feed_forward.in_layer", "FF", "H"),
        linear(f"{p}.videoT.feed_forward.out_layer", "H", "FF"),
        linear(f"{p}.videoT.visual_modulation.out_layer", "VIS_MOD", "TD"),
        # audio decoder
        linear(f"{p}.audioT.self_attention.to_query", "HA", "HA"),
        linear(f"{p}.audioT.self_attention.to_key", "HA", "HA"),
        linear(f"{p}.audioT.self_attention.to_value", "HA", "HA"),
        linear(f"{p}.audioT.self_attention.out_layer", "HA", "HA"),
        linear(f"{p}.audioT.cross_attention.to_query", "HA", "HA"),
        linear(f"{p}.audioT.cross_attention.to_key", "HA", "HA"),
        linear(f"{p}.audioT.cross_attention.to_value", "HA", "HA"),
        linear(f"{p}.audioT.cross_attention.out_layer", "HA", "HA"),
        linear(f"{p}.audioT.feed_forward.in_layer", "FFA", "HA"),
        linear(f"{p}.audioT.feed_forward.out_layer", "HA", "FFA"),
        linear(f"{p}.audioT.visual_modulation.out_layer", "VIS_MOD_A", "TDA"),
        # video -> audio cross attention (query from video, key/value from audio)
        linear(f"{p}.va_cross_attention.to_query", "H", "H"),
        linear(f"{p}.va_cross_attention.to_key", "H", "HA"),
        linear(f"{p}.va_cross_attention.to_value", "H", "HA"),
        linear(f"{p}.va_cross_attention.out_layer", "H", "H"),
        # audio -> video cross attention (query from audio, key/value from video)
        linear(f"{p}.av_cross_attention.to_query", "HA", "HA"),
        linear(f"{p}.av_cross_attention.to_key", "HA", "H"),
        linear(f"{p}.av_cross_attention.to_value", "HA", "H"),
        linear(f"{p}.av_cross_attention.out_layer", "HA", "HA"),
        # cross-stream AdaLN (cross_gates layout)
        linear(f"{p}.va_modulation.out_layer", "VA_MOD", "TD"),
        linear(f"{p}.av_modulation.out_layer", "AV_MOD", "TDA"),
    )


def _extra_components() -> tuple:
    return (
        kept_component("visual_embeddings", "Linear", "transformer", "video input projection kept high precision"),
        kept_component("visual_token_type_embeddings", "Embedding", "transformer", "I2VA token-type embeddings kept high precision"),
        kept_component("video_time_embeddings", "MLPEmbedder", "transformer", "video timestep embedding kept high precision"),
        kept_component("video_text_embeddings", "Linear", "transformer", "video text input projection kept high precision"),
        kept_component("video_pooled_text_embeddings", "Linear", "transformer", "video pooled text projection kept high precision"),
        kept_component("out_layer", "LastLayer", "transformer", "video final layer kept high precision"),
        kept_component("audio_embeddings", "Linear", "transformer", "audio input projection kept high precision"),
        kept_component("audio_time_embeddings", "MLPEmbedder", "transformer", "audio timestep embedding kept high precision"),
        kept_component("audio_text_embeddings", "Linear", "transformer", "audio text input projection kept high precision"),
        kept_component("audio_pooled_text_embeddings", "Linear", "transformer", "audio pooled text projection kept high precision"),
        kept_component("audio_outLayer", "Linear", "transformer", "audio final layer kept high precision"),
    )


def build_kandinsky6_static_contract() -> StockDitContract:
    return StockDitContract(
        family="kandinsky6",
        schema_version=CONTRACT_SCHEMA_VERSION,
        preferred_format="int8_tensorwise",
        dims=_dims(),
        block_groups=(
            BlockGroup(prefix="video_text_transformer_blocks", count=_TEXT_B, modules=_video_text_block_modules()),
            BlockGroup(prefix="audio_text_transformer_blocks", count=_TEXT_B, modules=_audio_text_block_modules()),
            BlockGroup(prefix="visual_blocks", count=_VIS_B, modules=_visual_block_modules()),
        ),
        extra_components=_extra_components(),
        metadata={
            "export_name": "Kandinsky 6",
            "architecture": "fused_av_transformer",
            "model_dim": _H,
            "model_dim_a": _HA,
            "num_text_blocks": _TEXT_B,
            "num_visual_blocks": _VIS_B,
            "ff_dim": _FF,
            "ff_dim_a": _FFA,
            "is_multimodal": True,
        },
    )


class Kandinsky6Adapter:
    """Adapter for the Kandinsky 6 joint video+audio DiT."""

    family = "kandinsky6"
    supported_model_ids = ["kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"]

    def inspect(self, source: ModelSource):
        contract = build_kandinsky6_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata("kandinsky6"),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "int8_tensorwise", *, mixed: bool = False) -> QuantPolicy:
        """Return the layer-selection policy for Kandinsky 6.

        With mixed=True the cross-stream attention is protected to preserve
        video/audio coherence.
        """
        return QuantPolicy(
            name="kandinsky6_mixed" if mixed else "kandinsky6_default",
            algorithm="int8_tensorwise",
            target_dtype=target_dtype,
            include=[
                "video_text_transformer_blocks.*",
                "audio_text_transformer_blocks.*",
                "visual_blocks.*",
            ],
            exclude=[
                "visual_blocks.*.va_cross_attention.*",
                "visual_blocks.*.av_cross_attention.*",
            ] if mixed else [],
            keep_components=["visual_embeddings", "audio_embeddings"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(Kandinsky6Adapter())
